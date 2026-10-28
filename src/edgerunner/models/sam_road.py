"""SAM-Road as a Hydra-configured LightningModule.

Subclass of upstream `SAMRoad` (network, losses, metrics, test step and optimizer groups reused).
Overridden: batch casting (the dataset ships uint8), train / val steps (topology loss safe on
road-less batches, no pdb, images logged to TensorBoard instead of wandb tables) and the LR
schedule (milestones from the config instead of a hard-coded epoch 9).
"""

from __future__ import annotations

import contextlib
import io
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import addict
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from lightning.pytorch.loggers import TensorBoardLogger

from .._sam_road import sam_model

SAMRoad = sam_model().SAMRoad
log = logging.getLogger(__name__)


class SAMRoadModule(SAMRoad):
    def __init__(
        self,
        cfg: Mapping[str, Any],
        lr_milestones: Sequence[int] = (9,),
        lr_gamma: float = 0.1,
        viz_n: int = 4,
        viz_size: int = 256,
        provenance: Mapping[str, Any] | None = None,
    ):
        """`cfg`: upstream SAM-Road config keys (SAM_VERSION, SAM_CKPT_PATH, PATCH_SIZE, ...).

        `provenance` (code and index versions, see `train.provenance`) is only stored in the
        hyper-parameters, hence in every checkpoint and in tb/hparams.yaml.
        """
        with contextlib.redirect_stdout(io.StringIO()):  # upstream pprints every parameter name
            super().__init__(addict.Dict(cfg))
        n_fresh = sum(1 for k, _ in self.named_parameters() if k not in self.matched_param_names)
        log.info(
            "SAM checkpoint %s: %d tensors loaded, %d fresh",
            cfg["SAM_CKPT_PATH"],
            len(self.matched_param_names),
            n_fresh,
        )
        self.save_hyperparameters()

    def on_after_batch_transfer(self, batch: dict[str, Any], dataloader_idx: int) -> dict[str, Any]:
        for key in ("rgb", "keypoint_mask", "road_mask"):
            batch[key] = batch[key].float()  # rgb 0-255, masks {0, 1}
        return batch

    def _losses(self, batch: dict[str, Any]) -> tuple[torch.Tensor, ...]:
        valid = batch["valid"]
        mask_logits, mask_scores, topo_logits, topo_scores = self(
            batch["rgb"], batch["graph_points"], batch["pairs"], valid
        )
        gt_masks = torch.stack([batch["keypoint_mask"], batch["road_mask"]], dim=3)
        mask_loss = self.mask_criterion(mask_logits, gt_masks)
        topo = self.topo_criterion(topo_logits, batch["connected"].float().unsqueeze(-1))
        w = valid.float().unsqueeze(-1)
        topo_loss = (topo * w).sum() / w.sum().clamp(min=1.0)  # 0, not NaN, without any road
        return mask_loss, topo_loss, mask_scores, topo_scores

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        mask_loss, topo_loss, _, _ = self._losses(batch)
        loss = mask_loss + topo_loss
        self.log_dict(
            {"train_mask_loss": mask_loss, "train_topo_loss": topo_loss, "train_loss": loss},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=batch["rgb"].shape[0],
        )
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        mask_loss, topo_loss, mask_scores, topo_scores = self._losses(batch)
        self.log_dict(
            {
                "val_mask_loss": mask_loss,
                "val_topo_loss": topo_loss,
                "val_loss": mask_loss + topo_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["rgb"].shape[0],
        )
        if batch_idx == 0:
            self._log_images(batch, mask_scores)

        self.keypoint_iou.update(mask_scores[..., 0], batch["keypoint_mask"].int())
        self.road_iou.update(mask_scores[..., 1], batch["road_mask"].int())
        valid = batch["valid"].int()
        topo_gt = (1 - valid) * -1 + valid * batch["connected"].int()  # -1 = ignored padding
        self.topo_f1.update(topo_scores, topo_gt.unsqueeze(-1))

    def _log_images(self, batch: dict[str, Any], mask_scores: torch.Tensor) -> None:
        """Rows of rgb | gt keypoint | gt road | pred keypoint | pred road."""
        tbs = [lg for lg in self.loggers if isinstance(lg, TensorBoardLogger)]
        if not tbs or not self.trainer.is_global_zero:
            return
        n = min(self.hparams.viz_n, batch["rgb"].shape[0])
        rgb = batch["rgb"][:n].permute(0, 3, 1, 2) / 255.0
        maps = [
            batch["keypoint_mask"][:n],
            batch["road_mask"][:n],
            mask_scores[:n, ..., 0],
            mask_scores[:n, ..., 1],
        ]
        cols = [rgb] + [m.unsqueeze(1).expand(-1, 3, -1, -1) for m in maps]
        size = (self.hparams.viz_size,) * 2
        tiles = torch.stack([F.interpolate(c.float(), size=size, mode="area") for c in cols], 1)
        grid = torchvision.utils.make_grid(tiles.flatten(0, 1), nrow=len(cols), pad_value=1.0)
        for tb in tbs:
            tb.experiment.add_image("val/masks", grid.clamp(0, 1).cpu(), self.global_step)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = super().configure_optimizers()["optimizer"]  # upstream param groups / LRs
        sched = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(self.hparams.lr_milestones), gamma=self.hparams.lr_gamma
        )
        return {"optimizer": optimizer, "lr_scheduler": sched}
