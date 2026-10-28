"""SAMRoadModule on CPU at 256 px with a stub SAM checkpoint (only the position embedding)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from edgerunner.data.dataset import collate

from .test_dataset import make, sam_cfg

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def module(tmp_path_factory: pytest.TempPathFactory):  # noqa: ANN201
    from edgerunner.models.sam_road import SAMRoadModule

    ckpt = tmp_path_factory.mktemp("sam") / "stub.pth"
    torch.save({"image_encoder.pos_embed": torch.zeros(1, 64, 64, 768)}, ckpt)
    cfg = dict(sam_cfg(256, 0.4), SAM_VERSION="vit_b", SAM_CKPT_PATH=str(ckpt), PATCH_SIZE=256,
               BASE_LR=1e-3, ENCODER_LR_FACTOR=0.1, TOPONET_VERSION="normal")
    return SAMRoadModule(cfg, lr_milestones=[3, 6])


def test_losses_finite_and_safe_without_roads(module, toy: Path) -> None:  # noqa: ANN001
    batch = collate([make(toy, ["0"])[i] for i in range(2)])
    batch = module.on_after_batch_transfer(batch, 0)
    assert batch["rgb"].dtype == torch.float32
    mask_loss, topo_loss, scores, _ = module._losses(batch)
    assert torch.isfinite(mask_loss) and torch.isfinite(topo_loss) and topo_loss > 0
    assert scores.shape == (2, 256, 256, 2)

    batch["valid"] = torch.zeros_like(batch["valid"])  # road-less batch
    _, topo_loss, _, _ = module._losses(batch)
    assert topo_loss.item() == 0.0


def test_schedule_from_config(module) -> None:  # noqa: ANN001
    sched = module.configure_optimizers()["lr_scheduler"]
    assert sorted(sched.milestones) == [3, 6]
    assert module.hparams.cfg["PATCH_SIZE"] == 256
