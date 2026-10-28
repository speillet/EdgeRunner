"""Training entry point: `just train key=value ...` (keys of configs/train.yaml)."""

from __future__ import annotations

from typing import Any, cast

import hydra
import lightning.pytorch as pl
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .data.dataset import make_loaders


def plain(node: Any) -> dict[str, Any]:
    """Config node → resolved plain dict."""
    return cast(dict[str, Any], OmegaConf.to_container(node, resolve=True))


def build_loaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    return make_loaders(
        cfg.data_root, plain(cfg.model), gsd=cfg.gsd, seed=cfg.seed, **plain(cfg.data)
    )


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from .models.sam_road import SAMRoadModule

    pl.seed_everything(cfg.seed, workers=True)
    torch.backends.cudnn.benchmark = True  # fixed input shapes
    torch.set_float32_matmul_precision("high")
    out = HydraConfig.get().runtime.output_dir
    train_dl, val_dl = build_loaders(cfg)
    model = SAMRoadModule(plain(cfg.model), lr_milestones=list(cfg.lr_milestones))
    trainer = pl.Trainer(
        **plain(cfg.trainer),
        logger=TensorBoardLogger(out, name="tb", version="", default_hp_metric=False),
        callbacks=[
            ModelCheckpoint(
                f"{out}/ckpt",
                "{epoch:02d}-{val_loss:.4f}",
                monitor="val_loss",
                save_top_k=3,
                save_last=True,
            ),
            LearningRateMonitor("step"),
        ],
    )
    if cfg.clearml and trainer.is_global_zero:  # mirrors the TensorBoard logs
        from clearml import Task

        task = Task.init(
            project_name=cfg.clearml_project,
            task_name=cfg.run_name,
            auto_connect_frameworks={"pytorch": False, "tensorboard": True},
            reuse_last_task_id=False,
        )
        task.connect(plain(cfg), name="hydra")
    trainer.fit(model, train_dl, val_dl, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    main()
