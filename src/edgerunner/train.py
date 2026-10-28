"""Training entry point: `just train key=value ...` (keys of configs/train.yaml)."""

from __future__ import annotations

import hydra
import lightning.pytorch as pl
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .data.dataset import make_loaders


def build_loaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    data = OmegaConf.to_container(cfg.data, resolve=True)
    sam_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    return make_loaders(cfg.data_root, sam_cfg, gsd=cfg.gsd, seed=cfg.seed, **data)


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from .models.sam_road import SAMRoadModule

    pl.seed_everything(cfg.seed, workers=True)
    torch.backends.cudnn.benchmark = True  # fixed input shapes
    torch.set_float32_matmul_precision("high")
    out = HydraConfig.get().runtime.output_dir
    if cfg.clearml and rank_zero_only.rank == 0:  # mirrors the TensorBoard logs
        from clearml import Task

        task = Task.init(
            project_name=cfg.clearml_project,
            task_name=cfg.run_name,
            auto_connect_frameworks={"pytorch": False, "tensorboard": True},
            reuse_last_task_id=False,
        )
        task.connect(OmegaConf.to_container(cfg, resolve=True), name="hydra")

    train_dl, val_dl = build_loaders(cfg)
    model = SAMRoadModule(
        OmegaConf.to_container(cfg.model, resolve=True), lr_milestones=list(cfg.lr_milestones)
    )
    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer, resolve=True),
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
    trainer.fit(model, train_dl, val_dl, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    main()
