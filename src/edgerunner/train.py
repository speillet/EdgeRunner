"""Training entry point: `just train key=value ...` (keys of configs/train.yaml)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
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

REPO = Path(__file__).resolve().parents[2]


def plain(node: Any) -> dict[str, Any]:
    """Config node → resolved plain dict."""
    return cast(dict[str, Any], OmegaConf.to_container(node, resolve=True))


def build_loaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    return make_loaders(
        cfg.data_root, plain(cfg.model), gsd=cfg.gsd, seed=cfg.seed, **plain(cfg.data)
    )


def provenance(data_root: str) -> dict[str, Any]:
    """Versions a run depends on: git commits (repo, sam_road) and hashes of the data index."""

    def git(*args: str) -> str:
        try:
            run = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
        except OSError:  # no git on this machine
            return ""
        return run.stdout.strip()

    out: dict[str, Any] = {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "sam_road": git("-C", "third_party/sam_road", "rev-parse", "HEAD"),
    }
    for name in ("tiles", "windows"):
        index = Path(data_root, "index", f"{name}.parquet").read_bytes()
        out[f"{name}_sha256"] = hashlib.sha256(index).hexdigest()[:16]
    return out


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from .models.sam_road import SAMRoadModule

    pl.seed_everything(cfg.seed, workers=True)
    torch.backends.cudnn.benchmark = True  # fixed input shapes
    torch.set_float32_matmul_precision("high")
    out = HydraConfig.get().runtime.output_dir
    train_dl, val_dl = build_loaders(cfg)
    model = SAMRoadModule(
        plain(cfg.model),
        lr_milestones=list(cfg.lr_milestones),
        provenance=provenance(cfg.data_root),
    )
    ckpt = ModelCheckpoint(
        f"{out}/ckpt",
        "{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=3,
        save_last=True,
    )
    trainer = pl.Trainer(
        **plain(cfg.trainer),
        logger=TensorBoardLogger(out, name="tb", version="", default_hp_metric=False),
        callbacks=[ckpt, LearningRateMonitor("step")],
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
    if ckpt.best_model_path:  # none under fast_dev_run
        # score of the run (one fold of the cross-validation): best checkpoint on the val folds
        (metrics,) = trainer.validate(model, val_dl, ckpt_path="best", verbose=False)
        if trainer.is_global_zero:
            folds = [str(f) for f in cfg.data.val_folds]
            summary = {"val_folds": folds, "best_ckpt": ckpt.best_model_path}
            Path(out, "metrics.json").write_text(json.dumps({**summary, **metrics}, indent=2))


if __name__ == "__main__":
    main()
