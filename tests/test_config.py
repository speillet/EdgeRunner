"""configs/train.yaml: metre settings resolve at the run GSD, loaders build from it."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from edgerunner.train import build_loaders

CONFIGS = str(Path(__file__).resolve().parents[1] / "configs")


def cfg(*overrides: str):  # noqa: ANN201
    with initialize_config_dir(CONFIGS, version_base="1.3"):
        return compose("train", overrides=list(overrides))


def test_metres_follow_gsd() -> None:
    m = OmegaConf.to_container(cfg().model, resolve=True)
    assert (m["PATCH_SIZE"], m["NEIGHBOR_RADIUS"], m["ROAD_NMS_RADIUS"]) == (1024, 160, 40)
    m = OmegaConf.to_container(cfg("gsd=0.8", "patch_size=512").model, resolve=True)
    assert (m["PATCH_SIZE"], m["NEIGHBOR_RADIUS"], m["ROAD_NMS_RADIUS"]) == (512, 80, 20)


def test_loaders_from_config(toy: Path) -> None:
    c = cfg(f"data_root={toy}", "patch_size=256", "data.num_workers=0", "data.batch_size=2",
            "data.val_max_patches=2", "model.TOPO_SAMPLE_NUM=16")
    train_dl, val_dl = build_loaders(c)
    batch = next(iter(train_dl))
    assert batch["rgb"].shape == (2, 256, 256, 3) and batch["pairs"].shape[1] == 16
    assert len(val_dl.dataset) == 2
