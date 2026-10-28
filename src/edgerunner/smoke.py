"""Dataset smoke test: PNG overlays, label stats, DataLoader throughput.

    just smoke data_root=data/toy                 # options before the Hydra overrides:
    just smoke --out outputs/smoke --n 32 data_root=/var/nvme/me/edgerunner data.num_workers=10
Overlay: road mask red, keypoints green, sampled graph points yellow, connected pairs white.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from hydra import compose, initialize_config_dir

from .data.dataset import IGNSamRoadDataset
from .train import build_loaders

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def overlay(item: dict[str, Any], path: Path, max_side: int = 1024) -> None:
    bgr = item["rgb"].numpy()[..., ::-1].copy()
    for key, color, alpha in (("road_mask", (0, 0, 255), 0.5), ("keypoint_mask", (0, 255, 0), 0.6)):
        m = item[key].numpy() > 0
        bgr[m] = ((1 - alpha) * bgr[m] + alpha * np.array(color)).astype(np.uint8)
    if item["valid"].any():
        pts = item["graph_points"].numpy().astype(int)
        ok = (item["valid"] & item["connected"]).numpy()
        for s, t in item["pairs"].numpy()[ok]:
            cv2.line(bgr, tuple(pts[s]), tuple(pts[t]), (255, 255, 255), 1)
        for p in pts:
            cv2.circle(bgr, tuple(p), 3, (0, 255, 255), -1)
    if bgr.shape[0] > max_side:
        bgr = cv2.resize(bgr, (max_side, max_side), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), bgr)


def main() -> None:
    p = argparse.ArgumentParser(description="Dataset smoke test.")
    p.add_argument("--out", default="outputs/smoke")
    p.add_argument("--n", type=int, default=64, help="samples per split for the stats")
    p.add_argument("--overlays", type=int, default=8)
    p.add_argument("--batches", type=int, default=50, help="train batches timed")
    p.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. data_root=data/toy")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(str(CONFIGS), version_base="1.3"):
        cfg = compose("train", overrides=a.overrides)
    train_dl, val_dl = build_loaders(cfg)

    for name, dl in (("train", train_dl), ("val", val_dl)):
        ds = dl.dataset
        assert isinstance(ds, IGNSamRoadDataset)
        n = min(a.n, len(ds))
        t0, empty, points, pos = time.time(), 0, [], []
        for i in range(n):
            item = ds[i]
            v = item["valid"]
            empty += int(not v.any())
            points.append(item["graph_points"].shape[0] if v.any() else 0)
            pos.append(item["connected"][v].float().mean().item() if v.any() else np.nan)
            if i < a.overlays:
                overlay(
                    item,
                    out / f"{name}_{i}_{item['tile_id']}_{item['row_off']}_"
                    f"{item['col_off']}_r{item['rot']}.png",
                )
        print(
            f"{name}: {(time.time() - t0) / max(n, 1) * 1e3:.0f} ms/sample (1 process), "
            f"{empty}/{n} patches without road, {np.mean(points):.0f} graph points/patch, "
            f"{np.nanmean(pos):.2f} of valid pairs connected"
        )

    if a.batches:
        it = iter(train_dl)
        next(it)  # worker start-up excluded
        t0 = time.time()
        for _ in range(a.batches):
            next(it)
        rate = a.batches * cfg.data.batch_size / (time.time() - t0)
        print(
            f"TIMING: {rate:.1f} samples/s, {cfg.data.num_workers} workers, "
            f"data_root {cfg.data_root}"
        )
    print(f"overlays in {out}")


if __name__ == "__main__":
    main()
