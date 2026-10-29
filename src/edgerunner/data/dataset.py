"""On-the-fly IGN dataset for SAM-Road: a COG window + the roads of its tile, nothing stored.

Data root layout: `index/{tiles,windows}.parquet`, `cog/<uri>`, `bdtopo/troncon/dept=XX/`.
Pixels are read from the COG by window at the run GSD (GDAL serves the matching overview). Roads
are read once per tile footprint (+ margin) and worker, kept in a small LRU. Masks and topology
labels are drawn per sample, on the patch only.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import addict
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .. import px
from .batch import collate as collate
from .labels import LabelConfig, render_window, topo_labels
from .raster import RasterReader
from .roads import Roads, read_roads
from .sampling import eval_grid, random_window
from .transforms import visible_pairs
from .types import Sample


class IGNSamRoadDataset(Dataset[Sample]):
    """Map-style dataset returning SAM-Road's batch items (see `read`).

    mode="train": random window in a random tile of `folds` (weighted by valid area), random rot90.
    mode="eval": the `windows.parquet` windows of `folds`, each cut into patches (one patch when
    the window footprint equals the patch footprint), deterministic labels.
    `sam_cfg` is the SAM-Road config (PATCH_SIZE, TOPO_SAMPLE_NUM, ROAD_NMS_RADIUS,
    NEIGHBOR_RADIUS, MAX_NEIGHBOR_QUERIES in pixels at `gsd`).
    """

    def __init__(
        self,
        root: str | Path,
        folds: Sequence[str | int],
        sam_cfg: Mapping[str, Any],
        labels: Mapping[str, Any] | None = None,
        mode: str = "train",
        gsd: float = 0.4,
        samples_per_epoch: int = 20_000,
        max_patches: int | None = None,
        min_valid: float = 0.5,
        roads_cache_size: int = 64,
        raster_cache_size: int = 32,
        native_gsd: float = 0.2,
        seed: int = 0,
    ):
        import geopandas as gpd

        assert mode in ("train", "eval")
        self.sam_cfg = addict.Dict(sam_cfg)
        self.labels = LabelConfig(**(labels or {}))
        self.mode, self.gsd, self.seed = mode, gsd, seed
        self.root = str(root).rstrip("/")  # str: may become an http(s) prefix later
        self.samples_per_epoch, self.min_valid = samples_per_epoch, min_valid
        self.roads_cache_size = roads_cache_size
        self.rasters = RasterReader(f"{self.root}/cog", raster_cache_size)

        self.patch = int(self.sam_cfg.PATCH_SIZE)
        native = self.patch * gsd / native_gsd
        self.native = round(native)  # window side in COG pixels
        if abs(native - self.native) > 1e-6:
            raise ValueError(f"patch {self.patch} at {gsd} m is not a whole number of COG pixels")

        folds = [str(f) for f in folds]
        index = Path(self.root) / "index" / "tiles.parquet"
        if not index.exists():
            raise FileNotFoundError(f"{index} not found: set data_root= (synthetic set: just toy)")
        tiles = gpd.read_parquet(index)
        self.tiles = tiles[tiles["fold"].astype(str).isin(folds)].reset_index(drop=True)
        if not len(self.tiles):
            raise ValueError(f"no tile in folds {folds}")
        self.tile_pos = {t: i for i, t in enumerate(self.tiles["tile_id"])}
        w = np.asarray(self.tiles["valid_frac"], dtype=float)
        self.tile_p = w / w.sum()
        self.tile_shapes = list(zip(self.tiles["height"], self.tiles["width"], strict=True))
        self.patches = self._eval_patches(folds, max_patches) if mode == "eval" else []

        self._pid: int | None = None
        self._roads: OrderedDict[str, Roads] = OrderedDict()
        self._rng: np.random.Generator | None = None

    def _eval_patches(
        self, folds: list[str], max_patches: int | None
    ) -> list[tuple[str, int, int]]:
        import pandas as pd

        wins = pd.read_parquet(
            f"{self.root}/index/windows.parquet",
            columns=["tile_id", "fold", "row_off", "col_off", "size"],
        )
        wins = wins[wins["fold"].astype(str).isin(folds) & wins["tile_id"].isin(self.tile_pos)]
        return eval_grid(
            [
                (str(t), int(r), int(c), int(s))
                for t, r, c, s in np.asarray(wins[["tile_id", "row_off", "col_off", "size"]])
            ],
            self.native,
            self.seed,
            max_patches,
        )

    def __getstate__(self) -> dict[str, Any]:
        # workers (fork / forkserver) start without open handles
        return {
            **self.__dict__,
            "_pid": None,
            "_roads": OrderedDict(),
            "_rng": None,
        }

    def __len__(self) -> int:
        return self.samples_per_epoch if self.mode == "train" else len(self.patches)

    def tile(self, tile_id: str) -> Any:
        return self.tiles.iloc[self.tile_pos[tile_id]]

    def _check_process(self) -> None:
        """Per-process state: handles inherited through fork are dropped, RNG reseeded."""
        if self._pid == os.getpid():
            return
        self.close()
        self._pid, self._roads = os.getpid(), OrderedDict()
        # torch seeds each worker (rank-aware under Lightning's seed_everything(workers=True))
        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
        self._rng = np.random.default_rng([torch.initial_seed(), rank])

    def close(self) -> None:
        self.rasters.close()

    def _tile_roads(self, tile: Any) -> Roads:
        roads = self._roads.get(tile.tile_id)
        if roads is None:
            pad = (px(self.labels.margin_m, self.gsd) + 1) * self.gsd
            x0, y0, x1, y1 = tile.geometry.bounds
            roads = read_roads(
                f"{self.root}/bdtopo/troncon", (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
            )
            self._roads[tile.tile_id] = roads
            if len(self._roads) > self.roads_cache_size:
                self._roads.popitem(last=False)
        self._roads.move_to_end(tile.tile_id)
        return roads

    def _random_window(self) -> tuple[Any, int, int]:
        assert self._rng is not None
        index, row, col = random_window(
            self.tile_shapes,
            self.tile_p,
            self.native,
            self._rng,
            lambda i, r, c: self.rasters.valid_fraction(self.tiles.iloc[i].uri, r, c, self.native),
            self.min_valid,
        )
        return self.tiles.iloc[index], row, col

    def __getitem__(self, idx: int) -> Sample:
        self._check_process()
        if self.mode == "train":
            assert self._rng is not None
            tile, r, c = self._random_window()
            return self.read(tile, r, c, rot=int(self._rng.integers(4)))
        tile_id, r, c = self.patches[idx]
        return self.read(self.tile(tile_id), r, c, topo_seed=(self.seed + idx) % 2**32)

    def read(self, tile: Any, r: int, c: int, rot: int = 0, topo_seed: int | None = None) -> Sample:
        """Patch whose top-left COG pixel is (r, c) in `tile`, rotated `rot` × 90° CCW."""
        self._check_process()
        s, p = self.native, self.patch
        patch = self.rasters.read(tile.uri, r, c, s, p)
        lab = render_window(self._tile_roads(tile), ~patch.transform, p, self.gsd, self.labels)
        if topo_seed is None:
            assert self._rng is not None
            topo_seed = int(self._rng.integers(2**32))
        points, samples = topo_labels(
            self.sam_cfg, lab, p, self.gsd, self.labels, rot, seed=topo_seed
        )
        pairs, connected, valid = (np.asarray(x) for x in zip(*samples, strict=True))
        pixel_valid = np.ascontiguousarray(np.rot90(patch.pixel_valid, rot))
        valid &= visible_pairs(points, pairs, pixel_valid)
        rgb, road, kp = (
            np.ascontiguousarray(np.rot90(a, rot, (0, 1)))
            for a in (patch.rgb, lab.road_mask, lab.keypoint_mask)
        )
        return {
            "rgb": torch.from_numpy(rgb),  # uint8 [P, P, 3], cast by the module on the GPU
            "keypoint_mask": torch.from_numpy(kp),  # uint8 {0, 1} [P, P]
            "road_mask": torch.from_numpy(road),
            "pixel_valid": torch.from_numpy(pixel_valid),  # bool [P, P]
            "graph_points": torch.as_tensor(points, dtype=torch.float32),  # [N, 2] (x, y)
            "pairs": torch.as_tensor(pairs, dtype=torch.int32),  # [S, Q, 2]
            "connected": torch.as_tensor(connected, dtype=torch.bool),  # [S, Q]
            "valid": torch.as_tensor(valid, dtype=torch.bool),  # [S, Q]
            "tile_id": str(tile.tile_id),
            "row_off": r,
            "col_off": c,
            "rot": rot,
        }


def make_loaders(
    root: str | Path,
    sam_cfg: Mapping[str, Any],
    train_folds: Sequence[str],
    val_folds: Sequence[str],
    batch_size: int = 4,
    num_workers: int = 8,
    samples_per_epoch: int = 20_000,
    val_max_patches: int | None = 1000,
    multiprocessing_context: str = "forkserver",
    **dataset_kw: Any,
) -> tuple[DataLoader, DataLoader]:
    """Train loader (random windows of `train_folds`) and val loader (eval windows of `val_folds`).

    forkserver by default: forking once pyarrow / GDAL threads run can deadlock the workers.
    """
    common = {"root": root, "sam_cfg": sam_cfg, **dataset_kw}
    train = IGNSamRoadDataset(
        folds=train_folds, mode="train", samples_per_epoch=samples_per_epoch, **common
    )
    val = IGNSamRoadDataset(folds=val_folds, mode="eval", max_patches=val_max_patches, **common)
    kw: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers:
        kw["multiprocessing_context"] = multiprocessing_context
    return DataLoader(train, drop_last=True, **kw), DataLoader(val, **kw)
