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
from math import ceil
from pathlib import Path
from typing import Any

import addict
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# HTTP/S3 COGs later: no directory listing on open, cached ranges
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("VSI_CACHE", "TRUE")

import rasterio  # noqa: E402
from affine import Affine  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.windows import Window  # noqa: E402
from rasterio.windows import transform as window_transform  # noqa: E402

from .. import px  # noqa: E402
from .labels import LabelConfig, render_window, topo_labels  # noqa: E402
from .roads import Roads, read_roads  # noqa: E402


class IGNSamRoadDataset(Dataset):
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
        if raster_cache_size < 1:
            raise ValueError("raster_cache_size must be >= 1")
        self.raster_cache_size = raster_cache_size

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
        self.patches = self._eval_patches(folds, max_patches) if mode == "eval" else []

        self._pid: int | None = None
        self._src: OrderedDict[str, Any] = OrderedDict()
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
        s, out = self.native, []
        for tile_id, r0, c0, size in np.asarray(wins[["tile_id", "row_off", "col_off", "size"]]):
            if size < s:
                continue
            offs = np.linspace(0, size - s, ceil(size / s)).round().astype(int)
            out += [(tile_id, int(r0 + dr), int(c0 + dc)) for dr in offs for dc in offs]
        if max_patches and len(out) > max_patches:  # fixed subset, same every epoch
            keep = np.random.default_rng(self.seed).choice(len(out), max_patches, replace=False)
            out = [out[i] for i in sorted(keep)]
        return out

    def __getstate__(self) -> dict[str, Any]:
        # workers (fork / forkserver) start without open handles
        return {
            **self.__dict__,
            "_pid": None,
            "_src": OrderedDict(),
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
        """Release this process's cached raster handles."""
        for src in self._src.values():
            src.close()
        self._src.clear()

    def __del__(self) -> None:
        if hasattr(self, "_src"):
            self.close()

    def _open(self, uri: str) -> Any:
        if uri not in self._src:
            self._src[uri] = rasterio.open(f"{self.root}/cog/{uri}")
        self._src.move_to_end(uri)
        if len(self._src) > self.raster_cache_size:
            _, evicted = self._src.popitem(last=False)
            evicted.close()
        return self._src[uri]

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

    def _window(self, r: int, c: int) -> Window:
        s = self.native
        return Window.from_slices((r, r + s), (c, c + s))

    def _valid_frac(self, tile: Any, r: int, c: int) -> float:
        """Share of non-nodata pixels (any band != 0), read from a coarse overview."""
        ov = self._open(tile.uri).read((1, 2, 3), window=self._window(r, c), out_shape=(3, 32, 32))
        return float(np.any(ov != 0, axis=0).mean())

    def _random_window(self) -> tuple[Any, int, int]:
        assert self._rng is not None
        s, tries = self.native, 0
        while True:  # up to 20 draws to reach min_valid, else the last one
            tile = self.tiles.iloc[self._rng.choice(len(self.tiles), p=self.tile_p)]
            r = int(self._rng.integers(0, tile["height"] - s + 1))
            c = int(self._rng.integers(0, tile["width"] - s + 1))
            tries += 1
            if tries == 20 or self._valid_frac(tile, r, c) >= self.min_valid:
                return tile, r, c

    def __getitem__(self, idx: int) -> dict[str, Any]:
        self._check_process()
        if self.mode == "train":
            assert self._rng is not None
            tile, r, c = self._random_window()
            return self.read(tile, r, c, rot=int(self._rng.integers(4)))
        tile_id, r, c = self.patches[idx]
        np.random.seed((self.seed + idx) % 2**32)  # upstream sample_patch draws from np.random
        return self.read(self.tile(tile_id), r, c)

    def read(self, tile: Any, r: int, c: int, rot: int = 0) -> dict[str, Any]:
        """Patch whose top-left COG pixel is (r, c) in `tile`, rotated `rot` × 90° CCW."""
        self._check_process()
        s, p = self.native, self.patch
        src = self._open(tile.uri)
        win = self._window(r, c)
        rgb = src.read((1, 2, 3), window=win, out_shape=(3, p, p), resampling=Resampling.average)
        # Honour GDAL masks and the IGN all-zero RGB convention (also used by the toy).
        pixel_valid = np.any(rgb != 0, axis=0) & (
            src.dataset_mask(window=win, out_shape=(p, p), resampling=Resampling.nearest) > 0
        )
        patch_tf = window_transform(win, src.transform) @ Affine.scale(s / p)
        lab = render_window(self._tile_roads(tile), ~patch_tf, p, self.gsd, self.labels)
        points, samples = topo_labels(self.sam_cfg, lab, p, self.gsd, self.labels, rot)
        pairs, connected, valid = (np.asarray(x) for x in zip(*samples, strict=True))
        pixel_valid = np.ascontiguousarray(np.rot90(pixel_valid, rot))
        # No topology supervision when either sampled endpoint is outside visible imagery.
        point_rc = np.floor(points[:, ::-1] + 0.5).astype(int)
        inside = ((point_rc >= 0) & (point_rc < p)).all(axis=1)
        rc = np.clip(point_rc, 0, p - 1)
        point_valid = inside & pixel_valid[rc[:, 0], rc[:, 1]]
        valid &= point_valid[pairs].all(axis=-1)
        rgb, road, kp = (
            np.ascontiguousarray(np.rot90(a, rot, (0, 1)))
            for a in (np.moveaxis(rgb, 0, -1), lab.road_mask, lab.keypoint_mask)
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


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stacks tensors, zero-pads `graph_points` (as sam_road), keeps metadata as lists.

    Module-level so that forkserver workers can pickle it.
    """
    out: dict[str, Any] = {}
    for key in batch[0]:
        vals = [item[key] for item in batch]
        if key == "graph_points":
            n = max(v.shape[0] for v in vals)
            out[key] = torch.stack([torch.cat([v, v.new_zeros(n - v.shape[0], 2)]) for v in vals])
        elif isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals)
        else:
            out[key] = vals
    return out


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
