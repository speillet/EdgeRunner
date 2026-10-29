"""Lecture de fenêtres raster et cache LRU de handles, propre à chaque processus."""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("VSI_CACHE", "TRUE")

import rasterio  # noqa: E402
from affine import Affine  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.windows import Window  # noqa: E402
from rasterio.windows import transform as window_transform  # noqa: E402


@dataclass
class RasterPatch:
    rgb: np.ndarray  # uint8 [P,P,3]
    pixel_valid: np.ndarray  # bool [P,P]
    transform: Affine  # pixel → coordonnées projetées


def window(row: int, col: int, side: int) -> Window:
    return Window.from_slices((row, row + side), (col, col + side))


def read_patch(src: Any, row: int, col: int, side: int, output_size: int) -> RasterPatch:
    """Lit un handle rasterio ; utilisable avec un MemoryFile, sans index ni routes."""
    win = window(row, col, side)
    rgb = src.read(
        (1, 2, 3),
        window=win,
        out_shape=(3, output_size, output_size),
        resampling=Resampling.average,
    )
    pixel_valid = np.any(rgb != 0, axis=0) & (
        src.dataset_mask(
            window=win, out_shape=(output_size, output_size), resampling=Resampling.nearest
        )
        > 0
    )
    transform = window_transform(win, src.transform) @ Affine.scale(side / output_size)
    return RasterPatch(np.moveaxis(rgb, 0, -1), pixel_valid, transform)


class RasterReader:
    def __init__(self, root: str, cache_size: int = 32):
        if cache_size < 1:
            raise ValueError("raster_cache_size must be >= 1")
        self.root, self.cache_size = root, cache_size
        self._pid: int | None = None
        self._sources: OrderedDict[str, Any] = OrderedDict()

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_pid": None, "_sources": OrderedDict()}

    def open(self, uri: str) -> Any:
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if uri not in self._sources:
            self._sources[uri] = rasterio.open(f"{self.root}/{uri}")
        self._sources.move_to_end(uri)
        if len(self._sources) > self.cache_size:
            _, evicted = self._sources.popitem(last=False)
            evicted.close()
        return self._sources[uri]

    def read(self, uri: str, row: int, col: int, side: int, output_size: int) -> RasterPatch:
        return read_patch(self.open(uri), row, col, side, output_size)

    def valid_fraction(self, uri: str, row: int, col: int, side: int) -> float:
        """Convention IGN RVB nul, sur overview grossier pour le rejet rapide des fenêtres."""
        rgb = self.open(uri).read((1, 2, 3), window=window(row, col, side), out_shape=(3, 32, 32))
        return float(np.any(rgb != 0, axis=0).mean())

    def close(self) -> None:
        for src in self._sources.values():
            src.close()
        self._sources.clear()

    def __del__(self) -> None:
        if hasattr(self, "_sources"):
            self.close()
