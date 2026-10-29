"""Lecture et cycle de vie des rasters sans index GeoParquet ni SAM-Road."""

import pickle
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.io import MemoryFile

from edgerunner.data.raster import RasterReader, read_patch

pytestmark = pytest.mark.integration

TRANSFORM = Affine.translation(1000, 2000) @ Affine.scale(0.2, -0.2)
PROFILE = {
    "driver": "GTiff",
    "width": 8,
    "height": 8,
    "count": 3,
    "dtype": "uint8",
    "transform": TRANSFORM,
}


def test_read_patch_resamples_and_preserves_georeferencing() -> None:
    with MemoryFile() as mem, mem.open(**PROFILE) as src:
        src.write(np.full((3, 8, 8), 128, dtype=np.uint8))
        patch = read_patch(src, 2, 4, 4, 2)
    assert patch.rgb.shape == (2, 2, 3) and patch.rgb.dtype == np.uint8
    assert (patch.rgb == 128).all() and patch.pixel_valid.all()
    assert patch.transform @ (0, 0) == pytest.approx(TRANSFORM @ (4, 2))
    assert patch.transform @ (2, 2) == pytest.approx(TRANSFORM @ (8, 6))


@pytest.mark.parametrize("explicit_mask", [False, True])
def test_invalid_pixels_follow_gdal_mask_or_zero_rgb(explicit_mask: bool) -> None:
    with MemoryFile() as mem, mem.open(**PROFILE) as src:
        rgb = np.full((3, 8, 8), 128, dtype=np.uint8)
        if not explicit_mask:
            rgb[:, :, 4:] = 0
        src.write(rgb)
        if explicit_mask:
            mask = np.full((8, 8), 255, dtype=np.uint8)
            mask[:, 4:] = 0
            src.write_mask(mask)
        patch = read_patch(src, 0, 0, 8, 8)
    assert patch.pixel_valid[:, :4].all() and not patch.pixel_valid[:, 4:].any()
    if explicit_mask:
        assert (patch.rgb == 128).all()


@pytest.fixture
def raster_root(tmp_path: Path) -> Path:
    for i in range(3):
        with rasterio.open(tmp_path / f"{i}.tif", "w", **PROFILE) as dst:
            dst.write(np.full((3, 8, 8), 128, dtype=np.uint8))
    return tmp_path


def test_cache_closes_least_recently_used_and_reopens(raster_root: Path) -> None:
    reader = RasterReader(str(raster_root), cache_size=2)
    first, second = reader.open("0.tif"), reader.open("1.tif")
    assert reader.open("0.tif") is first
    third = reader.open("2.tif")
    assert second.closed and not first.closed and not third.closed
    reopened = reader.open("1.tif")
    assert first.closed and reopened is not second and not reopened.closed
    reader.close()
    assert reopened.closed and third.closed
    reader.close()  # idempotent


def test_pickle_drops_handles_without_closing_parent(raster_root: Path) -> None:
    reader = RasterReader(str(raster_root))
    parent = reader.open("0.tif")
    clone = pickle.loads(pickle.dumps(reader))
    assert not parent.closed
    child = clone.open("0.tif")
    assert child is not parent and not child.closed
    clone.close()
    assert child.closed and not parent.closed
    reader.close()


def test_process_change_closes_inherited_handles(raster_root: Path) -> None:
    reader = RasterReader(str(raster_root))
    inherited = reader.open("0.tif")
    reader._pid = -1  # simulate a fork without running GDAL in an actual forked process
    reopened = reader.open("0.tif")
    assert inherited.closed and reopened is not inherited and not reopened.closed
    reader.close()
