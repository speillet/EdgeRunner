"""On-the-fly dataset on synthetic tiles: BD TOPO filters, batch contract, eval grid, workers."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from edgerunner.data.dataset import IGNSamRoadDataset, collate
from edgerunner.data.roads import read_roads
from edgerunner.data.toy import X0, Y_TOP

pytestmark = pytest.mark.integration


def sam_cfg(patch: int, gsd: float) -> dict[str, Any]:
    px = lambda m: round(m / gsd)  # noqa: E731
    return {
        "PATCH_SIZE": patch,
        "TOPO_SAMPLE_NUM": 32,
        "ROAD_NMS_RADIUS": px(16),
        "NEIGHBOR_RADIUS": px(64),
        "MAX_NEIGHBOR_QUERIES": 16,
    }


def make(
    toy: Path, folds: list[str], mode: str = "train", patch: int = 256, gsd: float = 0.4, **kw: Any
) -> IGNSamRoadDataset:
    return IGNSamRoadDataset(
        toy, folds, sam_cfg(patch, gsd), mode=mode, gsd=gsd, samples_per_epoch=8, **kw
    )


def test_read_roads_filters(toy: Path) -> None:
    r = read_roads(str(toy / "bdtopo" / "troncon"), (X0, Y_TOP - 200, X0 + 1000, Y_TOP))
    assert len(r) > 0
    ys = {round(y) for g in r.geoms for y in g.xy[1]}
    assert Y_TOP - 75 not in ys  # tunnel
    assert all(not (g.xy[0][0] == g.xy[0][-1] == X0 + 75) for g in r.geoms)  # footpath
    assert Y_TOP - 25 not in ys and Y_TOP - 35 not in ys  # fictitious, not in service
    assert set(r.width_m) == {3.0, 5.0}  # Chemin without width → nature default


def test_train_item_contract(toy: Path) -> None:
    ds = make(toy, ["0", "1", "2"])
    item = ds[0]
    assert item["rgb"].shape == (256, 256, 3) and item["rgb"].dtype == torch.uint8
    for k in ("road_mask", "keypoint_mask"):
        assert item[k].shape == (256, 256) and item[k].dtype == torch.uint8
        assert set(item[k].unique().tolist()) <= {0, 1}
    assert item["pairs"].shape == (32, 16, 2) and item["pairs"].dtype == torch.int32
    assert item["connected"].shape == item["valid"].shape == (32, 16)
    assert item["road_mask"].any() and item["valid"].any()
    pts = item["graph_points"]
    assert pts.ndim == 2 and pts.shape[1] == 2 and pts.min() > -5 and pts.max() < 261


def test_mask_matches_painted_roads(toy: Path) -> None:
    ds = make(toy, ["0"])
    item = ds.read(ds.tile("toy_0"), 100, 100)
    grey = (item["rgb"].int() - 128).abs().sum(-1) < 6
    on_road = grey[item["road_mask"].bool()].float().mean()
    assert on_road > 0.8  # labels sit on the roads painted in the image


def test_eval_grid_and_determinism(toy: Path) -> None:
    ev = make(toy, ["4"], mode="eval")
    assert len(ev) == 4  # toy_4: 4 windows of 512 (≥ 50 % valid), 1 patch of 512 COG px each
    a, b = ev[0], ev[0]
    assert torch.equal(a["graph_points"], b["graph_points"]) and a["rot"] == 0
    fine = make(toy, ["4"], mode="eval", gsd=0.2)  # 256 px at 0.2 m: 2 × 2 patches per window
    assert len(fine) == 16


def test_same_footprint_at_0_8_m(toy: Path) -> None:
    a = make(toy, ["1"], patch=256, gsd=0.4).read(make(toy, ["1"]).tile("toy_1"), 0, 0)
    ds8 = make(toy, ["1"], patch=128, gsd=0.8)
    b = ds8.read(ds8.tile("toy_1"), 0, 0)
    fa = a["road_mask"].float().mean().item()
    fb = b["road_mask"].float().mean().item()
    assert abs(fa - fb) < 0.03


def test_valid_frac_sees_nodata(toy: Path) -> None:
    ds = make(toy, ["4"], gsd=0.2)  # 256 COG px windows
    ds._check_process()
    assert (
        ds.rasters.valid_fraction(ds.tile("toy_4").uri, 0, 768, ds.native) == 0.0
    )  # last quarter of the tile is 0
    assert ds.rasters.valid_fraction(ds.tile("toy_4").uri, 0, 0, ds.native) == 1.0


def test_pickles_without_handles(toy: Path) -> None:
    ds = make(toy, ["0"])
    ds[0]
    assert ds.rasters._sources and ds._roads
    clone = pickle.loads(pickle.dumps(ds))
    assert not clone.rasters._sources and not clone._roads and clone._rng is None
    assert clone[0]["rgb"].shape == (256, 256, 3)


@pytest.mark.parametrize("ctx", ["fork", "forkserver"])
def test_workers(toy: Path, ctx: str) -> None:
    ds = make(toy, ["0", "1"])
    dl = DataLoader(
        ds, batch_size=2, num_workers=2, collate_fn=collate, multiprocessing_context=ctx
    )
    batches = list(dl)
    assert len(batches) == 4 and batches[0]["rgb"].shape == (2, 256, 256, 3)
    offs = {(b["tile_id"][i], b["row_off"][i], b["col_off"][i]) for b in batches for i in range(2)}
    assert len(offs) > 4  # workers draw different windows
    assert np.isfinite(batches[0]["graph_points"].numpy()).all()


@pytest.mark.parametrize("rot", [0, 1, 2, 3])
def test_nodata_mask_and_topology_follow_rotation(toy: Path, rot: int) -> None:
    ds = make(toy, ["4"])
    item = ds.read(ds.tile("toy_4"), 0, 512, rot=rot)
    expected = np.ones((256, 256), dtype=bool)
    expected[:, 128:] = False
    assert np.array_equal(item["pixel_valid"].numpy(), np.rot90(expected, rot))
    pairs = item["pairs"][item["valid"]].long()
    assert len(pairs) > 0
    rc = (item["graph_points"][pairs].flip(-1) + 0.5).floor().long()
    assert ((rc >= 0) & (rc < 256)).all()
    assert item["pixel_valid"][rc[..., 0], rc[..., 1]].all()
    ds_empty = make(toy, ["4"], gsd=0.2)
    empty = ds_empty.read(ds_empty.tile("toy_4"), 0, 768, rot=rot)
    assert not empty["pixel_valid"].any() and not empty["valid"].any()


def test_validation_preserves_numpy_rng_and_labels_across_reads(toy: Path) -> None:
    ds = make(toy, ["4"], mode="eval")
    before = np.random.get_state()
    first = ds[0]
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    ds[1]  # interleave another patch
    again = ds[0]
    for key in ("graph_points", "pairs", "valid", "connected"):
        assert torch.equal(first[key], again[key])
