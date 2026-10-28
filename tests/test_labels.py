"""Window labels: geometry → masks / sat2graph graph / SAM-Road topology samples."""

from __future__ import annotations

import addict
import numpy as np
import pytest
import shapely
from affine import Affine

from edgerunner.data.labels import LabelConfig, WindowGraphLabeler, render_window, topo_labels
from edgerunner.data.roads import Roads

GSD, P = 0.4, 256
INV = ~(Affine.translation(1000, 2000) @ Affine.scale(GSD, -GSD))  # patch (0, 0) at x=1000 y=2000
CFG = LabelConfig()
SAM = addict.Dict(
    PATCH_SIZE=P,
    TOPO_SAMPLE_NUM=64,
    ROAD_NMS_RADIUS=40,
    NEIGHBOR_RADIUS=160,
    MAX_NEIGHBOR_QUERIES=16,
)


def w(r: float, c: float) -> tuple[float, float]:
    """Patch pixel (row, col) → world (x, y)."""
    return 1000 + c * GSD, 2000 - r * GSD


def roads(*lines: list[tuple[float, float]]) -> Roads:
    return Roads.from_geoms([shapely.LineString([w(*p) for p in line]) for line in lines])


T_JUNCTION = roads([(128, -600), (128, 100)], [(128, 100), (128, 200)], [(128, 100), (700, 100)])


def test_t_junction_keypoints_and_border() -> None:
    lab = render_window(T_JUNCTION, INV, P, GSD, CFG)
    kp, road = lab.keypoint_mask, lab.road_mask
    assert kp.shape == road.shape == (P, P)
    assert kp[128, 100] == 1  # junction
    assert kp[128, 200] == 1  # dead end inside the patch
    assert kp[:, :20].max() == 0 and kp[236:, :].max() == 0  # roads leaving: no fake endpoint
    m = lab.margin
    assert m == 160 and len(lab.graph[(128 + m, 100 + m)]) == 3


def test_road_mask_is_3_m_wide() -> None:
    road = render_window(T_JUNCTION, INV, P, GSD, CFG).road_mask
    col = road[:, 150]
    assert col[128] == 1 and 6 <= col.sum() <= 10  # 3 m = 8 px at 0.4 m


def test_real_width_option() -> None:
    r = Roads.from_geoms([shapely.LineString([w(128, -600), w(128, 900)])], [10.0])
    road = render_window(r, INV, P, GSD, LabelConfig(road_width_m=None)).road_mask
    assert 23 <= road[:, 150].sum() <= 27  # 10 m = 25 px


def test_bridge_is_a_crossover_not_a_junction() -> None:
    lab = render_window(
        roads([(128, -600), (128, 900)], [(-600, 128), (900, 128)]), INV, P, GSD, CFG
    )
    assert lab.keypoint_mask[118:139, 118:139].max() == 0
    gen = WindowGraphLabeler(SAM, lab.graph, 10, 10, 80)
    m = lab.margin
    assert any(np.hypot(x - 128 - m, y - 128 - m) < 1 for x, y in gen.crossover_points)
    pts, _ = gen.sample_patch(((m, m), (m + P, m + P)))
    assert np.hypot(pts[:, 0] - 128, pts[:, 1] - 128).min() > 5  # excluded around the crossing


def test_parallel_roads_are_never_connected() -> None:
    lab = render_window(
        roads([(100, -600), (100, 900)], [(150, -600), (150, 900)]), INV, P, GSD, CFG
    )
    np.random.seed(0)
    pts, samples = topo_labels(SAM, lab, P, GSD, CFG)
    n_conn = 0
    for pairs, connected, valid in samples:
        for (s, t), c, v in zip(pairs, connected, valid, strict=True):
            if v:
                same_road = abs(pts[s, 1] - pts[t, 1]) < 10
                assert not (c and not same_road)
                n_conn += c
    assert n_conn > 0


@pytest.mark.parametrize("rot", [0, 1, 2, 3])
def test_points_follow_rotation(rot: int) -> None:
    lab = render_window(T_JUNCTION, INV, P, GSD, CFG)
    np.random.seed(rot)
    pts, _ = topo_labels(SAM, lab, P, GSD, CFG, rot=rot)
    road = np.rot90(lab.road_mask, rot, (0, 1))
    rc = np.clip(np.rint(pts[:, ::-1]).astype(int), 0, P - 1)  # (x, y) → (row, col)
    assert road[rc[:, 0], rc[:, 1]].mean() > 0.9


def test_empty_window_gives_padding_sample() -> None:
    lab = render_window(Roads.from_geoms([]), INV, P, GSD, CFG)
    pts, samples = topo_labels(SAM, lab, P, GSD, CFG)
    assert pts.shape == (1, 2) and len(samples) == SAM.TOPO_SAMPLE_NUM
    assert not any(any(valid) for _, _, valid in samples)


def test_same_footprint_at_coarser_gsd() -> None:
    inv2 = ~(Affine.translation(1000, 2000) @ Affine.scale(0.8, -0.8))
    lab = render_window(T_JUNCTION, inv2, P // 2, 0.8, CFG)
    assert lab.keypoint_mask[64, 50] == 1 and lab.keypoint_mask[64, 100] == 1
