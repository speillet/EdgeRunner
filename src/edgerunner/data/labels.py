"""SAM-Road labels for one patch, drawn from the roads of its tile.

Everything is built on the patch grown by `margin` px, then cropped: roads leaving the patch keep
their real neighbours (no fake endpoint at the border) and the topology BFS of sam_road can walk
out of the patch and back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import rtree
import scipy.spatial
import shapely

from .. import px
from .._sam_road import graph_utils, sam_dataset
from .roads import Roads
from .sampling import numpy_seed

Graph = dict[tuple[int, int], list[tuple[int, int]]]  # sat2graph adjacency, (row, col) nodes


@dataclass(frozen=True)
class LabelConfig:
    """Ground distances in metres, converted at the patch GSD."""

    margin_m: float = 64.0  # >= NEIGHBOR_RADIUS: BFS paths may leave the patch
    simplify_m: float = 1.0  # keeps corners, drops BD TOPO micro-vertices
    road_width_m: float | None = 3.0  # None → BD TOPO carriageway width
    keypoint_radius_m: float = 3.0
    # pixel constants of sam_road's GraphLabelGenerator, tuned for 1 m/px
    subdivide_m: float = 4.0
    crossover_exclude_m: float = 4.0
    interesting_m: float = 32.0


@dataclass
class WindowLabels:
    road_mask: np.ndarray  # uint8 {0, 1}, [P, P]
    keypoint_mask: np.ndarray  # uint8 {0, 1}, [P, P]
    graph: Graph  # grown frame
    margin: int  # px: the patch is [margin, margin + P) in the grown frame


def render_window(
    roads: Roads, inv_transform: Any, size: int, gsd: float, cfg: LabelConfig
) -> WindowLabels:
    """Masks + graph of the `size`² patch whose world → pixel transform is `inv_transform`."""
    m = px(cfg.margin_m, gsd)
    full = size + 2 * m
    fwd = ~inv_transform
    (xa, ya), (xb, yb) = fwd @ (-m, -m), fwd @ (size + m, size + m)
    rect = (min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb))

    road = np.zeros((full, full), np.uint8)
    adj: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for i in roads.tree.query(shapely.box(*rect)):
        width = px(cfg.road_width_m or float(roads.width_m[i]), gsd)
        for part in shapely.get_parts(shapely.clip_by_rect(roads.geoms[i], *rect)):
            if part.geom_type != "LineString":
                continue
            xy = shapely.get_coordinates(shapely.simplify(part, cfg.simplify_m))
            if len(xy) < 2:
                continue
            cols, rows = inv_transform @ (xy[:, 0], xy[:, 1])
            rc = np.rint(np.stack([rows, cols], axis=1) + m).astype(np.int64)
            cv2.polylines(road, [rc[:, ::-1].astype(np.int32)], False, 1, thickness=width)
            nodes = [tuple(p) for p in rc.tolist()]
            for a, b in zip(nodes[:-1], nodes[1:], strict=True):
                if a != b:
                    adj.setdefault(a, set()).add(b)
                    adj.setdefault(b, set()).add(a)

    keypoint = np.zeros_like(road)
    radius = px(cfg.keypoint_radius_m, gsd)
    for (r, c), neis in adj.items():
        if len(neis) != 2:  # sam_road rule: intersections and dead ends
            cv2.circle(keypoint, (c, r), radius, 1, -1)

    crop = (slice(m, m + size), slice(m, m + size))
    graph = {k: sorted(v) for k, v in adj.items()}
    return WindowLabels(road[crop].copy(), keypoint[crop].copy(), graph, m)


class WindowGraphLabeler(sam_dataset.GraphLabelGenerator):
    """sam_road's GraphLabelGenerator on one window graph instead of a whole image.

    Only `__init__` is rewritten: its hard-coded pixel radii become parameters and the rtree is
    bulk-loaded. `sample_patch` (NMS, neighbour queries, BFS connectivity, rotation) is upstream's.
    """

    def __init__(
        self,
        config: Any,
        graph: Graph,
        subdivide_px: int,
        crossover_exclude_px: float,
        interesting_px: float,
    ):
        self.config = config
        self.subdivide_resolution = subdivide_px
        if not graph:  # upstream sample_patch returns its padding sample when the rtree is empty
            self.graph_rtee = rtree.index.Index()
            self.exclude_indices: set[int] = set()
            self.crossover_points: list[tuple[float, float]] = []
            return

        # (row, col) → (x, y), as upstream does for cityscale
        self.full_graph_origin = graph_utils.igraph_from_adj_dict(graph, lambda v: v[:, ::-1])
        self.crossover_points = graph_utils.find_crossover_points(self.full_graph_origin)
        self.full_graph_subdivide = graph_utils.subdivide_graph(
            self.full_graph_origin, subdivide_px
        )
        pts = np.array(self.full_graph_subdivide.vs["point"], dtype=np.float64)
        self.subdivide_points = pts
        self.graph_rtee = rtree.index.Index((i, (x, y, x, y), None) for i, (x, y) in enumerate(pts))
        self.graph_kdtree = scipy.spatial.KDTree(pts)

        # bridges: points near a crossover are never sampled
        self.exclude_indices = set()
        for p in self.crossover_points:
            self.exclude_indices.update(self.graph_kdtree.query_ball_point(p, crossover_exclude_px))

        itsc = np.flatnonzero(np.array(self.full_graph_subdivide.degree()) != 2)
        self.nms_score_override = np.zeros(len(pts), np.float32)
        self.nms_score_override[itsc] = 2.0  # intersections always survive the NMS

        # points around intersections and crossovers are sampled more often
        interesting: set[int] = set()
        for p in [*pts[itsc], *self.crossover_points]:
            interesting.update(self.graph_kdtree.query_ball_point(p, interesting_px))
        self.sample_weights = np.full(len(pts), 0.1, np.float32)
        self.sample_weights[sorted(interesting)] = 0.9


def topo_labels(
    sam_cfg: Any,
    labels: WindowLabels,
    size: int,
    gsd: float,
    cfg: LabelConfig,
    rot: int = 0,
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, list[Any]]:
    """`sample_patch` on the patch: points (x, y) in the patch frame + (pairs, connected, valid)."""
    gen = WindowGraphLabeler(
        sam_cfg,
        labels.graph,
        px(cfg.subdivide_m, gsd),
        cfg.crossover_exclude_m / gsd,
        cfg.interesting_m / gsd,
    )
    m = labels.margin
    with numpy_seed(seed):
        return gen.sample_patch(((m, m), (m + size, m + size)), rot_index=rot)
