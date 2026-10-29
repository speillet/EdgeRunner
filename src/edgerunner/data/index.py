"""Index of the IGN COGs: `tiles.parquet` (one fold per tile) and `windows.parquet` (eval grid).

Same split as road_graph_extractor's index (rewritten here, nothing imported): a zone is a
4-connected cluster of 5 km tiles (a cross of up to 5 tiles), numbered in that project's selection
order (departments sorted, then grid column west → east, then south → north). Per department,
after one seeded shuffle, the first zone goes to `test` and the others to folds "0".."4" in turn.
Tiles outside Bretagne are the annotated zones (`annot_<dept>`, fold `ext_test`).

    uv run python -m edgerunner.data.index /mnt/stores/store-DAI/.../cog data/index
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import shapely
from rasterio.windows import bounds

from .raster import window

BRETAGNE = ("022", "029", "035", "056")
STEP_KM = 5  # BD ORTHO tile side
COLUMN_GAP_KM = 12  # zone centres are >= 20 km apart, coastal crosses drift by <= 7 km


def tile_xy(tile_id: str) -> tuple[int, int]:
    """Lambert-93 km in the tile name, e.g. 35-2026-0350-6790-LA93-0M20-E080 → (350, 6790)."""
    _, _, x, y, *_ = tile_id.split("-")
    return int(x), int(y)


def clusters(xy: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """4-connected tile groups, west → east by grid column, then south → north.

    Coastal crosses are truncated and their mean drifts: columns are cut on x gaps.
    """
    seen: set[tuple[int, int]] = set()
    groups = []
    for start in sorted(xy):
        if start in seen:
            continue
        seen.add(start)
        stack, group = [start], []
        while stack:
            x, y = stack.pop()
            group.append((x, y))
            for n in ((x + STEP_KM, y), (x - STEP_KM, y), (x, y + STEP_KM), (x, y - STEP_KM)):
                if n in xy and n not in seen:
                    seen.add(n)
                    stack.append(n)
        groups.append(sorted(group))
    means = [np.mean(g, axis=0) for g in groups]
    order = sorted(range(len(groups)), key=lambda i: means[i][0])
    col, keyed = 0, []
    for k, i in enumerate(order):
        col += int(k > 0 and means[i][0] - means[order[k - 1]][0] > COLUMN_GAP_KM)
        keyed.append((col, means[i][1], i))
    return [groups[i] for _, _, i in sorted(keyed)]


def assign_folds(
    tile_ids: Mapping[str, Sequence[str]], n_test: int = 1, k: int = 5, seed: int = 0
) -> dict[str, tuple[str, str]]:
    """{dept: tile ids} → {tile id: (zone_id, fold)}."""
    zone_of: dict[str, str] = {}
    n = 0
    for dept in sorted(tile_ids):
        if dept not in BRETAGNE:
            zone_of.update(dict.fromkeys(tile_ids[dept], f"annot_{dept}"))
            continue
        by_xy = {tile_xy(t): t for t in tile_ids[dept]}
        for group in clusters(set(by_xy)):
            zone_of.update({by_xy[q]: str(n) for q in group})
            n += 1
    rng, fold_of, j = np.random.default_rng(seed), {}, 0
    for dept in sorted(tile_ids):
        zones = sorted({zone_of[t] for t in tile_ids[dept]})  # string order, as the original ids
        if dept not in BRETAGNE:
            fold_of.update(dict.fromkeys(zones, "ext_test"))
            continue
        rng.shuffle(zones)
        for i, z in enumerate(zones):
            if i < n_test:
                fold_of[z] = "test"
            else:
                fold_of[z], j = str(j % k), j + 1
    return {t: (z, fold_of[z]) for t, z in zone_of.items()}


def scan(path: Path, eval_size: int = 2048, min_valid: float = 0.5, factor: int = 32) -> tuple:
    """Valid fraction (RGB != 0, coarse overview) of the tile and of each eval window."""
    with rasterio.open(path) as src:
        rgb = src.read((1, 2, 3), out_shape=(3, src.height // factor, src.width // factor))
        h, w, tf = src.height, src.width, src.transform
        tile_box = shapely.box(*src.bounds)
    valid = np.any(rgb != 0, axis=0)
    wins = []
    for r in range(0, h - eval_size + 1, eval_size):
        for c in range(0, w - eval_size + 1, eval_size):
            vf = float(valid[r // factor : (r + eval_size) // factor,
                             c // factor : (c + eval_size) // factor].mean())  # fmt: skip
            if vf >= min_valid:
                box = shapely.box(*bounds(window(r, c, eval_size), tf))
                wins.append({"row_off": r, "col_off": c, "size": eval_size, "valid_frac": vf,
                             "geometry": box})  # fmt: skip
    return h, w, float(valid.mean()), tile_box, wins


def build_index(cog_dir: str | Path, out: str | Path, workers: int = 8) -> Any:
    import geopandas as gpd

    cog_dir, out = Path(cog_dir), Path(out)
    paths = sorted(cog_dir.glob("*/*.tif"))
    by_dept: dict[str, list[str]] = {}
    for p in paths:
        by_dept.setdefault(p.parent.name, []).append(p.stem)
    folds = assign_folds(by_dept)
    with ThreadPoolExecutor(workers) as pool:  # CIFS latency bound
        scans = list(pool.map(scan, paths))
    tiles, wins = [], []
    for p, (h, w, vf, box, tile_wins) in zip(paths, scans, strict=True):
        zone, fold = folds[p.stem]
        meta = {"tile_id": p.stem, "fold": fold, "zone_id": zone}
        tiles.append({**meta, "uri": f"{p.parent.name}/{p.name}", "dept": p.parent.name,
                      "year": int(p.stem.split("-")[1]), "width": w, "height": h,
                      "valid_frac": vf, "geometry": box})  # fmt: skip
        wins += [{**meta, **win} for win in tile_wins]
    out.mkdir(parents=True, exist_ok=True)
    tiles = gpd.GeoDataFrame(tiles, crs="EPSG:2154")
    tiles.to_parquet(out / "tiles.parquet")
    gpd.GeoDataFrame(wins, crs="EPSG:2154").to_parquet(out / "windows.parquet")
    return tiles


def main() -> None:
    p = argparse.ArgumentParser(description="Write tiles.parquet and windows.parquet.")
    p.add_argument("cog_dir", help="folder of <dept>/<tile>.tif")
    p.add_argument("out")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    tiles = build_index(a.cog_dir, a.out, a.workers)
    stats = tiles.groupby("fold").agg(tiles=("tile_id", "size"), zones=("zone_id", "nunique"))
    print(stats.to_string(), f"\n{len(tiles)} tiles → {a.out}")


if __name__ == "__main__":
    main()
