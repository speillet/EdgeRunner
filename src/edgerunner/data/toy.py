"""Tiny synthetic IGN-like dataset for tests and local dev runs (the real data lives on the cluster).

Same layout as the real one: `cog/99/toy_<i>.tif` (tiled GeoTIFF + overviews, EPSG:2154, 0.20 m),
`bdtopo/troncon/dept=99/*.parquet` (WKB geometry + `bbox` struct + BD TOPO attributes),
`index/tiles.parquet` and `index/windows.parquet`. One tile per fold "0".."n-1".

    uv run python -m edgerunner.data.toy data/toy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import shapely

X0, Y_TOP = 300_000.0, 6_800_000.0  # Bretagne-ish Lambert-93 origin


def toy_roads(n_tiles: int, tile_m: float) -> list[dict]:
    """A road grid noded at its crossings, a bridge crossing it without nodes, and roads the
    loader must drop (tunnel, footpath, fictitious, out of service)."""
    rows, step = [], 100.0
    xs = np.arange(X0 + 50, X0 + n_tiles * tile_m, step)
    ys = np.arange(Y_TOP - 50, Y_TOP - tile_m, -step)

    def add(coords: list[tuple[float, float]], **kw: object) -> None:
        attrs = {"nature": "Route à 1 chaussée", "largeur_de_chaussee": 5.0,
                 "etat_de_l_objet": "En service", "fictif": False,
                 "position_par_rapport_au_sol": "0"}
        rows.append({**attrs, **kw, "geometry": shapely.LineString(coords)})

    for y in ys:  # one troncon per grid edge: crossings are shared vertices
        for xa, xb in zip(xs[:-1], xs[1:], strict=True):
            add([(xa, y), (xb, y)])
    for x in xs:
        for ya, yb in zip(ys[:-1], ys[1:], strict=True):
            add([(x, ya), (x, yb)], nature="Chemin", largeur_de_chaussee=None)
    # bridge: off the grid nodes (sam_road ignores crossings at segment ends)
    add([(X0 + 10, Y_TOP - 30), (X0 + tile_m - 30, Y_TOP - tile_m + 10)],
        position_par_rapport_au_sol="1")
    add([(X0 + 20, Y_TOP - 75), (X0 + n_tiles * tile_m - 20, Y_TOP - 75)],
        position_par_rapport_au_sol="-1")  # tunnel
    add([(X0 + 75, Y_TOP - 20), (X0 + 75, Y_TOP - tile_m + 20)], nature="Sentier")
    add([(X0 + 20, Y_TOP - 25), (X0 + 380, Y_TOP - 25)], fictif=True)
    add([(X0 + 20, Y_TOP - 35), (X0 + 380, Y_TOP - 35)], etat_de_l_objet="En projet")
    return rows


def _paint(size: int, gsd: float, x0: float, roads: list[dict], rng: np.random.Generator,
           nodata_cols: int) -> np.ndarray:
    """Smooth fields + grey roads at their width (the loader-kept ones only)."""
    low = rng.uniform(60, 160, (3, size // 64, size // 64)).astype(np.float32)
    img = np.stack([cv2.resize(b, (size, size), interpolation=cv2.INTER_CUBIC) for b in low])
    img = np.clip(img + rng.normal(0, 6, img.shape), 1, 255).astype(np.uint8)
    canvas = np.ascontiguousarray(np.moveaxis(img, 0, -1))
    for rd in roads:
        if (rd["nature"] == "Sentier" or rd["position_par_rapport_au_sol"].startswith("-")
                or rd["fictif"] or rd["etat_de_l_objet"] != "En service"):
            continue
        xy = shapely.get_coordinates(rd["geometry"])
        pts = np.stack([(xy[:, 0] - x0) / gsd, (Y_TOP - xy[:, 1]) / gsd], 1).round()
        width = max(1, round((rd["largeur_de_chaussee"] or 3.0) / gsd))
        cv2.polylines(canvas, [pts.astype(np.int32)], False, (128, 128, 128), thickness=width)
    if nodata_cols:
        canvas[:, size - nodata_cols:] = 0
    return np.moveaxis(canvas, -1, 0)


def make_toy_dataset(out: str | Path, n_tiles: int = 5, size: int = 3072, gsd: float = 0.2,
                     eval_size: int = 2048, seed: int = 0) -> Path:
    import geopandas as gpd
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin

    out = Path(out)
    rng = np.random.default_rng(seed)
    tile_m = size * gsd
    roads = toy_roads(n_tiles, tile_m)
    (out / "cog" / "99").mkdir(parents=True, exist_ok=True)
    tiles, wins = [], []
    for i in range(n_tiles):
        x0 = X0 + i * tile_m
        nodata = size // 4 if i == n_tiles - 1 else 0  # last tile: a nodata strip
        img = _paint(size, gsd, x0, roads, rng, nodata)
        uri = f"99/toy_{i}.tif"
        profile = {"driver": "GTiff", "width": size, "height": size, "count": 3,
                   "dtype": "uint8", "crs": "EPSG:2154", "transform": from_origin(x0, Y_TOP, gsd, gsd),
                   "tiled": True, "blockxsize": 256, "blockysize": 256, "compress": "deflate"}
        with rasterio.open(out / "cog" / uri, "w", **profile) as dst:
            dst.write(img)
            dst.build_overviews([2, 4, 8, 16], Resampling.average)
        valid = np.any(img != 0, axis=0)
        fold = str(i)
        tiles.append({"tile_id": f"toy_{i}", "uri": uri, "dept": "99", "zone_id": str(i),
                      "fold": fold, "width": size, "height": size,
                      "valid_frac": float(valid.mean()),
                      "geometry": shapely.box(x0, Y_TOP - tile_m, x0 + tile_m, Y_TOP)})
        for r in range(0, size - eval_size + 1, eval_size):
            for c in range(0, size - eval_size + 1, eval_size):
                vf = float(valid[r:r + eval_size, c:c + eval_size].mean())
                if vf >= 0.5:
                    wins.append({"tile_id": f"toy_{i}", "fold": fold, "zone_id": str(i),
                                 "row_off": r, "col_off": c, "size": eval_size, "valid_frac": vf,
                                 "geometry": shapely.box(x0 + c * gsd, Y_TOP - (r + eval_size) * gsd,
                                                         x0 + (c + eval_size) * gsd, Y_TOP - r * gsd)})

    (out / "index").mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(tiles, crs="EPSG:2154").to_parquet(out / "index" / "tiles.parquet")
    gpd.GeoDataFrame(wins, crs="EPSG:2154").to_parquet(out / "index" / "windows.parquet")
    part = out / "bdtopo" / "troncon" / "dept=99"
    part.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(roads, crs="EPSG:2154").to_parquet(part / "part-0.parquet",
                                                        write_covering_bbox=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out")
    p.add_argument("--tiles", type=int, default=5)
    p.add_argument("--size", type=int, default=3072)
    a = p.parse_args()
    print(make_toy_dataset(a.out, a.tiles, a.size))


if __name__ == "__main__":
    main()
