"""BD TOPO road centrelines for one tile footprint, read on demand from the hive GeoParquet.

Expected layout (one directory, `dept=XX` partitions): EPSG:2154 WKB `geometry`, a struct column
`bbox` {xmin, ymin, xmax, ymax} (row groups are pruned on it) and the BD TOPO attributes used by
the filters below.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import shapely

# not visible from the air, or not a road
EXCLUDED_NATURES = ("Bac ou liaison maritime", "Sentier", "Escalier")
DEFAULT_WIDTH_M = {"Chemin": 3.0, "Route empierrée": 3.5}
FALLBACK_WIDTH_M = 4.0
COLUMNS = ["geometry", "nature", "largeur_de_chaussee", "position_par_rapport_au_sol"]


@dataclass
class Roads:
    geoms: np.ndarray  # shapely LineStrings, EPSG:2154
    width_m: np.ndarray  # carriageway width, default by nature when missing
    tree: shapely.STRtree

    @classmethod
    def from_geoms(cls, geoms: Any, width_m: Any = None) -> Roads:
        geoms = np.asarray(geoms, dtype=object)
        w = np.full(len(geoms), FALLBACK_WIDTH_M) if width_m is None else np.asarray(width_m, float)
        return cls(geoms, w, shapely.STRtree(geoms))

    def __len__(self) -> int:
        return len(self.geoms)


_DATASETS: dict[tuple[int, str], Any] = {}


def _dataset(vector_uri: str) -> Any:
    """One pyarrow dataset per process and uri (discovery lists the partitions once)."""
    import pyarrow.dataset as ds

    key = (os.getpid(), vector_uri)
    if key not in _DATASETS:
        _DATASETS[key] = ds.dataset(vector_uri, format="parquet", partitioning="hive")
    return _DATASETS[key]


def _wkb(col: Any) -> np.ndarray:
    arr = col.combine_chunks()
    arr = getattr(arr, "storage", arr)  # Parquet GEOMETRY / geoarrow.wkb extension → binary
    return shapely.from_wkb(arr.to_numpy(zero_copy_only=False))


def read_roads(vector_uri: str, bounds: tuple[float, float, float, float]) -> Roads:
    """Roads in service whose bbox intersects `bounds` (L93), tunnels and footpaths excluded.

    `vector_uri` is the whole region on purpose: a tile on a department border needs the roads
    of its neighbour.
    """
    import pyarrow.compute as pc

    xmin, ymin, xmax, ymax = bounds
    flt = (
        (pc.field("bbox", "xmax") >= xmin)
        & (pc.field("bbox", "xmin") <= xmax)
        & (pc.field("bbox", "ymax") >= ymin)
        & (pc.field("bbox", "ymin") <= ymax)
        & (pc.field("etat_de_l_objet") == "En service")
        & (pc.field("fictif") == False)  # noqa: E712
        & ~pc.field("nature").isin(list(EXCLUDED_NATURES))
    )
    tb = _dataset(vector_uri).to_table(columns=COLUMNS, filter=flt)
    pos = tb["position_par_rapport_au_sol"].to_pylist()
    keep = np.array([not str(p or "0").startswith("-") for p in pos], bool)  # tunnels
    geoms = _wkb(tb["geometry"])[keep]
    natures = np.array(tb["nature"].to_pylist(), dtype=object)[keep]
    w = np.array(
        [np.nan if v is None else v for v in tb["largeur_de_chaussee"].to_pylist()], float
    )[keep]
    default = np.array([DEFAULT_WIDTH_M.get(n, FALLBACK_WIDTH_M) for n in natures])
    return Roads.from_geoms(geoms, np.where(np.isfinite(w) & (w > 0), w, default))
