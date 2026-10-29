"""Opérations sur les labels d'un exemple, indépendantes des entrées/sorties et de SAM."""

import numpy as np


def visible_pairs(points: np.ndarray, pairs: np.ndarray, pixel_valid: np.ndarray) -> np.ndarray:
    """Vrai si les deux points (x,y), arrondis au pixel le plus proche, sont visibles."""
    rc = np.floor(points[:, ::-1] + 0.5).astype(int)
    height, width = pixel_valid.shape
    inside = ((rc >= 0) & (rc < [height, width])).all(axis=1)
    clipped = np.clip(rc, 0, [height - 1, width - 1])
    point_valid = inside & pixel_valid[clipped[:, 0], clipped[:, 1]]
    return point_valid[pairs].all(axis=-1)
