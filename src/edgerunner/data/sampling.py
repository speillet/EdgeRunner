"""Choix des fenêtres et adaptation du hasard NumPy amont, sans lecture de fichiers."""

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from math import ceil

import numpy as np


def eval_grid(
    windows: Iterable[tuple[str, int, int, int]],
    side: int,
    seed: int = 0,
    max_patches: int | None = None,
) -> list[tuple[str, int, int]]:
    """Découpe (tile_id, row, col, size), puis conserve un sous-ensemble fixe."""
    out = []
    for tile_id, row, col, size in windows:
        if size < side:
            continue
        offsets = np.linspace(0, size - side, ceil(size / side)).round().astype(int)
        out += [(tile_id, int(row + dr), int(col + dc)) for dr in offsets for dc in offsets]
    if max_patches and len(out) > max_patches:
        keep = np.random.default_rng(seed).choice(len(out), max_patches, replace=False)
        out = [out[i] for i in sorted(keep)]
    return out


def random_window(
    shapes: Sequence[tuple[int, int]],
    probabilities: np.ndarray,
    side: int,
    rng: np.random.Generator,
    valid_fraction: Callable[[int, int, int], float],
    min_valid: float = 0.5,
    max_tries: int = 20,
) -> tuple[int, int, int]:
    """Tire (index de dalle, row, col) ; après max_tries, conserve le dernier tirage."""
    for attempt in range(max_tries):
        index = int(rng.choice(len(shapes), p=probabilities))
        height, width = shapes[index]
        row = int(rng.integers(0, height - side + 1))
        col = int(rng.integers(0, width - side + 1))
        if attempt == max_tries - 1 or valid_fraction(index, row, col) >= min_valid:
            return index, row, col
    raise ValueError("max_tries must be >= 1")


@contextmanager
def numpy_seed(seed: int | None) -> Iterator[None]:
    """Isole le RNG global utilisé par SAM-Road, restauré même en cas d'erreur.

    À utiliser dans les workers (un processus par flux), pas depuis plusieurs threads.
    """
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        yield
    finally:
        np.random.set_state(state)
