"""Fenêtres et RNG sans raster, GeoParquet ni modèle."""

import numpy as np
import pytest

from edgerunner.data.sampling import eval_grid, numpy_seed, random_window


def test_eval_grid_covers_edges_and_skips_small_windows() -> None:
    assert eval_grid([("a", 10, 20, 6), ("b", 0, 0, 3)], side=4) == [
        ("a", 10, 20),
        ("a", 10, 22),
        ("a", 12, 20),
        ("a", 12, 22),
    ]


def test_eval_subset_is_fixed_and_keeps_original_order() -> None:
    windows = [("a", 0, 0, 16)]
    full = eval_grid(windows, side=4)
    subset = eval_grid(windows, side=4, seed=7, max_patches=5)
    assert subset == eval_grid(windows, side=4, seed=7, max_patches=5)
    assert len(subset) == len(set(subset)) == 5
    assert subset == sorted(subset, key=full.index)


@pytest.mark.parametrize("accepted", [False, True])
def test_random_window_retries_and_respects_tile_weights(accepted: bool) -> None:
    draws = []

    def valid_fraction(index: int, row: int, col: int) -> float:
        draws.append((index, row, col))
        return float(accepted)

    result = random_window(
        [(4, 4), (10, 12)],
        np.array([0.0, 1.0]),
        4,
        np.random.default_rng(5),
        valid_fraction,
        max_tries=3,
    )
    assert result[0] == 1 and 0 <= result[1] <= 6 and 0 <= result[2] <= 8
    assert len(draws) == (1 if accepted else 2)
    if accepted:
        assert result == draws[0]
    # The exhausted attempt is returned without another validity read, as before.


@pytest.mark.parametrize("raises", [False, True])
def test_numpy_seed_restores_global_state_even_on_error(raises: bool) -> None:
    before = np.random.get_state()
    expected_rng = np.random.RandomState()
    expected_rng.set_state(before)
    expected = expected_rng.random_sample(4)
    try:
        with numpy_seed(123):
            actual = np.random.random(4)
            np.testing.assert_array_equal(actual, np.random.RandomState(123).random_sample(4))
            if raises:
                raise RuntimeError("upstream failure")
    except RuntimeError:
        assert raises
    restored = np.random.RandomState()
    restored.set_state(np.random.get_state())
    np.testing.assert_array_equal(restored.random_sample(4), expected)
