import numpy as np

from edgerunner.data.transforms import visible_pairs


def test_visible_pairs_checks_both_endpoints_and_image_bounds() -> None:
    pixels = np.ones((3, 4), dtype=bool)
    pixels[1, 2] = False
    points = np.array([[0, 0], [1, 1], [2, 1], [-1, 0], [4, 1], [1, 3]])
    pairs = np.array([[[0, 1], [0, 2], [2, 1], [0, 3], [0, 4], [5, 1]]])
    assert visible_pairs(points, pairs, pixels).tolist() == [
        [True, False, False, False, False, False]
    ]
