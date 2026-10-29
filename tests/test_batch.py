"""Contrat du collate, sur des exemples en mémoire."""

import torch

from edgerunner.data.batch import collate
from edgerunner.data.types import Sample


def test_collate_pads_points_and_preserves_fields() -> None:
    items: list[Sample] = [
        {
            "rgb": torch.zeros(2, 2, 3, dtype=torch.uint8),
            "road_mask": torch.zeros(2, 2, dtype=torch.uint8),
            "keypoint_mask": torch.zeros(2, 2, dtype=torch.uint8),
            "pixel_valid": torch.ones(2, 2, dtype=torch.bool),
            "graph_points": torch.ones(n, 2),
            "pairs": torch.zeros(1, 2, 2, dtype=torch.int32),
            "connected": torch.zeros(1, 2, dtype=torch.bool),
            "valid": torch.ones(1, 2, dtype=torch.bool),
            "tile_id": str(n),
            "row_off": n,
            "col_off": n + 1,
            "rot": 0,
        }
        for n in (3, 5)
    ]
    batch = collate(items)
    assert batch["graph_points"].shape == (2, 5, 2)
    assert batch["graph_points"][0, 3:].count_nonzero() == 0
    assert batch["graph_points"][0, :3].eq(1).all()
    assert batch["rgb"].shape == (2, 2, 2, 3) and batch["rgb"].dtype == torch.uint8
    assert batch["pixel_valid"].dtype == torch.bool and batch["pixel_valid"].all()
    assert batch["pairs"].shape == (2, 1, 2, 2)
    assert batch["valid"].all() and not batch["connected"].any()
    assert batch["tile_id"] == ["3", "5"] and batch["row_off"] == [3, 5]
    assert batch["col_off"] == [4, 6] and batch["rot"] == [0, 0]
