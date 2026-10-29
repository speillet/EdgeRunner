"""Assemblage des exemples en batch, indépendant du dataset et de SAM."""

import torch

from .types import Batch, Sample


def collate(batch: list[Sample]) -> Batch:
    """Empile les tenseurs, complète les points de zéros et regroupe les métadonnées."""
    points = [item["graph_points"] for item in batch]
    n = max(v.shape[0] for v in points)
    return {
        "rgb": torch.stack([b["rgb"] for b in batch]),
        "keypoint_mask": torch.stack([b["keypoint_mask"] for b in batch]),
        "road_mask": torch.stack([b["road_mask"] for b in batch]),
        "pixel_valid": torch.stack([b["pixel_valid"] for b in batch]),
        "graph_points": torch.stack(
            [torch.cat([v, v.new_zeros(n - v.shape[0], 2)]) for v in points]
        ),
        "pairs": torch.stack([b["pairs"] for b in batch]),
        "connected": torch.stack([b["connected"] for b in batch]),
        "valid": torch.stack([b["valid"] for b in batch]),
        "tile_id": [b["tile_id"] for b in batch],
        "row_off": [b["row_off"] for b in batch],
        "col_off": [b["col_off"] for b in batch],
        "rot": [b["rot"] for b in batch],
    }
