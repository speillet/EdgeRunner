"""Losses sur petits tenseurs CPU : aucun modèle, checkpoint ou fichier requis."""

import math

import pytest
import torch
from torch.nn import BCEWithLogitsLoss

from edgerunner.data.types import Targets
from edgerunner.models.losses import compute_losses


@pytest.mark.parametrize("empty", [False, True])
def test_mask_loss_ignores_invalid_pixels_and_preserves_gradients(empty: bool) -> None:
    logits = torch.zeros((1, 2, 2, 2), requires_grad=True)
    topo = torch.zeros((1, 1, 2, 1), requires_grad=True)
    pixels = torch.tensor([[[not empty, False], [False, False]]])
    targets: Targets = {
        "keypoint_mask": torch.zeros((1, 2, 2)),
        "road_mask": torch.ones((1, 2, 2)),
        "pixel_valid": pixels,
        "connected": torch.zeros((1, 1, 2), dtype=torch.bool),
        "valid": torch.zeros((1, 1, 2), dtype=torch.bool),
    }
    criterion = BCEWithLogitsLoss()
    mask_loss, topo_loss = compute_losses(
        logits, topo, targets, criterion, BCEWithLogitsLoss(reduction="none")
    )
    assert mask_loss.item() == pytest.approx(0 if empty else math.log(2))
    assert topo_loss.item() == 0
    targets["road_mask"][~pixels] = 0
    targets["keypoint_mask"][~pixels] = 1
    with torch.no_grad():
        logits[~pixels] = 20
    changed, _ = compute_losses(
        logits, topo, targets, criterion, BCEWithLogitsLoss(reduction="none")
    )
    torch.testing.assert_close(mask_loss, changed)
    (changed + topo_loss).backward()
    assert logits.grad[~pixels].count_nonzero() == 0
    assert topo.grad.count_nonzero() == 0
    if not empty:
        assert logits.grad[pixels].abs().sum() > 0


def test_topology_loss_is_normalized_by_valid_pairs() -> None:
    logits = torch.zeros((1, 1, 1, 2), requires_grad=True)
    topo = torch.tensor([[[[0.0], [math.log(3)], [20.0]]]], requires_grad=True)
    targets: Targets = {
        "keypoint_mask": torch.zeros((1, 1, 1)),
        "road_mask": torch.zeros((1, 1, 1)),
        "pixel_valid": torch.ones((1, 1, 1), dtype=torch.bool),
        "connected": torch.tensor([[[True, True, False]]]),
        "valid": torch.tensor([[[True, True, False]]]),
    }
    _, loss = compute_losses(
        logits, topo, targets, BCEWithLogitsLoss(), BCEWithLogitsLoss(reduction="none")
    )
    assert loss.item() == pytest.approx((-math.log(0.5) - math.log(0.75)) / 2)
    loss.backward()
    torch.testing.assert_close(topo.grad, torch.tensor([[[[-0.25], [-0.125], [0.0]]]]))
