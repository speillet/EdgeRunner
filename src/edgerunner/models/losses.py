"""Calcul des losses sur tenseurs, indépendant du réseau et de Lightning."""

from collections.abc import Callable

import torch
from torch import Tensor

from ..data.types import Targets


def compute_losses(
    mask_logits: Tensor,
    topo_logits: Tensor,
    targets: Targets,
    mask_criterion: Callable[[Tensor, Tensor], Tensor],
    topo_criterion: Callable[[Tensor, Tensor], Tensor],
) -> tuple[Tensor, Tensor]:
    """Critère masque réduit en scalaire ; critère topo non réduit, comme dans SAM-Road.

    Les pixels sans image et les paires invalides ne contribuent pas. Les lots entièrement
    invalides conservent une loss nulle différentiable.
    """
    gt_masks = torch.stack([targets["keypoint_mask"], targets["road_mask"]], dim=-1)
    pixels = targets["pixel_valid"]
    if pixels.any():
        mask_loss = mask_criterion(mask_logits[pixels], gt_masks[pixels].float())
    else:
        mask_loss = mask_logits.sum() * 0.0
    topo = topo_criterion(topo_logits, targets["connected"].float().unsqueeze(-1))
    weights = targets["valid"].float().unsqueeze(-1)
    topo_loss = (topo * weights).sum() / weights.sum().clamp(min=1.0)
    return mask_loss, topo_loss
