"""Contrats des tenseurs SAM-Road ; les métadonnées sont scalaires avant collate."""

from typing import TypedDict

from torch import Tensor


class Targets(TypedDict):
    keypoint_mask: Tensor  # uint8 [P,P] / float [B,P,P], {0,1}
    road_mask: Tensor
    pixel_valid: Tensor  # bool [P,P] / [B,P,P]
    connected: Tensor  # bool [S,Q] / [B,S,Q]
    valid: Tensor  # bool [S,Q] / [B,S,Q], inclut visibilité et padding


class TensorFields(Targets):
    rgb: Tensor  # uint8 [P,P,3] / float [B,P,P,3], 0..255
    graph_points: Tensor  # float32 [N,2] / [B,N,2], (x,y), complété de zéros
    pairs: Tensor  # int32 [S,Q,2] / [B,S,Q,2]


class Sample(TensorFields):
    tile_id: str
    row_off: int
    col_off: int
    rot: int


class Batch(TensorFields):
    tile_id: list[str]
    row_off: list[int]
    col_off: list[int]
    rot: list[int]
