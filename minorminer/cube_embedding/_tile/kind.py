"""Contains the kinds of tiles and the kinds of z-couplings the package supports."""

from enum import Enum

__all__ = ["TileKind", "ZCoupling"]


class TileKind(Enum):
    LADDER = 0  # Tile has a ladder shape.
    SQUARE = 1  # Tile has a square shape.


class ZCoupling(Enum):
    ZERO_ONE = 0
    ONE_ZERO = 1
    ZERO_ONE_ONE_ZERO = 2
    EITHER = 3
