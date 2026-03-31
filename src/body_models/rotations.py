"""Rotation type definitions for body models."""

from typing import Literal

RotationType = Literal["axis_angle", "quat", "sixd", "matrix", "rotmat"]

VALID_ROTATION_TYPES: tuple[RotationType, ...] = (
    "axis_angle",
    "quat",
    "sixd",
    "matrix",
    "rotmat",
)


def is_rotmat_type(rotation_type: RotationType) -> bool:
    """Return True for 3x3 rotation-matrix representations."""
    return rotation_type in ("matrix", "rotmat")
