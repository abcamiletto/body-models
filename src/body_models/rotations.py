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
