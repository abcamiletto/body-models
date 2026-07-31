"""Rotation representations accepted by body models."""

from typing import Literal

# `matrix` is an arbitrary 3×3 transform; `rotmat` is a proper SO(3) rotation.
RotationType = Literal["axis_angle", "quat", "sixd", "matrix", "rotmat"]

VALID_ROTATION_TYPES: tuple[RotationType, ...] = (
    "axis_angle",
    "quat",
    "sixd",
    "matrix",
    "rotmat",
)


def rotation_ndim(rotation_type: RotationType) -> int:
    """Return the number of trailing dimensions in one encoded rotation."""
    if rotation_type not in VALID_ROTATION_TYPES:
        raise ValueError(f"Invalid rotation_type: {rotation_type!r}")
    return 2 if rotation_type in ("matrix", "rotmat") else 1


def rotation_shape(rotation_type: RotationType) -> tuple[int, ...]:
    """Return the array shape of one encoded rotation."""
    if rotation_type == "axis_angle":
        return (3,)
    if rotation_type == "quat":
        return (4,)
    if rotation_type == "sixd":
        return (6,)
    if rotation_type in ("matrix", "rotmat"):
        return (3, 3)
    raise ValueError(f"Invalid rotation_type: {rotation_type!r}")


__all__ = ["VALID_ROTATION_TYPES", "RotationType", "rotation_ndim", "rotation_shape"]
