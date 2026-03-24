"""Compatibility helpers for nanomanifold rotation representations."""

from typing import Any, Literal

from nanomanifold import SO3

RotationType = Literal["axis_angle", "quat", "quat_wxyz", "quat_xyzw", "sixd", "matrix", "rotmat"]

VALID_ROTATION_TYPES: tuple[RotationType, ...] = (
    "axis_angle",
    "quat",
    "quat_wxyz",
    "quat_xyzw",
    "sixd",
    "matrix",
    "rotmat",
)

_ROTATION_TYPE_ALIASES = {
    "quat": "quat_wxyz",
    "matrix": "rotmat",
}


def to_nanomanifold_rotation_type(rotation_type: str) -> str:
    """Map repo aliases onto nanomanifold 0.5 rotation representation names."""
    return _ROTATION_TYPE_ALIASES.get(rotation_type, rotation_type)


def is_rotmat_type(rotation_type: RotationType) -> bool:
    """Return True for 3x3 rotation-matrix representations."""
    return rotation_type in ("matrix", "rotmat")


def convert(
    value,
    *,
    src: str,
    dst: str,
    src_convention: str | None = None,
    dst_convention: str | None = None,
    xp: Any = None,
):
    """Convert rotations while preserving legacy repo aliases."""
    return SO3.convert(
        value,
        src=to_nanomanifold_rotation_type(src),
        dst=to_nanomanifold_rotation_type(dst),
        src_convention=src_convention,
        dst_convention=dst_convention,
        xp=xp,
    )


def identity_as(ref, *, batch_dims: tuple[int, ...], rotation_type: str = "quat", xp: Any = None):
    """Create identity rotations while preserving legacy repo aliases."""
    return SO3.identity_as(
        ref,
        batch_dims=batch_dims,
        rotation_type=to_nanomanifold_rotation_type(rotation_type),
        xp=xp,
    )
