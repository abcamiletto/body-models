"""Typed motion parameter mappings for body models."""

from typing import Any, NotRequired, TypedDict

from jaxtyping import Float

Array = Any


class AnnyMotion(TypedDict):
    body_pose: Float[Array, "*batch 64 N"] | Float[Array, "*batch 64 3 3"]
    head_pose: Float[Array, "*batch 60 N"] | Float[Array, "*batch 60 3 3"]
    hand_pose: Float[Array, "*batch 38 N"] | Float[Array, "*batch 38 3 3"]
    global_rotation: NotRequired[Float[Array, "*batch N"] | Float[Array, "*batch 3 3"]]
    global_translation: NotRequired[Float[Array, "*batch 3"]]


class FlameMotion(TypedDict):
    head_pose: Float[Array, "*batch 4 N"] | Float[Array, "*batch 4 3 3"]
    head_rotation: NotRequired[Float[Array, "*batch N"] | Float[Array, "*batch 3 3"]]
    global_rotation: NotRequired[Float[Array, "*batch N"] | Float[Array, "*batch 3 3"]]
    global_translation: NotRequired[Float[Array, "*batch 3"]]


class GarmentMeasurementsMotion(TypedDict):
    body_pose: Float[Array, "*batch 25 N"] | Float[Array, "*batch 25 3 3"]
    head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"]
    hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"]
    pelvis_rotation: NotRequired[Float[Array, "*batch N"] | Float[Array, "*batch 3 3"]]
    global_rotation: NotRequired[Float[Array, "*batch N"] | Float[Array, "*batch 3 3"]]
    global_translation: NotRequired[Float[Array, "*batch 3"]]


class GnmMotion(TypedDict):
    head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"]
    head_rotation: NotRequired[Float[Array, "*batch N"] | Float[Array, "*batch 3 3"]]
    global_rotation: NotRequired[Float[Array, "*batch N"] | Float[Array, "*batch 3 3"]]
    global_translation: NotRequired[Float[Array, "*batch 3"]]


class ManoMotion(TypedDict):
    hand_pose: Float[Array, "*batch 15 N"] | Float[Array, "*batch 15 3 3"]
    wrist_rotation: NotRequired[Float[Array, "*batch N"] | Float[Array, "*batch 3 3"]]
    global_rotation: NotRequired[Float[Array, "*batch N"] | Float[Array, "*batch 3 3"]]
    global_translation: NotRequired[Float[Array, "*batch 3"]]


class MhrMotion(TypedDict):
    body_pose: Float[Array, "*batch 94"]
    head_pose: Float[Array, "*batch 6"]
    hand_pose: Float[Array, "*batch 104"]
    global_rotation: NotRequired[Float[Array, "*batch 3"]]
    global_translation: NotRequired[Float[Array, "*batch 3"]]


class SkelMotion(TypedDict):
    body_pose: Float[Array, "*batch 43"]
    head_pose: Float[Array, "*batch 3"]
    global_rotation: NotRequired[Float[Array, "*batch 3"]]
    global_translation: NotRequired[Float[Array, "*batch 3"]]


class SmplMotion(TypedDict):
    body_pose: Float[Array, "*batch 23 N"] | Float[Array, "*batch 23 3 3"]
    pelvis_rotation: NotRequired[Float[Array, "*batch N"] | Float[Array, "*batch 3 3"]]
    global_rotation: NotRequired[Float[Array, "*batch N"] | Float[Array, "*batch 3 3"]]
    global_translation: NotRequired[Float[Array, "*batch 3"]]


class SmplhMotion(TypedDict):
    body_pose: Float[Array, "*batch 21 N"] | Float[Array, "*batch 21 3 3"]
    hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"]
    pelvis_rotation: NotRequired[Float[Array, "*batch N"] | Float[Array, "*batch 3 3"]]
    global_rotation: NotRequired[Float[Array, "*batch N"] | Float[Array, "*batch 3 3"]]
    global_translation: NotRequired[Float[Array, "*batch 3"]]


class SmplxMotion(TypedDict):
    body_pose: Float[Array, "*batch 21 N"] | Float[Array, "*batch 21 3 3"]
    head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"]
    hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"]
    pelvis_rotation: NotRequired[Float[Array, "*batch N"] | Float[Array, "*batch 3 3"]]
    global_rotation: NotRequired[Float[Array, "*batch N"] | Float[Array, "*batch 3 3"]]
    global_translation: NotRequired[Float[Array, "*batch 3"]]


class SomaMotion(TypedDict):
    body_pose: Float[Array, "*batch 23 N"] | Float[Array, "*batch 23 3 3"]
    head_pose: Float[Array, "*batch 5 N"] | Float[Array, "*batch 5 3 3"]
    hand_pose: Float[Array, "*batch 48 N"] | Float[Array, "*batch 48 3 3"]
    global_rotation: NotRequired[Float[Array, "*batch N"] | Float[Array, "*batch 3 3"]]
    global_translation: NotRequired[Float[Array, "*batch 3"]]


__all__ = [
    "AnnyMotion",
    "FlameMotion",
    "GarmentMeasurementsMotion",
    "GnmMotion",
    "ManoMotion",
    "MhrMotion",
    "SkelMotion",
    "SmplMotion",
    "SmplhMotion",
    "SmplxMotion",
    "SomaMotion",
]
