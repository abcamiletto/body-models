"""Public MyoFullBody API."""

from body_models.rigid import RigidBodyParameters as Parameters
from body_models.skeletons.myofullbody import from_mujoco_qpos

__all__ = ["Parameters", "from_mujoco_qpos"]
