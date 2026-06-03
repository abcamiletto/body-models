from body_models.common.pose_assets import load_npz
from body_models.constants import Joint


MANO_JOINT_NAMES = [
    "wrist",
    "index1",
    "index2",
    "index3",
    "middle1",
    "middle2",
    "middle3",
    "pinky1",
    "pinky2",
    "pinky3",
    "ring1",
    "ring2",
    "ring3",
    "thumb1",
    "thumb2",
    "thumb3",
]

LEFT_MANO_JOINTS = {
    Joint.LEFT_WRIST: "wrist",
    Joint.LEFT_THUMB_CMC: "thumb1",
    Joint.LEFT_THUMB_MCP: "thumb2",
    Joint.LEFT_THUMB_IP: "thumb3",
    Joint.LEFT_INDEX_MCP: "index1",
    Joint.LEFT_INDEX_PIP: "index2",
    Joint.LEFT_INDEX_DIP: "index3",
    Joint.LEFT_MIDDLE_MCP: "middle1",
    Joint.LEFT_MIDDLE_PIP: "middle2",
    Joint.LEFT_MIDDLE_DIP: "middle3",
    Joint.LEFT_RING_MCP: "ring1",
    Joint.LEFT_RING_PIP: "ring2",
    Joint.LEFT_RING_DIP: "ring3",
    Joint.LEFT_PINKY_MCP: "pinky1",
    Joint.LEFT_PINKY_PIP: "pinky2",
    Joint.LEFT_PINKY_DIP: "pinky3",
}

RIGHT_MANO_JOINTS = {
    Joint.RIGHT_WRIST: "wrist",
    Joint.RIGHT_THUMB_CMC: "thumb1",
    Joint.RIGHT_THUMB_MCP: "thumb2",
    Joint.RIGHT_THUMB_IP: "thumb3",
    Joint.RIGHT_INDEX_MCP: "index1",
    Joint.RIGHT_INDEX_PIP: "index2",
    Joint.RIGHT_INDEX_DIP: "index3",
    Joint.RIGHT_MIDDLE_MCP: "middle1",
    Joint.RIGHT_MIDDLE_PIP: "middle2",
    Joint.RIGHT_MIDDLE_DIP: "middle3",
    Joint.RIGHT_RING_MCP: "ring1",
    Joint.RIGHT_RING_PIP: "ring2",
    Joint.RIGHT_RING_DIP: "ring3",
    Joint.RIGHT_PINKY_MCP: "pinky1",
    Joint.RIGHT_PINKY_PIP: "pinky2",
    Joint.RIGHT_PINKY_DIP: "pinky3",
}

_POSES = load_npz("body_models.parts.mano")

MANO_HAND_PRESETS = {
    "left": _POSES["left"],
    "right": _POSES["right"],
}

__all__ = ["MANO_JOINT_NAMES", "LEFT_MANO_JOINTS", "RIGHT_MANO_JOINTS", "MANO_HAND_PRESETS"]
