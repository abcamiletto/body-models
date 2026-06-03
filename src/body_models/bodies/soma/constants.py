from body_models.common.pose_assets import load_npz
from body_models.constants import Joint


SOMA_JOINTS = {
    Joint.LEFT_SHOULDER: "LeftArm",
    Joint.RIGHT_SHOULDER: "RightArm",
    Joint.LEFT_ELBOW: "LeftForeArm",
    Joint.RIGHT_ELBOW: "RightForeArm",
    Joint.LEFT_WRIST: "LeftHand",
    Joint.RIGHT_WRIST: "RightHand",
    Joint.LEFT_THUMB_CMC: "LeftHandThumb1",
    Joint.RIGHT_THUMB_CMC: "RightHandThumb1",
    Joint.LEFT_THUMB_MCP: "LeftHandThumb2",
    Joint.RIGHT_THUMB_MCP: "RightHandThumb2",
    Joint.LEFT_THUMB_IP: "LeftHandThumb3",
    Joint.RIGHT_THUMB_IP: "RightHandThumb3",
    Joint.LEFT_INDEX_MCP: "LeftHandIndex2",
    Joint.RIGHT_INDEX_MCP: "RightHandIndex2",
    Joint.LEFT_INDEX_PIP: "LeftHandIndex3",
    Joint.RIGHT_INDEX_PIP: "RightHandIndex3",
    Joint.LEFT_INDEX_DIP: "LeftHandIndex4",
    Joint.RIGHT_INDEX_DIP: "RightHandIndex4",
    Joint.LEFT_MIDDLE_MCP: "LeftHandMiddle2",
    Joint.RIGHT_MIDDLE_MCP: "RightHandMiddle2",
    Joint.LEFT_MIDDLE_PIP: "LeftHandMiddle3",
    Joint.RIGHT_MIDDLE_PIP: "RightHandMiddle3",
    Joint.LEFT_MIDDLE_DIP: "LeftHandMiddle4",
    Joint.RIGHT_MIDDLE_DIP: "RightHandMiddle4",
    Joint.LEFT_RING_MCP: "LeftHandRing2",
    Joint.RIGHT_RING_MCP: "RightHandRing2",
    Joint.LEFT_RING_PIP: "LeftHandRing3",
    Joint.RIGHT_RING_PIP: "RightHandRing3",
    Joint.LEFT_RING_DIP: "LeftHandRing4",
    Joint.RIGHT_RING_DIP: "RightHandRing4",
    Joint.LEFT_PINKY_MCP: "LeftHandPinky2",
    Joint.RIGHT_PINKY_MCP: "RightHandPinky2",
    Joint.LEFT_PINKY_PIP: "LeftHandPinky3",
    Joint.RIGHT_PINKY_PIP: "RightHandPinky3",
    Joint.LEFT_PINKY_DIP: "LeftHandPinky4",
    Joint.RIGHT_PINKY_DIP: "RightHandPinky4",
    Joint.LEFT_HIP: "LeftLeg",
    Joint.RIGHT_HIP: "RightLeg",
    Joint.LEFT_KNEE: "LeftShin",
    Joint.RIGHT_KNEE: "RightShin",
    Joint.LEFT_ANKLE: "LeftFoot",
    Joint.RIGHT_ANKLE: "RightFoot",
    Joint.LEFT_FOOT: "LeftToeBase",
    Joint.RIGHT_FOOT: "RightToeBase",
}

_POSES = load_npz("body_models.bodies.soma")

SOMA_BODY_PRESETS = _POSES["body"]
SOMA_HAND_PRESETS = _POSES["hand"]

__all__ = [
    "SOMA_JOINTS",
    "SOMA_BODY_PRESETS",
    "SOMA_HAND_PRESETS",
]
