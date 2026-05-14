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
    Joint.LEFT_INDEX_MCP: "LeftHandIndex1",
    Joint.RIGHT_INDEX_MCP: "RightHandIndex1",
    Joint.LEFT_INDEX_PIP: "LeftHandIndex2",
    Joint.RIGHT_INDEX_PIP: "RightHandIndex2",
    Joint.LEFT_INDEX_DIP: "LeftHandIndex3",
    Joint.RIGHT_INDEX_DIP: "RightHandIndex3",
    Joint.LEFT_MIDDLE_MCP: "LeftHandMiddle1",
    Joint.RIGHT_MIDDLE_MCP: "RightHandMiddle1",
    Joint.LEFT_MIDDLE_PIP: "LeftHandMiddle2",
    Joint.RIGHT_MIDDLE_PIP: "RightHandMiddle2",
    Joint.LEFT_MIDDLE_DIP: "LeftHandMiddle3",
    Joint.RIGHT_MIDDLE_DIP: "RightHandMiddle3",
    Joint.LEFT_RING_MCP: "LeftHandRing1",
    Joint.RIGHT_RING_MCP: "RightHandRing1",
    Joint.LEFT_RING_PIP: "LeftHandRing2",
    Joint.RIGHT_RING_PIP: "RightHandRing2",
    Joint.LEFT_RING_DIP: "LeftHandRing3",
    Joint.RIGHT_RING_DIP: "RightHandRing3",
    Joint.LEFT_PINKY_MCP: "LeftHandPinky1",
    Joint.RIGHT_PINKY_MCP: "RightHandPinky1",
    Joint.LEFT_PINKY_PIP: "LeftHandPinky2",
    Joint.RIGHT_PINKY_PIP: "RightHandPinky2",
    Joint.LEFT_PINKY_DIP: "LeftHandPinky3",
    Joint.RIGHT_PINKY_DIP: "RightHandPinky3",
    Joint.LEFT_HIP: "LeftLeg",
    Joint.RIGHT_HIP: "RightLeg",
    Joint.LEFT_KNEE: "LeftShin",
    Joint.RIGHT_KNEE: "RightShin",
    Joint.LEFT_ANKLE: "LeftFoot",
    Joint.RIGHT_ANKLE: "RightFoot",
    Joint.LEFT_FOOT: "LeftToeBase",
    Joint.RIGHT_FOOT: "RightToeBase",
}


SOMA_APOSE = {11: (0.0, 0.0, -0.55), 39: (0.0, 0.0, 0.55)}
SOMA_IPOSE = {12: (0.0, 0.0, -1.5), 40: (0.0, 0.0, 1.5)}

__all__ = [
    "SOMA_JOINTS",
    "SOMA_APOSE",
    "SOMA_IPOSE",
]
