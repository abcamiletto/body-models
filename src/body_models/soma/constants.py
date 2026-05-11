from body_models.constants import Joint


SOMA_JOINTS = {
    Joint.LEFT_SHOULDER: "LeftArm",
    Joint.RIGHT_SHOULDER: "RightArm",
    Joint.LEFT_ELBOW: "LeftForeArm",
    Joint.RIGHT_ELBOW: "RightForeArm",
    Joint.LEFT_WRIST: "LeftHand",
    Joint.RIGHT_WRIST: "RightHand",
    Joint.LEFT_THUMB_IP: "LeftHandThumbEnd",
    Joint.RIGHT_THUMB_IP: "RightHandThumbEnd",
    Joint.LEFT_INDEX_DIP: "LeftHandIndexEnd",
    Joint.RIGHT_INDEX_DIP: "RightHandIndexEnd",
    Joint.LEFT_MIDDLE_DIP: "LeftHandMiddleEnd",
    Joint.RIGHT_MIDDLE_DIP: "RightHandMiddleEnd",
    Joint.LEFT_RING_DIP: "LeftHandRingEnd",
    Joint.RIGHT_RING_DIP: "RightHandRingEnd",
    Joint.LEFT_PINKY_DIP: "LeftHandPinkyEnd",
    Joint.RIGHT_PINKY_DIP: "RightHandPinkyEnd",
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

__all__ = ["SOMA_JOINTS", "SOMA_APOSE", "SOMA_IPOSE"]
