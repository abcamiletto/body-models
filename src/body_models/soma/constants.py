from body_models.constants import Joint


SOMA_JOINTS = {
    Joint.LEFT_SHOULDER: "LeftArm",
    Joint.RIGHT_SHOULDER: "RightArm",
    Joint.LEFT_ELBOW: "LeftForeArm",
    Joint.RIGHT_ELBOW: "RightForeArm",
    Joint.LEFT_WRIST: "LeftHand",
    Joint.RIGHT_WRIST: "RightHand",
    Joint.LEFT_THUMB_TIP: "LeftHandThumbEnd",
    Joint.RIGHT_THUMB_TIP: "RightHandThumbEnd",
    Joint.LEFT_INDEX_TIP: "LeftHandIndexEnd",
    Joint.RIGHT_INDEX_TIP: "RightHandIndexEnd",
    Joint.LEFT_MIDDLE_TIP: "LeftHandMiddleEnd",
    Joint.RIGHT_MIDDLE_TIP: "RightHandMiddleEnd",
    Joint.LEFT_RING_TIP: "LeftHandRingEnd",
    Joint.RIGHT_RING_TIP: "RightHandRingEnd",
    Joint.LEFT_PINKY_TIP: "LeftHandPinkyEnd",
    Joint.RIGHT_PINKY_TIP: "RightHandPinkyEnd",
    Joint.LEFT_HIP: "LeftLeg",
    Joint.RIGHT_HIP: "RightLeg",
    Joint.LEFT_KNEE: "LeftShin",
    Joint.RIGHT_KNEE: "RightShin",
    Joint.LEFT_ANKLE: "LeftFoot",
    Joint.RIGHT_ANKLE: "RightFoot",
    Joint.LEFT_FOOT: "LeftToeBase",
    Joint.RIGHT_FOOT: "RightToeBase",
}


__all__ = ["SOMA_JOINTS"]
