from body_models.constants import Joint


SMPLX_JOINTS = {
    Joint.LEFT_SHOULDER: "L_Shoulder",
    Joint.RIGHT_SHOULDER: "R_Shoulder",
    Joint.LEFT_ELBOW: "L_Elbow",
    Joint.RIGHT_ELBOW: "R_Elbow",
    Joint.LEFT_WRIST: "L_Wrist",
    Joint.RIGHT_WRIST: "R_Wrist",
    Joint.LEFT_THUMB_TIP: "L_Thumb3",
    Joint.RIGHT_THUMB_TIP: "R_Thumb3",
    Joint.LEFT_INDEX_TIP: "L_Index3",
    Joint.RIGHT_INDEX_TIP: "R_Index3",
    Joint.LEFT_MIDDLE_TIP: "L_Middle3",
    Joint.RIGHT_MIDDLE_TIP: "R_Middle3",
    Joint.LEFT_RING_TIP: "L_Ring3",
    Joint.RIGHT_RING_TIP: "R_Ring3",
    Joint.LEFT_PINKY_TIP: "L_Pinky3",
    Joint.RIGHT_PINKY_TIP: "R_Pinky3",
    Joint.LEFT_HIP: "L_Hip",
    Joint.RIGHT_HIP: "R_Hip",
    Joint.LEFT_KNEE: "L_Knee",
    Joint.RIGHT_KNEE: "R_Knee",
    Joint.LEFT_ANKLE: "L_Ankle",
    Joint.RIGHT_ANKLE: "R_Ankle",
    Joint.LEFT_FOOT: "L_Foot",
    Joint.RIGHT_FOOT: "R_Foot",
}


__all__ = ["SMPLX_JOINTS"]
