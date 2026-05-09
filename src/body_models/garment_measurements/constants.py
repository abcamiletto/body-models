from body_models.constants import Joint


GARMENT_JOINTS = {
    Joint.LEFT_SHOULDER: "upper_arm_L",
    Joint.RIGHT_SHOULDER: "upper_arm_R",
    Joint.LEFT_ELBOW: "lower_arm_01_L",
    Joint.RIGHT_ELBOW: "lower_arm_01_R",
    Joint.LEFT_WRIST: "palm_L",
    Joint.RIGHT_WRIST: "palm_R",
    Joint.LEFT_THUMB_TIP: "thumb_03_L",
    Joint.RIGHT_THUMB_TIP: "thumb_03_R",
    Joint.LEFT_INDEX_TIP: "index_03_L",
    Joint.RIGHT_INDEX_TIP: "index_03_R",
    Joint.LEFT_MIDDLE_TIP: "middle_03_L",
    Joint.RIGHT_MIDDLE_TIP: "middle_03_R",
    Joint.LEFT_RING_TIP: "ring_03_L",
    Joint.RIGHT_RING_TIP: "ring_03_R",
    Joint.LEFT_PINKY_TIP: "pinky_03_L",
    Joint.RIGHT_PINKY_TIP: "pinky_03_R",
    Joint.LEFT_HIP: "thigh_L",
    Joint.RIGHT_HIP: "thigh_R",
    Joint.LEFT_KNEE: "calf_L",
    Joint.RIGHT_KNEE: "calf_R",
    Joint.LEFT_ANKLE: "foot_L",
    Joint.RIGHT_ANKLE: "foot_R",
    Joint.LEFT_FOOT: "toes_L",
    Joint.RIGHT_FOOT: "toes_R",
}


__all__ = ["GARMENT_JOINTS"]
