from body_models.constants import Joint


GARMENT_JOINTS = {
    Joint.LEFT_SHOULDER: "upper_arm_L",
    Joint.RIGHT_SHOULDER: "upper_arm_R",
    Joint.LEFT_ELBOW: "lower_arm_01_L",
    Joint.RIGHT_ELBOW: "lower_arm_01_R",
    Joint.LEFT_WRIST: "palm_L",
    Joint.RIGHT_WRIST: "palm_R",
    Joint.LEFT_THUMB_CMC: "thumb_01_L",
    Joint.RIGHT_THUMB_CMC: "thumb_01_R",
    Joint.LEFT_THUMB_MCP: "thumb_02_L",
    Joint.RIGHT_THUMB_MCP: "thumb_02_R",
    Joint.LEFT_THUMB_IP: "thumb_03_L",
    Joint.RIGHT_THUMB_IP: "thumb_03_R",
    Joint.LEFT_INDEX_MCP: "index_01_L",
    Joint.RIGHT_INDEX_MCP: "index_01_R",
    Joint.LEFT_INDEX_PIP: "index_02_L",
    Joint.RIGHT_INDEX_PIP: "index_02_R",
    Joint.LEFT_INDEX_DIP: "index_03_L",
    Joint.RIGHT_INDEX_DIP: "index_03_R",
    Joint.LEFT_MIDDLE_MCP: "middle_01_L",
    Joint.RIGHT_MIDDLE_MCP: "middle_01_R",
    Joint.LEFT_MIDDLE_PIP: "middle_02_L",
    Joint.RIGHT_MIDDLE_PIP: "middle_02_R",
    Joint.LEFT_MIDDLE_DIP: "middle_03_L",
    Joint.RIGHT_MIDDLE_DIP: "middle_03_R",
    Joint.LEFT_RING_MCP: "ring_01_L",
    Joint.RIGHT_RING_MCP: "ring_01_R",
    Joint.LEFT_RING_PIP: "ring_02_L",
    Joint.RIGHT_RING_PIP: "ring_02_R",
    Joint.LEFT_RING_DIP: "ring_03_L",
    Joint.RIGHT_RING_DIP: "ring_03_R",
    Joint.LEFT_PINKY_MCP: "pinky_01_L",
    Joint.RIGHT_PINKY_MCP: "pinky_01_R",
    Joint.LEFT_PINKY_PIP: "pinky_02_L",
    Joint.RIGHT_PINKY_PIP: "pinky_02_R",
    Joint.LEFT_PINKY_DIP: "pinky_03_L",
    Joint.RIGHT_PINKY_DIP: "pinky_03_R",
    Joint.LEFT_HIP: "thigh_L",
    Joint.RIGHT_HIP: "thigh_R",
    Joint.LEFT_KNEE: "calf_L",
    Joint.RIGHT_KNEE: "calf_R",
    Joint.LEFT_ANKLE: "foot_L",
    Joint.RIGHT_ANKLE: "foot_R",
    Joint.LEFT_FOOT: "toes_L",
    Joint.RIGHT_FOOT: "toes_R",
}


GARMENT_TPOSE = {
    "upper_arm_l": (0.0, 0.0, 0.6),
    "upper_arm_r": (0.0, 0.0, -0.6),
    "clavicle_l": (0.0, 0.0, 0.15),
    "clavicle_r": (0.0, 0.0, -0.15),
    "thigh_l": (0.0, 0.0, 0.12),
    "thigh_r": (0.0, 0.0, -0.12),
}
GARMENT_APOSE = {}
GARMENT_IPOSE = {
    "upper_arm_l": (0.0, 0.0, -0.35),
    "upper_arm_r": (0.0, 0.0, 0.35),
}

__all__ = ["GARMENT_JOINTS", "GARMENT_TPOSE", "GARMENT_APOSE", "GARMENT_IPOSE"]
