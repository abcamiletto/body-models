from body_models.constants import Joint


SKEL_JOINTS = {
    Joint.LEFT_SHOULDER: "humerus_l",
    Joint.RIGHT_SHOULDER: "humerus_r",
    Joint.LEFT_ELBOW: "ulna_l",
    Joint.RIGHT_ELBOW: "ulna_r",
    Joint.LEFT_WRIST: "hand_l",
    Joint.RIGHT_WRIST: "hand_r",
    Joint.LEFT_HIP: "femur_l",
    Joint.RIGHT_HIP: "femur_r",
    Joint.LEFT_KNEE: "tibia_l",
    Joint.RIGHT_KNEE: "tibia_r",
    Joint.LEFT_ANKLE: "talus_l",
    Joint.RIGHT_ANKLE: "talus_r",
    Joint.LEFT_FOOT: "toes_l",
    Joint.RIGHT_FOOT: "toes_r",
}


SKEL_APOSE = {29: 0.55, 39: -0.55}
SKEL_IPOSE = {29: 1.35, 39: -1.35}

__all__ = ["SKEL_JOINTS", "SKEL_APOSE", "SKEL_IPOSE"]
