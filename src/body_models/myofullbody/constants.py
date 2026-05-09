from body_models.constants import Joint


MYOFULLBODY_JOINTS = {
    Joint.LEFT_SHOULDER: "humerus_l",
    Joint.RIGHT_SHOULDER: "humerus_r",
    Joint.LEFT_ELBOW: "ulna_l",
    Joint.RIGHT_ELBOW: "ulna_r",
    Joint.LEFT_WRIST: "lunate_l",
    Joint.RIGHT_WRIST: "lunate_r",
    Joint.LEFT_THUMB_TIP: "distal_thumb_l",
    Joint.RIGHT_THUMB_TIP: "distal_thumb_r",
    Joint.LEFT_INDEX_TIP: "2distph_l",
    Joint.RIGHT_INDEX_TIP: "2distph_r",
    Joint.LEFT_MIDDLE_TIP: "3distph_l",
    Joint.RIGHT_MIDDLE_TIP: "3distph_r",
    Joint.LEFT_RING_TIP: "4distph_l",
    Joint.RIGHT_RING_TIP: "4distph_r",
    Joint.LEFT_PINKY_TIP: "5distph_l",
    Joint.RIGHT_PINKY_TIP: "5distph_r",
    Joint.LEFT_HIP: "femur_l",
    Joint.RIGHT_HIP: "femur_r",
    Joint.LEFT_KNEE: "tibia_l",
    Joint.RIGHT_KNEE: "tibia_r",
    Joint.LEFT_ANKLE: "talus_l",
    Joint.RIGHT_ANKLE: "talus_r",
    Joint.LEFT_FOOT: "toes_l",
    Joint.RIGHT_FOOT: "toes_r",
}


__all__ = ["MYOFULLBODY_JOINTS"]
