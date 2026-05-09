from body_models.constants import Joint


MHR_JOINTS = {
    Joint.LEFT_SHOULDER: "l_uparm",
    Joint.RIGHT_SHOULDER: "r_uparm",
    Joint.LEFT_ELBOW: "l_lowarm",
    Joint.RIGHT_ELBOW: "r_lowarm",
    Joint.LEFT_WRIST: "l_wrist",
    Joint.RIGHT_WRIST: "r_wrist",
    Joint.LEFT_THUMB_TIP: "l_thumb_null",
    Joint.RIGHT_THUMB_TIP: "r_thumb_null",
    Joint.LEFT_INDEX_TIP: "l_index_null",
    Joint.RIGHT_INDEX_TIP: "r_index_null",
    Joint.LEFT_MIDDLE_TIP: "l_middle_null",
    Joint.RIGHT_MIDDLE_TIP: "r_middle_null",
    Joint.LEFT_RING_TIP: "l_ring_null",
    Joint.RIGHT_RING_TIP: "r_ring_null",
    Joint.LEFT_PINKY_TIP: "l_pinky_null",
    Joint.RIGHT_PINKY_TIP: "r_pinky_null",
    Joint.LEFT_HIP: "l_upleg",
    Joint.RIGHT_HIP: "r_upleg",
    Joint.LEFT_KNEE: "l_lowleg",
    Joint.RIGHT_KNEE: "r_lowleg",
    Joint.LEFT_ANKLE: "l_foot",
    Joint.RIGHT_ANKLE: "r_foot",
    Joint.LEFT_FOOT: "l_ball",
    Joint.RIGHT_FOOT: "r_ball",
}


__all__ = ["MHR_JOINTS"]
