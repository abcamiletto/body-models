from body_models.constants import Joint


LEFT_BRAINCO_JOINTS = {
    Joint.LEFT_WRIST: "left_base_skel",
    Joint.LEFT_THUMB_TIP: "left_thumb_distal_skel",
    Joint.LEFT_INDEX_TIP: "left_index_distal_skel",
    Joint.LEFT_MIDDLE_TIP: "left_middle_distal_skel",
    Joint.LEFT_RING_TIP: "left_ring_distal_skel",
    Joint.LEFT_PINKY_TIP: "left_pinky_distal_skel",
}

RIGHT_BRAINCO_JOINTS = {
    Joint.RIGHT_WRIST: "right_base_skel",
    Joint.RIGHT_THUMB_TIP: "right_thumb_distal_skel",
    Joint.RIGHT_INDEX_TIP: "right_index_distal_skel",
    Joint.RIGHT_MIDDLE_TIP: "right_middle_distal_skel",
    Joint.RIGHT_RING_TIP: "right_ring_distal_skel",
    Joint.RIGHT_PINKY_TIP: "right_pinky_distal_skel",
}


__all__ = ["LEFT_BRAINCO_JOINTS", "RIGHT_BRAINCO_JOINTS"]
