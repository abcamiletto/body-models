from body_models.constants import Joint


LEFT_BRAINCO_JOINTS = {
    Joint.LEFT_WRIST: "left_base_skel",
    Joint.LEFT_THUMB_CMC: "left_thumb_metacarpal_skel",
    Joint.LEFT_THUMB_MCP: "left_thumb_proximal_skel",
    Joint.LEFT_THUMB_IP: "left_thumb_distal_skel",
    Joint.LEFT_INDEX_MCP: "left_index_proximal_skel",
    Joint.LEFT_INDEX_DIP: "left_index_distal_skel",
    Joint.LEFT_MIDDLE_MCP: "left_middle_proximal_skel",
    Joint.LEFT_MIDDLE_DIP: "left_middle_distal_skel",
    Joint.LEFT_RING_MCP: "left_ring_proximal_skel",
    Joint.LEFT_RING_DIP: "left_ring_distal_skel",
    Joint.LEFT_PINKY_MCP: "left_pinky_proximal_skel",
    Joint.LEFT_PINKY_DIP: "left_pinky_distal_skel",
}

RIGHT_BRAINCO_JOINTS = {
    Joint.RIGHT_WRIST: "right_base_skel",
    Joint.RIGHT_THUMB_CMC: "right_thumb_metacarpal_skel",
    Joint.RIGHT_THUMB_MCP: "right_thumb_proximal_skel",
    Joint.RIGHT_THUMB_IP: "right_thumb_distal_skel",
    Joint.RIGHT_INDEX_MCP: "right_index_proximal_skel",
    Joint.RIGHT_INDEX_DIP: "right_index_distal_skel",
    Joint.RIGHT_MIDDLE_MCP: "right_middle_proximal_skel",
    Joint.RIGHT_MIDDLE_DIP: "right_middle_distal_skel",
    Joint.RIGHT_RING_MCP: "right_ring_proximal_skel",
    Joint.RIGHT_RING_DIP: "right_ring_distal_skel",
    Joint.RIGHT_PINKY_MCP: "right_pinky_proximal_skel",
    Joint.RIGHT_PINKY_DIP: "right_pinky_distal_skel",
}


__all__ = ["LEFT_BRAINCO_JOINTS", "RIGHT_BRAINCO_JOINTS"]
