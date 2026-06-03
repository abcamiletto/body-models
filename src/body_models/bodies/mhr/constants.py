from body_models.common.pose_assets import load_npz
from body_models.constants import Joint


MHR_BODY_POSE_DIM = 100
MHR_HAND_POSE_DIM = 104
MHR_BODY_POSE_SPLIT = 68
MHR_HAND_POSE_SPLIT = 54

MHR_JOINTS = {
    Joint.LEFT_SHOULDER: "l_uparm",
    Joint.RIGHT_SHOULDER: "r_uparm",
    Joint.LEFT_ELBOW: "l_lowarm",
    Joint.RIGHT_ELBOW: "r_lowarm",
    Joint.LEFT_WRIST: "l_wrist",
    Joint.RIGHT_WRIST: "r_wrist",
    Joint.LEFT_THUMB_CMC: "l_thumb1",
    Joint.RIGHT_THUMB_CMC: "r_thumb1",
    Joint.LEFT_THUMB_MCP: "l_thumb2",
    Joint.RIGHT_THUMB_MCP: "r_thumb2",
    Joint.LEFT_THUMB_IP: "l_thumb3",
    Joint.RIGHT_THUMB_IP: "r_thumb3",
    Joint.LEFT_INDEX_MCP: "l_index1",
    Joint.RIGHT_INDEX_MCP: "r_index1",
    Joint.LEFT_INDEX_PIP: "l_index2",
    Joint.RIGHT_INDEX_PIP: "r_index2",
    Joint.LEFT_INDEX_DIP: "l_index3",
    Joint.RIGHT_INDEX_DIP: "r_index3",
    Joint.LEFT_MIDDLE_MCP: "l_middle1",
    Joint.RIGHT_MIDDLE_MCP: "r_middle1",
    Joint.LEFT_MIDDLE_PIP: "l_middle2",
    Joint.RIGHT_MIDDLE_PIP: "r_middle2",
    Joint.LEFT_MIDDLE_DIP: "l_middle3",
    Joint.RIGHT_MIDDLE_DIP: "r_middle3",
    Joint.LEFT_RING_MCP: "l_ring1",
    Joint.RIGHT_RING_MCP: "r_ring1",
    Joint.LEFT_RING_PIP: "l_ring2",
    Joint.RIGHT_RING_PIP: "r_ring2",
    Joint.LEFT_RING_DIP: "l_ring3",
    Joint.RIGHT_RING_DIP: "r_ring3",
    Joint.LEFT_PINKY_MCP: "l_pinky1",
    Joint.RIGHT_PINKY_MCP: "r_pinky1",
    Joint.LEFT_PINKY_PIP: "l_pinky2",
    Joint.RIGHT_PINKY_PIP: "r_pinky2",
    Joint.LEFT_PINKY_DIP: "l_pinky3",
    Joint.RIGHT_PINKY_DIP: "r_pinky3",
    Joint.LEFT_HIP: "l_upleg",
    Joint.RIGHT_HIP: "r_upleg",
    Joint.LEFT_KNEE: "l_lowleg",
    Joint.RIGHT_KNEE: "r_lowleg",
    Joint.LEFT_ANKLE: "l_foot",
    Joint.RIGHT_ANKLE: "r_foot",
    Joint.LEFT_FOOT: "l_ball",
    Joint.RIGHT_FOOT: "r_ball",
}

_POSES = load_npz("body_models.bodies.mhr")

MHR_BODY_PRESETS = _POSES["body"]
MHR_HAND_PRESETS = _POSES["hand"]

__all__ = [
    "MHR_BODY_POSE_DIM",
    "MHR_HAND_POSE_DIM",
    "MHR_BODY_POSE_SPLIT",
    "MHR_HAND_POSE_SPLIT",
    "MHR_JOINTS",
    "MHR_BODY_PRESETS",
    "MHR_HAND_PRESETS",
]
