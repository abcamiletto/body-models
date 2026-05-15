from body_models.common.pose_assets import load_npz
from body_models.constants import Joint


SMPLX_JOINTS = {
    Joint.LEFT_SHOULDER: "L_Shoulder",
    Joint.RIGHT_SHOULDER: "R_Shoulder",
    Joint.LEFT_ELBOW: "L_Elbow",
    Joint.RIGHT_ELBOW: "R_Elbow",
    Joint.LEFT_WRIST: "L_Wrist",
    Joint.RIGHT_WRIST: "R_Wrist",
    Joint.LEFT_THUMB_CMC: "L_Thumb1",
    Joint.RIGHT_THUMB_CMC: "R_Thumb1",
    Joint.LEFT_THUMB_MCP: "L_Thumb2",
    Joint.RIGHT_THUMB_MCP: "R_Thumb2",
    Joint.LEFT_THUMB_IP: "L_Thumb3",
    Joint.RIGHT_THUMB_IP: "R_Thumb3",
    Joint.LEFT_INDEX_MCP: "L_Index1",
    Joint.RIGHT_INDEX_MCP: "R_Index1",
    Joint.LEFT_INDEX_PIP: "L_Index2",
    Joint.RIGHT_INDEX_PIP: "R_Index2",
    Joint.LEFT_INDEX_DIP: "L_Index3",
    Joint.RIGHT_INDEX_DIP: "R_Index3",
    Joint.LEFT_MIDDLE_MCP: "L_Middle1",
    Joint.RIGHT_MIDDLE_MCP: "R_Middle1",
    Joint.LEFT_MIDDLE_PIP: "L_Middle2",
    Joint.RIGHT_MIDDLE_PIP: "R_Middle2",
    Joint.LEFT_MIDDLE_DIP: "L_Middle3",
    Joint.RIGHT_MIDDLE_DIP: "R_Middle3",
    Joint.LEFT_RING_MCP: "L_Ring1",
    Joint.RIGHT_RING_MCP: "R_Ring1",
    Joint.LEFT_RING_PIP: "L_Ring2",
    Joint.RIGHT_RING_PIP: "R_Ring2",
    Joint.LEFT_RING_DIP: "L_Ring3",
    Joint.RIGHT_RING_DIP: "R_Ring3",
    Joint.LEFT_PINKY_MCP: "L_Pinky1",
    Joint.RIGHT_PINKY_MCP: "R_Pinky1",
    Joint.LEFT_PINKY_PIP: "L_Pinky2",
    Joint.RIGHT_PINKY_PIP: "R_Pinky2",
    Joint.LEFT_PINKY_DIP: "L_Pinky3",
    Joint.RIGHT_PINKY_DIP: "R_Pinky3",
    Joint.LEFT_HIP: "L_Hip",
    Joint.RIGHT_HIP: "R_Hip",
    Joint.LEFT_KNEE: "L_Knee",
    Joint.RIGHT_KNEE: "R_Knee",
    Joint.LEFT_ANKLE: "L_Ankle",
    Joint.RIGHT_ANKLE: "R_Ankle",
    Joint.LEFT_FOOT: "L_Foot",
    Joint.RIGHT_FOOT: "R_Foot",
}

_POSES = load_npz("body_models.smplx")

SMPLX_BODY_PRESETS = _POSES["body"]
SMPLX_HAND_PRESETS = _POSES["hand"]

__all__ = ["SMPLX_JOINTS", "SMPLX_BODY_PRESETS", "SMPLX_HAND_PRESETS"]
