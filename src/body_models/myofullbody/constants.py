from body_models.common.pose_assets import load_npz
from body_models.constants import Joint


MYOFULLBODY_JOINTS = {
    Joint.LEFT_SHOULDER: "humerus_l",
    Joint.RIGHT_SHOULDER: "humerus_r",
    Joint.LEFT_ELBOW: "ulna_l",
    Joint.RIGHT_ELBOW: "ulna_r",
    Joint.LEFT_WRIST: "lunate_l",
    Joint.RIGHT_WRIST: "lunate_r",
    Joint.LEFT_THUMB_MCP: "proximal_thumb_l",
    Joint.RIGHT_THUMB_MCP: "proximal_thumb_r",
    Joint.LEFT_THUMB_IP: "distal_thumb_l",
    Joint.RIGHT_THUMB_IP: "distal_thumb_r",
    Joint.LEFT_INDEX_MCP: "2proxph_l",
    Joint.RIGHT_INDEX_MCP: "2proxph_r",
    Joint.LEFT_INDEX_PIP: "2midph_l",
    Joint.RIGHT_INDEX_PIP: "2midph_r",
    Joint.LEFT_INDEX_DIP: "2distph_l",
    Joint.RIGHT_INDEX_DIP: "2distph_r",
    Joint.LEFT_MIDDLE_MCP: "3proxph_l",
    Joint.RIGHT_MIDDLE_MCP: "3proxph_r",
    Joint.LEFT_MIDDLE_PIP: "3midph_l",
    Joint.RIGHT_MIDDLE_PIP: "3midph_r",
    Joint.LEFT_MIDDLE_DIP: "3distph_l",
    Joint.RIGHT_MIDDLE_DIP: "3distph_r",
    Joint.LEFT_RING_MCP: "4proxph_l",
    Joint.RIGHT_RING_MCP: "4proxph_r",
    Joint.LEFT_RING_PIP: "4midph_l",
    Joint.RIGHT_RING_PIP: "4midph_r",
    Joint.LEFT_RING_DIP: "4distph_l",
    Joint.RIGHT_RING_DIP: "4distph_r",
    Joint.LEFT_PINKY_MCP: "5proxph_l",
    Joint.RIGHT_PINKY_MCP: "5proxph_r",
    Joint.LEFT_PINKY_PIP: "5midph_l",
    Joint.RIGHT_PINKY_PIP: "5midph_r",
    Joint.LEFT_PINKY_DIP: "5distph_l",
    Joint.RIGHT_PINKY_DIP: "5distph_r",
    Joint.LEFT_HIP: "femur_l",
    Joint.RIGHT_HIP: "femur_r",
    Joint.LEFT_KNEE: "tibia_l",
    Joint.RIGHT_KNEE: "tibia_r",
    Joint.LEFT_ANKLE: "talus_l",
    Joint.RIGHT_ANKLE: "talus_r",
    Joint.LEFT_FOOT: "toes_l",
    Joint.RIGHT_FOOT: "toes_r",
}


_POSES = load_npz("body_models.myofullbody")

MYOFULLBODY_BODY_PRESETS = _POSES["body"]

__all__ = ["MYOFULLBODY_JOINTS", "MYOFULLBODY_BODY_PRESETS"]
