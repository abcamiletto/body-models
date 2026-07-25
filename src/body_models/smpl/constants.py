from body_models.common.pose_assets import load_npz
from body_models.constants import Joint


SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]

SMPL_JOINTS = {
    Joint.LEFT_SHOULDER: "left_shoulder",
    Joint.RIGHT_SHOULDER: "right_shoulder",
    Joint.LEFT_ELBOW: "left_elbow",
    Joint.RIGHT_ELBOW: "right_elbow",
    Joint.LEFT_WRIST: "left_wrist",
    Joint.RIGHT_WRIST: "right_wrist",
    Joint.LEFT_HIP: "left_hip",
    Joint.RIGHT_HIP: "right_hip",
    Joint.LEFT_KNEE: "left_knee",
    Joint.RIGHT_KNEE: "right_knee",
    Joint.LEFT_ANKLE: "left_ankle",
    Joint.RIGHT_ANKLE: "right_ankle",
    Joint.LEFT_FOOT: "left_foot",
    Joint.RIGHT_FOOT: "right_foot",
}

_POSES = load_npz("body_models.smpl")

SMPL_BODY_PRESETS = _POSES["body"]

__all__ = ["SMPL_JOINT_NAMES", "SMPL_JOINTS", "SMPL_BODY_PRESETS"]
