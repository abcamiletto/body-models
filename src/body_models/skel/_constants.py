from body_models._common import pose_assets
from body_models._constants import Joint

SKEL_JOINTS = {
    Joint.PELVIS: "pelvis",
    Joint.HEAD: "head",
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


_POSES = pose_assets.load_npz("body_models.skel")

SKEL_BODY_PRESETS = _POSES["body"]

__all__ = ["SKEL_BODY_PRESETS", "SKEL_JOINTS"]
