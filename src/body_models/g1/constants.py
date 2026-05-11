from body_models.constants import Joint


G1_JOINTS = {
    Joint.LEFT_SHOULDER: "left_shoulder_pitch_skel",
    Joint.RIGHT_SHOULDER: "right_shoulder_pitch_skel",
    Joint.LEFT_ELBOW: "left_elbow_skel",
    Joint.RIGHT_ELBOW: "right_elbow_skel",
    Joint.LEFT_WRIST: "left_wrist_roll_skel",
    Joint.RIGHT_WRIST: "right_wrist_roll_skel",
    Joint.LEFT_HIP: "left_hip_pitch_skel",
    Joint.RIGHT_HIP: "right_hip_pitch_skel",
    Joint.LEFT_KNEE: "left_knee_skel",
    Joint.RIGHT_KNEE: "right_knee_skel",
    Joint.LEFT_ANKLE: "left_ankle_pitch_skel",
    Joint.RIGHT_ANKLE: "right_ankle_pitch_skel",
    Joint.LEFT_FOOT: "left_toe_base",
    Joint.RIGHT_FOOT: "right_toe_base",
}


G1_TPOSE = {16: 1.5707963267948966, 23: -1.5707963267948966, 18: 1.0, 25: 1.0}
G1_APOSE = {16: 0.75, 23: -0.75, 18: 1.0, 25: 1.0}
G1_IPOSE = {15: 0.6, 22: 0.6, 16: 0.2, 23: -0.2, 18: 1.0, 25: 1.0}

__all__ = ["G1_JOINTS", "G1_TPOSE", "G1_APOSE", "G1_IPOSE"]
