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


__all__ = ["G1_JOINTS"]
