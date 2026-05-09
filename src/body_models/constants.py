from enum import StrEnum


class Joint(StrEnum):
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_THUMB_TIP = "left_thumb_tip"
    RIGHT_THUMB_TIP = "right_thumb_tip"
    LEFT_INDEX_TIP = "left_index_tip"
    RIGHT_INDEX_TIP = "right_index_tip"
    LEFT_MIDDLE_TIP = "left_middle_tip"
    RIGHT_MIDDLE_TIP = "right_middle_tip"
    LEFT_RING_TIP = "left_ring_tip"
    RIGHT_RING_TIP = "right_ring_tip"
    LEFT_PINKY_TIP = "left_pinky_tip"
    RIGHT_PINKY_TIP = "right_pinky_tip"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"
    LEFT_FOOT = "left_foot"
    RIGHT_FOOT = "right_foot"


__all__ = ["Joint"]
