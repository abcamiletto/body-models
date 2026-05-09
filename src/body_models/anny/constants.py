from body_models.constants import Joint


ANNY_JOINTS = {
    Joint.LEFT_SHOULDER: "shoulder01.L",
    Joint.RIGHT_SHOULDER: "shoulder01.R",
    Joint.LEFT_ELBOW: "lowerarm01.L",
    Joint.RIGHT_ELBOW: "lowerarm01.R",
    Joint.LEFT_WRIST: "wrist.L",
    Joint.RIGHT_WRIST: "wrist.R",
    Joint.LEFT_THUMB_TIP: "finger1-3.L",
    Joint.RIGHT_THUMB_TIP: "finger1-3.R",
    Joint.LEFT_INDEX_TIP: "finger2-3.L",
    Joint.RIGHT_INDEX_TIP: "finger2-3.R",
    Joint.LEFT_MIDDLE_TIP: "finger3-3.L",
    Joint.RIGHT_MIDDLE_TIP: "finger3-3.R",
    Joint.LEFT_RING_TIP: "finger4-3.L",
    Joint.RIGHT_RING_TIP: "finger4-3.R",
    Joint.LEFT_PINKY_TIP: "finger5-3.L",
    Joint.RIGHT_PINKY_TIP: "finger5-3.R",
    Joint.LEFT_HIP: "upperleg01.L",
    Joint.RIGHT_HIP: "upperleg01.R",
    Joint.LEFT_KNEE: "lowerleg01.L",
    Joint.RIGHT_KNEE: "lowerleg01.R",
    Joint.LEFT_ANKLE: "foot.L",
    Joint.RIGHT_ANKLE: "foot.R",
    Joint.LEFT_FOOT: "toe2-1.L",
    Joint.RIGHT_FOOT: "toe2-1.R",
}


__all__ = ["ANNY_JOINTS"]
