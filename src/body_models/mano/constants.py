from body_models.constants import Joint


MANO_JOINT_NAMES = [
    "wrist",
    "index1",
    "index2",
    "index3",
    "middle1",
    "middle2",
    "middle3",
    "pinky1",
    "pinky2",
    "pinky3",
    "ring1",
    "ring2",
    "ring3",
    "thumb1",
    "thumb2",
    "thumb3",
]

LEFT_MANO_JOINTS = {
    Joint.LEFT_WRIST: "wrist",
    Joint.LEFT_THUMB_TIP: "thumb3",
    Joint.LEFT_INDEX_TIP: "index3",
    Joint.LEFT_MIDDLE_TIP: "middle3",
    Joint.LEFT_RING_TIP: "ring3",
    Joint.LEFT_PINKY_TIP: "pinky3",
}

RIGHT_MANO_JOINTS = {
    Joint.RIGHT_WRIST: "wrist",
    Joint.RIGHT_THUMB_TIP: "thumb3",
    Joint.RIGHT_INDEX_TIP: "index3",
    Joint.RIGHT_MIDDLE_TIP: "middle3",
    Joint.RIGHT_RING_TIP: "ring3",
    Joint.RIGHT_PINKY_TIP: "pinky3",
}

__all__ = ["MANO_JOINT_NAMES", "LEFT_MANO_JOINTS", "RIGHT_MANO_JOINTS"]
