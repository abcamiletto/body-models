FLAME_JOINT_NAMES = ["root", "neck", "jaw", "left_eye", "right_eye"]
FLAME_KINEMATIC_ROTATIONS = {0: ("head_rotation", None)} | {
    i: ("head_pose", i - 1) for i in range(1, len(FLAME_JOINT_NAMES))
}

__all__ = ["FLAME_JOINT_NAMES", "FLAME_KINEMATIC_ROTATIONS"]
