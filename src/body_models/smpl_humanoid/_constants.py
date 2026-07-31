from body_models._constants import Joint

SMPL_HUMANOID_VARIANTS = (
    "mannequin",
    "mannequin_lod1",
    "mannequin_lod2",
    "humenv",
    "phc",
    "smplsim",
)


SMPL_HUMANOID_JOINTS = {
    Joint.PELVIS: "Pelvis",
    Joint.NECK: "Neck",
    Joint.HEAD: "Head",
    Joint.LEFT_HIP: "L_Hip",
    Joint.RIGHT_HIP: "R_Hip",
    Joint.LEFT_KNEE: "L_Knee",
    Joint.RIGHT_KNEE: "R_Knee",
    Joint.LEFT_ANKLE: "L_Ankle",
    Joint.RIGHT_ANKLE: "R_Ankle",
    Joint.LEFT_FOOT: "L_Toe",
    Joint.RIGHT_FOOT: "R_Toe",
    Joint.LEFT_SHOULDER: "L_Shoulder",
    Joint.RIGHT_SHOULDER: "R_Shoulder",
    Joint.LEFT_ELBOW: "L_Elbow",
    Joint.RIGHT_ELBOW: "R_Elbow",
    Joint.LEFT_WRIST: "L_Wrist",
    Joint.RIGHT_WRIST: "R_Wrist",
}


JOINT_NAMES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Torso",
    "L_Knee",
    "R_Knee",
    "Spine",
    "L_Ankle",
    "R_Ankle",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
]

PARENTS = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
]

FINGER_CHAINS = (
    ("L_Index1", "L_Index2", "L_Index3"),
    ("L_Middle1", "L_Middle2", "L_Middle3"),
    ("L_Pinky1", "L_Pinky2", "L_Pinky3"),
    ("L_Ring1", "L_Ring2", "L_Ring3"),
    ("L_Thumb1", "L_Thumb2", "L_Thumb3"),
    ("R_Index1", "R_Index2", "R_Index3"),
    ("R_Middle1", "R_Middle2", "R_Middle3"),
    ("R_Pinky1", "R_Pinky2", "R_Pinky3"),
    ("R_Ring1", "R_Ring2", "R_Ring3"),
    ("R_Thumb1", "R_Thumb2", "R_Thumb3"),
)
FINGER_JOINT_NAMES = tuple(name for chain in FINGER_CHAINS for name in chain)
ROBOT_JOINT_NAMES = [*JOINT_NAMES, *FINGER_JOINT_NAMES]
_ROBOT_JOINT_INDEX = {name: index for index, name in enumerate(ROBOT_JOINT_NAMES)}
ROBOT_PARENTS = PARENTS.copy()
for chain in FINGER_CHAINS:
    hand_parent = "L_Hand" if chain[0].startswith("L_") else "R_Hand"
    ROBOT_PARENTS.extend(
        [
            _ROBOT_JOINT_INDEX[hand_parent],
            _ROBOT_JOINT_INDEX[chain[0]],
            _ROBOT_JOINT_INDEX[chain[1]],
        ]
    )

# Each pair is the public body name and its index in the canonical SMPL body_pose array.
BODY_JOINTS = (
    ("L_Hip", 0),
    ("L_Knee", 3),
    ("L_Ankle", 6),
    ("L_Toe", 9),
    ("R_Hip", 1),
    ("R_Knee", 4),
    ("R_Ankle", 7),
    ("R_Toe", 10),
    ("Torso", 2),
    ("Spine", 5),
    ("Chest", 8),
    ("Neck", 11),
    ("Head", 14),
    ("L_Thorax", 12),
    ("L_Shoulder", 15),
    ("L_Elbow", 17),
    ("L_Wrist", 19),
    ("L_Hand", 21),
    ("R_Thorax", 13),
    ("R_Shoulder", 16),
    ("R_Elbow", 18),
    ("R_Wrist", 20),
    ("R_Hand", 22),
)


SMPL_BODY_PRESETS = {
    "t_pose": [[0.0, 0.0, 0.0] for _ in range(len(BODY_JOINTS))],
    "a_pose": [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.45],
        [0.0, 0.0, -0.45],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.35],
        [0.0, 0.0, -0.35],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ],
}


__all__ = [
    "BODY_JOINTS",
    "FINGER_CHAINS",
    "FINGER_JOINT_NAMES",
    "JOINT_NAMES",
    "PARENTS",
    "ROBOT_JOINT_NAMES",
    "ROBOT_PARENTS",
    "SMPL_BODY_PRESETS",
    "SMPL_HUMANOID_JOINTS",
    "SMPL_HUMANOID_VARIANTS",
]
