# SMPL Robot: authored rigid character

This is a non-skinned robot generated from an SMPL identity. SMPL is used once,
offline, to measure the requested shape. The generated character is ordinary
MJCF containing:

- the canonical 24-joint SMPL body hierarchy;
- the same 23 local SMPL body rotations (69 scalar XYZ hinge coordinates);
- the canonical SMPL-X hand order: 15 articulated joints per hand, each with
  independent XYZ rotation (90 additional coordinates);
- shape-matched rigid joint offsets;
- a shape-conformed authored character split into 54 rigid links;
- sculpted five-finger hands with three rigid phalanges per digit;
- one warm pastel armor material and a closely related pastel joint material;
- no skin weights, blend shapes, or deformation at animation time.

The source character is purpose-built in Blender as a classic artist's
mannequin: a featureless egg head, rounded torso blocks, smooth tapered limb
shells, ball bearings at the major joints, and five modeled digits on each
hand. The hands use a sculpted palm with an integrated thenar form, tapered
phalanges, rounded fingertips, and recessed interphalangeal bearings. Its
front silhouette and vertical landmarks are calibrated against the
supplied mannequin reference image. The feet expose both the SMPL ankle and toe
links as separate rigid assemblies. Every exported Blender object encodes its
SMPL owner in its name, so runtime generation never cuts, skins, or
heuristically partitions the meshes. The palm includes a continuous sculpted
thenar surface; the invisible Thumb1 pivot is stored as mesh metadata rather
than rendered as a decorative bearing. For a new identity it reads only the SMPL
bone lengths, preserves the neutral bone directions, and stretches each body
link only along its longitudinal axis. Shell width/depth, head size, hand size,
finger thickness, and material design remain constant across identities. The
authored hand dimensions and finger offsets are intentionally unaffected by
SMPL body shape.

All authored left/right mesh vertices are exact sagittal reflections. SMPL's
small template asymmetries are removed during generation by averaging each
bilateral bone length and direction, then mirroring it. This applies to every
generated shape, not only the neutral preview.

Consequently an SMPL motion transfers with `SmplMannequin.parameters_from_smpl()`.
SMPL-X transfers natively by passing its 21 body rotations and 15 rotations
for each hand; the two terminal SMPL hand rotations are inserted
automatically.

## Build and inspect

```bash
uv run python examples/smpl_robot/generate_artifacts.py \
  --model /path/to/SMPL_NEUTRAL.npz \
  --smplx-model /path/to/SMPLX_NEUTRAL.npz \
  --output artifacts/smpl_robot

uv run --extra lod python examples/smpl_robot/generate_lods.py \
  artifacts/smpl_robot/neutral.xml artifacts/smpl_robot_lods

uv run --with matplotlib --with pillow \
  python examples/smpl_robot/render_previews.py artifacts/smpl_robot
```

The first command generates three differently shaped robots, GLB snapshots in
T/A/action poses, their standalone MJCF assets, and validation reports including
`fk_comparison.json`. The LOD command uses topology-preserving,
curvature-weighted optimal-placement quadric simplification and a global vertex
allocator to produce 40k, 15k, and 5k levels. Bilateral parts are optimized once
and reflected exactly; center parts are optimized as a half mesh and mirrored,
so every level is exactly sagittally symmetric. It also reports watertightness,
surface and normal error, symmetry, and FK equivalence in `manifest.json`.

Neutral SMPL-X defines the canonical joint directions and rest proportions;
SMPL shape coefficients change only relative bone lengths. The final command
produces PNG reviews and a contact sheet. A Blender studio renderer is also
included as `render_blender.py`.

## Move every joint interactively

Launch the rigid mannequin beside the real neutral SMPL-X mesh:

```bash
uv run --extra viewer python examples/smpl_robot/visualize_viser.py
```

Open `http://localhost:8080`. The sidebar exposes local XYZ rotation controls
for all 23 SMPL body joints and all 30 finger joints, global root controls,
T/A-pose presets, model visibility, and optional joint-center markers. Every
control drives both characters. The mannequin uses rigid forward kinematics;
SMPL-X uses its native linear-blend skinning so the comparison shows the same
pose with each representation's intended deformation. You can pass any
generated mannequin XML as the positional argument and override the configured
neutral SMPL-X asset with `--smplx-model /path/to/SMPLX_NEUTRAL.npz`.

## Play SMPL-X motions

Launch synchronized motion playback on the mannequin and skinned SMPL-X:

```bash
uv run --extra viewer python examples/smpl_robot/visualize_motion.py
```

The viewer downloads four small named samples directly from Hugging Face: an
[AMASS SMPL-X walking sequence from Habitat Humanoids](https://huggingface.co/datasets/ai-habitat/habitat_humanoids)
and three captioned SMPL-X-derived body-and-hand clips from
[MotionHub](https://huggingface.co/datasets/ZeyuLing/MotionHub). It does not
download either complete dataset. The sidebar provides motion selection,
play/pause, looping, timeline scrubbing, speed, root-motion, and visibility
controls.

Raw AMASS world motion is converted from Z-up to the viewer's Y-up frame;
MotionHub's normalized Y-up translation is preserved. Both characters receive
the same SMPL-X pelvis, 21 body joints, 30 hand joints, head pose, and root
translation. A static display offset reconciles the models' different root
origins without changing the motion. The SMPL-X body model itself is not
downloaded by this example; configure `smplx-neutral` with the `body-models set`
command or pass `--smplx-model /path/to/SMPLX_NEUTRAL.npz`.

To compare all three LODs with synchronized controls, run:

```bash
uv run --extra viewer python examples/smpl_robot/visualize_lod_comparison.py \
  artifacts/smpl_robot_lods/lod0_40k/neutral.xml \
  artifacts/smpl_robot_lods/lod1_15k/neutral.xml \
  artifacts/smpl_robot_lods/lod2_5k/neutral.xml
```

The downloadable sources use the same levels: `mannequin` selects LOD0,
`mannequin_lod1` selects LOD1, and `mannequin_lod2` selects LOD2.

The editable artist source is bundled as `smpl_robot_professional.blend`; run
`design_mannequin_character.py` through Blender 5.0+ to rebuild its GLB and
studio renders. Its `ARMOR_COLOR` and `JOINT_COLOR` constants are the only two
character palette controls.

Each MJCF references an adjacent `<name>_assets/` directory of indexed OBJ link
meshes. The generator preserves complete connected shells, merges shading
seams, and applies a conservative subdivision/smoothing pass before rigid
assignment. Keep the XML and that directory together.

## Generate one character

```python
import numpy as np
from body_models.smpl import SMPL
from body_models.smpl_humanoid import SmplMannequin, generate_smpl_robot
from body_models.smplx import SMPLX

smpl_model = SMPL(model_path="/path/to/SMPL_NEUTRAL.npz")
smplx_model = SMPLX(model_path="/path/to/SMPLX_NEUTRAL.npz", flat_hand_mean=True)
generate_smpl_robot(
    "my_robot.xml",
    source_model=smpl_model,
    smplx_model=smplx_model,
    shape=np.array([1.2, -2.0, 0.5, 0, 0, 0, 0, 0, 0, 0]),
)
robot = SmplMannequin(model_path="my_robot.xml")

# smplx_pose is [..., 21, 3] and each hand pose is [..., 15, 3] axis-angle.
robot_motion = robot.parameters_from_smpl(
    smplx_pose,
    global_translation=translation,
    global_rotation=root,
    left_hand_pose=left_hand_pose,
    right_hand_pose=right_hand_pose,
)
meshes = robot.forward_meshes(**robot_motion)
```
