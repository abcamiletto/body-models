![Body model lineup](assets/readme/body-model-lineup.png)

# body-models

A unified library for body models.

Provides a shared interface across SMPL, SMPL-H, MANO, SMPL-X, SKEL, FLAME, ANNY, MHR, SOMA, GarmentMeasurements, G1, and MyoFullBody models with PyTorch, NumPy, and JAX backends.

## Features

- **Multi-backend**: PyTorch, NumPy, and JAX
- **Disentangled outputs**: separate `forward_vertices` (mesh) and `forward_skeleton` (joint transforms) for rigged models
- **Mesh simplification**: lower-resolution forward pass via `simplify` constructor argument
- **Vertex subsets**: compute only specific vertices via `vertex_indices` argument
- **Rotation representations**: axis-angle, quaternion, 6D, rotation matrix, and projected matrix (`rotation_type` constructor argument)

## Installation

```bash
pip install body-models
```

Or with uv:

```bash
uv add body-models
```

### Optional backends

PyTorch and JAX are optional dependencies. Install them for the corresponding backends:

```bash
# For PyTorch backend
pip install body-models[torch]

# For JAX backend
pip install body-models[jax]
```

Note: NumPy/JAX backends can load MHR torch checkpoints without installing PyTorch.

## Model Setup

### Auto-download models

ANNY, MHR, SOMA, GarmentMeasurements, G1, and MyoFullBody models are automatically downloaded on first use:

```python
from body_models.anny.torch import ANNY
from body_models.g1.torch import G1
from body_models.garment_measurements.torch import GarmentMeasurements
from body_models.mhr.torch import MHR
from body_models.myofullbody.torch import MyoFullBody
from body_models.soma.torch import SOMA

model = ANNY()  # Downloads automatically (CC0 license)
model = G1()  # Downloads Unitree G1 MuJoCo assets from Hugging Face
model = GarmentMeasurements()  # Downloads upstream data, then preprocesses with bpy via uv
model = MHR()   # Downloads automatically (Apache 2.0)
model = MyoFullBody()  # Downloads MuscleMimic full-body MJCF + STLs (Apache 2.0)
model = SOMA()  # Downloads SOMA_neutral.npz from SOMA-X
```

You can also prefetch them and save the cache paths into config:

```bash
body-models download anny
body-models download g1
body-models download garment-measurements
body-models download mhr
body-models download myofullbody
body-models download soma
```

SOMA is implemented natively in `body-models`; it does not require installing `py-soma-x`.

### Registration-required models

SMPL, SMPL-H, MANO, SMPL-X, SKEL, and FLAME require registration. Download from:
- SMPL: https://smpl.is.tue.mpg.de/
- SMPL-H: https://mano.is.tue.mpg.de/
- MANO: https://mano.is.tue.mpg.de/
- SMPL-X: https://smpl-x.is.tue.mpg.de/
- SKEL: https://skel.is.tue.mpg.de/
- FLAME: https://flame.is.tue.mpg.de/

You can let the CLI download all supported models into the platform cache and save those paths into config:

```bash
body-models download anny
body-models download g1
body-models download garment-measurements
body-models download mhr
body-models download myofullbody
body-models download soma
body-models download smpl
body-models download smplh
body-models download mano
body-models download smplx
body-models download skel
body-models download flame
body-models download all
```

Or set credentials via environment variables first:

```bash
SMPL_USERNAME=you@example.com SMPL_PASSWORD=... body-models download smpl
SMPLH_USERNAME=you@example.com SMPLH_PASSWORD=... body-models download smplh
MANO_USERNAME=you@example.com MANO_PASSWORD=... body-models download mano
SMPLX_USERNAME=you@example.com SMPLX_PASSWORD=... body-models download smplx
SKEL_USERNAME=you@example.com SKEL_PASSWORD=... body-models download skel
FLAME_USERNAME=you@example.com FLAME_PASSWORD=... body-models download flame
```

SMPL `.pkl` and `.npz` files are both supported directly. SMPL-H supports the AMASS `model.npz` files and smplx-ready `.pkl` files. MANO supports `.pkl` and `.npz` files. You can also configure paths manually:

```bash
body-models set smpl-neutral /path/to/SMPL_NEUTRAL.pkl
body-models set smpl-male /path/to/SMPL_MALE.pkl
body-models set smpl-female /path/to/SMPL_FEMALE.pkl
body-models set smplh-neutral /path/to/smplh/neutral/model.npz
body-models set smplh-male /path/to/smplh/male/model.npz
body-models set smplh-female /path/to/smplh/female/model.npz
body-models set mano-right /path/to/MANO_RIGHT.pkl
body-models set mano-left /path/to/MANO_LEFT.pkl
body-models set smplx-neutral /path/to/SMPLX_NEUTRAL.npz
body-models set skel /path/to/skel_models_v1.1
body-models set flame /path/to/FLAME_NEUTRAL.pkl
body-models set soma /path/to/soma-assets
body-models set garment-measurements /path/to/GarmentMeasurements/data
```

Or pass file paths directly:

```python
from body_models.smpl.torch import SMPL

# Direct file path (no gender needed)
model = SMPL(model_path="/path/to/SMPL_NEUTRAL.pkl")

# From config using gender
model = SMPL(gender="neutral")  # Uses smpl-neutral config key
```

### Configuration

View current settings:

```bash
$ body-models
Config file: /path/to/config.toml  # Platform-dependent location

Current settings:
  smpl-male: /data/models/smpl/SMPL_MALE.pkl
  smpl-female: /data/models/smpl/SMPL_FEMALE.pkl
  smpl-neutral: /data/models/smpl/SMPL_NEUTRAL.pkl
  smplh-male: (not set)
  smplh-female: (not set)
  smplh-neutral: (not set)
  mano-right: (not set)
  mano-left: (not set)
  smplx-male: (not set)
  smplx-female: (not set)
  smplx-neutral: (not set)
  skel: (not set)
  flame: (not set)
  anny: (not set)
  g1: (not set)
  mhr: (not set)
  soma: (not set)
  garment-measurements: (not set)
  myofullbody: (not set)
```

Manage paths:

```bash
body-models set <model> <path>   # Set model path
body-models unset <model>        # Remove from config
body-models download <model>     # Download anny, g1, mhr, myofullbody, soma, smpl, smplh, mano, smplx, skel, flame, or all
```

## Quick Start

```python
# Import from specific backend (torch, numpy, or jax)
from body_models.smplx.torch import SMPLX

# Load model from config
model = SMPLX(gender="neutral")  # Uses smplx-neutral config key

# Or load from direct file path
model = SMPLX(model_path="/path/to/SMPLX_NEUTRAL.npz")

# Get default parameters
params = model.get_rest_pose(batch_dims=(4,))

# Generate mesh vertices
vertices = model.forward_vertices(**params)  # [4, V, 3]

# Get skeleton transforms
skeleton = model.forward_skeleton(**params)  # [4, J, 4, 4]
```

Available backends:
- `body_models.<model>.torch` - PyTorch (differentiable)
- `body_models.<model>.numpy` - NumPy
- `body_models.<model>.jax` - JAX/Flax (differentiable)

Some backends expose implementation kernels for performance-sensitive paths:

```python
from body_models.smpl.numpy import SMPL

model = SMPL(gender="neutral", kernel="numba")
```

## Common Interface

Rigged models inherit from `BodyModel` and share these properties:

| Property | Type | Description |
|----------|------|-------------|
| `num_joints` | `int` | Number of skeleton joints |
| `num_vertices` | `int` | Number of mesh vertices |
| `joint_names` | `list[str]` | Joint names |
| `faces` | `[F, 3]` | Mesh face indices |
| `skin_weights` | `[V, J]` | Skinning weights (raises for rigid models) |
| `rest_vertices` | `[V, 3]` | Vertices in rest pose |
| `is_rigid_body` | `bool` | `True` for rigid articulated models (G1, MyoFullBody) — meshes are rigidly attached to bodies, no LBS |
| `has_tendons` | `bool` | `True` if the model exposes MJCF muscle via-points + tendons (MyoFullBody) |

### Common Methods

```python
# Get zero-initialized parameters (model-specific keys)
params = model.get_rest_pose(batch_dims=(1,))

# Compute mesh vertices [B, V, 3] in meters
vertices = model.forward_vertices(**params)

# Compute joint transforms [B, J, 4, 4] in meters
transforms = model.forward_skeleton(**params)
```

### Mesh Simplification

Skinned mesh models support mesh simplification via the `simplify` constructor argument:

```python
# Reduce face count by half (2x simplification)
model = SMPL(gender="neutral", simplify=2.0)

# Reduce to ~1/4 of original faces
model = SMPLX(gender="neutral", simplify=4.0)
```

The `simplify` parameter is a divisor for the face count. Default is `1.0` (no simplification). Skinning weights and blend shapes are automatically mapped to the simplified mesh.

### Vertex Subsets

All mesh-based models support computing only specific vertices:

```python
# Only compute vertices 0, 100, 200
vertices = model.forward_vertices(**params, vertex_indices=[0, 100, 200])
# Returns [B, 3, 3] instead of [B, V, 3]
```

This avoids computing the full mesh when you only need a few vertices (e.g. for landmark loss).

### Rotation Representations

SMPL, SMPL-X, FLAME, and ANNY support multiple rotation representations via the `rotation_type` constructor argument:

```python
model = SMPL(gender="neutral", rotation_type="sixd")  # Use 6D rotations
```

Supported types:
| Type | Shape per joint | Description |
|------|----------------|-------------|
| `"axis_angle"` | `[3]` | Axis-angle (default) |
| `"quat"` | `[4]` | Quaternion (wxyz convention) |
| `"sixd"` | `[6]` | 6D continuous representation |
| `"rotmat"` | `[3, 3]` | Rotation matrix (assumed SO(3)) |
| `"matrix"` | `[3, 3]` | General 3x3 matrix (SVD-projected to SO(3)) |

The `"matrix"` type is useful when optimizing rotations without constraints -- inputs are projected to the nearest valid rotation matrix via SVD. The `"rotmat"` type assumes inputs are already valid rotation matrices and skips the projection.

### viser Export

All models can export skinned-mesh data directly for `viser.SceneApi.add_mesh_skinned()`:

```python
import viser
from body_models.smplx.torch import SMPLX

server = viser.ViserServer()
server.scene.set_up_direction("+y")

model = SMPLX(gender="neutral")
forward_kwargs = model.get_rest_pose(batch_dims=(1,))

body = server.scene.add_mesh_skinned(
    "/body",
    **model.to_viser_skinned_mesh(**forward_kwargs),
)

# Later, update only the bones from any forward_skeleton() parameter dict.
bones = model.to_viser_bones(**forward_kwargs)
body.bone_wxyzs = bones["bone_wxyzs"]
body.bone_positions = bones["bone_positions"]
```

- `forward_kwargs` is the same kwargs dict you would pass to `forward_vertices()` or `forward_skeleton()`.
- Use `to_viser_skinned_mesh(**forward_kwargs)` when the mesh or bind skeleton changes.
- Use `to_viser_bones(**forward_kwargs)` when you only want updated bone poses.
- Both helpers require `batch_dims=(1,)`, the full mesh, and the full joint set.

## Supported Models

### SMPL

The original parametric body model with 6890 vertices and 24 joints.

```python
from body_models.smpl.torch import SMPL  # or .numpy, .jax

model = SMPL(gender="neutral")  # "neutral", "male", or "female"

vertices = model.forward_vertices(
    shape,               # [B, 10] body shape betas
    body_pose,           # [B, 23, 3] axis-angle per joint
    pelvis_rotation,     # [B, 3] root joint rotation (optional)
    global_rotation,     # [B, 3] post-transform rotation (optional)
    global_translation,  # [B, 3] translation (optional)
)
```

### SMPL-H

SMPL body model with articulated MANO hands.

```python
from body_models.smplh.torch import SMPLH  # or .numpy, .jax

model = SMPLH(
    gender="neutral",     # "neutral", "male", or "female"
    flat_hand_mean=False, # Flat hands as mean pose
)

vertices = model.forward_vertices(
    shape,               # [B, 10] body shape betas
    body_pose,           # [B, 21, 3] axis-angle per body joint
    hand_pose,           # [B, 30, 3] axis-angle (left 15 + right 15)
    pelvis_rotation,     # [B, 3] root joint rotation (optional)
    global_rotation,     # [B, 3] post-transform rotation (optional)
    global_translation,  # [B, 3] translation (optional)
)
```

### SMPL-X

Expressive body model with articulated hands and facial expressions.

```python
from body_models.smplx.torch import SMPLX  # or .numpy, .jax

model = SMPLX(
    gender="neutral",     # "neutral", "male", or "female"
    flat_hand_mean=False, # Flat hands as mean pose
)

vertices = model.forward_vertices(
    shape,               # [B, 10] body shape betas
    body_pose,           # [B, 21, 3] axis-angle per body joint
    hand_pose,           # [B, 30, 3] axis-angle (left 15 + right 15)
    head_pose,           # [B, 3, 3] jaw + left eye + right eye
    expression,          # [B, 10] facial expression (optional)
    pelvis_rotation,     # [B, 3] root joint rotation (optional)
    global_rotation,     # [B, 3] post-transform rotation (optional)
    global_translation,  # [B, 3] translation (optional)
)
```

### SKEL

Anatomically realistic skeletal articulation based on OpenSim. Only "male" and "female" genders are supported (no "neutral").

```python
from body_models.skel.torch import SKEL  # or .numpy, .jax

model = SKEL(gender="male")  # "male" or "female" (no neutral)

vertices = model.forward_vertices(
    shape,               # [B, 10] body shape betas
    pose,                # [B, 46] anatomically constrained DOFs
    global_rotation,     # [B, 3] axis-angle (optional)
    global_translation,  # [B, 3] (optional)
)
```

### FLAME

FLAME (Faces Learned with an Articulated Model and Expressions) head model.

```python
from body_models.flame.torch import FLAME  # or .numpy, .jax

model = FLAME()  # Uses configured path

vertices = model.forward_vertices(
    shape,               # [B, 300] shape betas (can use fewer)
    expression,          # [B, 100] expression coefficients (optional)
    pose,                # [B, 4, 3] axis-angle for neck, jaw, left_eye, right_eye (optional)
    head_rotation,       # [B, 3] root joint rotation (optional)
    global_rotation,     # [B, 3] post-transform rotation (optional)
    global_translation,  # [B, 3] translation (optional)
)
```

### ANNY

Phenotype-based body model with intuitive shape parameters.

```python
from body_models.anny.torch import ANNY  # or .numpy, .jax

model = ANNY(
    rig="default",                # "default", "default_no_toes", "cmu_mb", "game_engine", "mixamo"
    topology="default",           # "default" or "makehuman"
    all_phenotypes=False,         # Include race/cupsize/firmness
    extrapolate_phenotypes=False, # Allow values outside [0, 1]
)

vertices = model.forward_vertices(
    gender,              # [B] in [0, 1] (0=male, 1=female)
    age,                 # [B] in [0, 1]
    muscle,              # [B] in [0, 1]
    weight,              # [B] in [0, 1]
    height,              # [B] in [0, 1]
    proportions,         # [B] in [0, 1]
    pose,                # [B, J, 3] axis-angle per joint
    global_rotation,     # [B, 3] axis-angle (optional)
    global_translation,  # [B, 3] (optional)
)
```

### MHR

Meta Human Renderer with neural pose correctives.

```python
from body_models.mhr.torch import MHR  # or .numpy, .jax

model = MHR(lod=1)  # Level of detail

vertices = model.forward_vertices(
    shape,               # [B, 45] identity blendshapes
    pose,                # [B, 204] pose parameters
    expression,          # [B, 72] facial expression (optional)
    global_rotation,     # [B, 3] axis-angle (optional)
    global_translation,  # [B, 3] (optional)
)
```

### GarmentMeasurements

GarmentCodeData PCA body shape model from `mbotsch/GarmentMeasurements`.

```python
from body_models.garment_measurements.torch import GarmentMeasurements  # or .numpy, .jax

model = GarmentMeasurements()

params = model.get_rest_pose(batch_dims=(1,))
params["shape"][:, 0] = 0.5
vertices = model.forward_vertices(
    params["shape"],              # [B, C] PCA weights in standard deviation units
    params["pose"],               # [B, J, 3] joint rotations
    params["global_rotation"],    # [B, 3] axis-angle
    params["global_translation"], # [B, 3]
)
skeleton = model.forward_skeleton(**params)
```

The runtime loads a preprocessed `garment_measurements.npz` asset containing the upstream PCA body mesh plus the FBX-derived skeleton, skinning weights, and mean-value-coordinate joint weights. You can pass either that generated file, a folder containing it, or the original upstream `GarmentMeasurements/data` folder. When an upstream folder is provided, `body-models` runs the self-contained PEP 723 generator through `uv` and stores the `.npz` in the platform cache.

```bash
uv run --python 3.11 --no-project src/body_models/garment_measurements/generate_asset.py \
  /path/to/GarmentMeasurements/data /path/to/generated/garment_measurements/model
body-models set garment-measurements /path/to/GarmentMeasurements/data
```

### G1

Unitree G1 as a rigid articulated model with STL link meshes attached to the Kimodo 34-joint skeleton.
`body_pose` controls the 29 XML hinge joints; pelvis/root motion is controlled by `global_rotation` and
`global_translation`, while the skeleton-only toe and hand-roll leaves stay at rest.

```python
import torch
from body_models import g1
from body_models.g1.torch import G1

model = G1(rotation_type="rotmat")  # Auto-downloads assets from Hugging Face if unset
mujoco_model = G1(rotation_type="rotmat", convention="mujoco")
params = model.get_rest_pose(batch_dims=(1,))

transforms = model.forward_skeleton(**params)  # [B, 34, 4, 4]
link_transforms = model.forward_links(**params)  # [B, num_links, 4, 4], ordered by model.link_names
vertices = model.forward_vertices(**params)    # rigid concatenated STL link vertices
torso_meshes = model.joint_meshes("waist_pitch_skel")
head_mesh = model.link_mesh("head_link.STL")

qpos = g1.to_mujoco_qpos(
    model,
    body_pose=params["body_pose"],
    global_translation=params["global_translation"],
)

hinge_model = G1(rotation_type="hinge")
body_pose = torch.zeros((1, len(hinge_model.qpos_joint_indices), 1))  # 29 XML hinge angles
vertices = hinge_model.forward_vertices(body_pose=body_pose)
```

Use `body-models download g1` to download and configure the assets explicitly.
When passed manually, `model_path` should contain `xml/g1.xml` and `meshes/g1/*.STL`.

### MyoFullBody

MuscleMimic / MyoSuite full-body musculoskeletal model parsed from MuJoCo MJCF
(`amathislab/musclemimic_models`). 101 body frames built from per-link STL
meshes, driven by 122 scalar joint coordinates (112 hinge + 10 slide) plus a
free-root pose. No muscle dynamics — body-models exposes only kinematics.

```python
import numpy as np
from body_models import myofullbody
from body_models.myofullbody.torch import MyoFullBody

model = MyoFullBody()  # Auto-downloads MJCF + STLs from GitHub on first use
params = model.get_rest_pose(batch_dims=(1,))

# body_pose is a flat scalar qpos vector in MJCF order (hinge angles + slide displacements).
qpos_idx = model.qpos_joint_names.index("hip_flexion_r")
params["body_pose"][:, qpos_idx] = 0.6

skeleton = model.forward_skeleton(**params)        # [B, 101, 4, 4]
links = model.forward_links(**params)              # [B, 102, 4, 4]
vertices = model.forward_vertices(**params)        # rigid concatenated STL verts

# Round-trip with MuJoCo qpos (free root + 122 joint coordinates):
qpos = myofullbody.to_mujoco_qpos(model, **params)        # [B, 7+122]
restored = myofullbody.from_mujoco_qpos(qpos)             # body_pose + global_*
```

When passed manually, `model_path` should be the upstream `musclemimic_models/model/`
directory (containing `body/myofullbody.xml` and the `meshes/`, `torso/`, `leg/`,
`arm/`, `head/`, `scene/` subtrees).

Muscle visualisation is exposed via the existing `forward_skeleton` output —
no new forward method. The model also stores `site_positions [S, 3]`,
`site_body_indices [S]`, and a `tendons` list (each entry has `name`,
`site_indices`, and `width`); call `model.world_sites(skeleton)` to lift the
body-local via-points into world space, then walk each tendon's `site_indices`
to draw straight-segment polylines (wrap surfaces are skipped). The bundled
`scripts/visualize_models.py` does this with `viser.add_line_segments`.

## Coordinate System

The unified API returns outputs in:
- **Y-up** coordinate system
- **Meters** as the unit

SMPL, SMPL-X, SKEL, FLAME, MHR, GarmentMeasurements, and SOMA are natively Y-up and pass through unchanged. ANNY (MakeHuman), G1 (MuJoCo), and MyoFullBody (MuJoCo) are natively Z-up and are rotated to Y-up at load time, so a single rendering or visualisation pipeline works across the entire lineup.

For G1, pass `convention="mujoco"` to keep the MuJoCo-native Z-up asset coordinates instead of the default `convention="soma"` Y-up coordinates.

## Development

```bash
uvx ruff format .   # Format code
uvx ruff check .    # Lint
uvx ty check        # Type check
```

## License

See individual model licenses for usage terms:
- SMPL: https://smpl.is.tue.mpg.de/
- SMPL-X: https://smpl-x.is.tue.mpg.de/
- SKEL: https://skel.is.tue.mpg.de/
- FLAME: https://flame.is.tue.mpg.de/
- ANNY: CC0 (MakeHuman data)
- MHR: Apache 2.0 (Meta Platforms, Inc.)
- SOMA: See NVIDIA SOMA-X license terms
- GarmentMeasurements: GPL-3.0
- G1: follows the license of the provided Unitree/Kimodo robot assets
- MyoFullBody: Apache 2.0 (MuscleMimic / MyoSuite)
