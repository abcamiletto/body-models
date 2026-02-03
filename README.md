# body-models

A unified Python library for parametric human body models. Provides a common interface across SMPL, SMPL-X, SKEL, ANNY, and MHR body models.

## Installation

```bash
pip install body-models
```

Or with uv:

```bash
uv add body-models
```

## Model Setup

### Auto-download models

ANNY and MHR models are automatically downloaded on first use:

```python
from body_models import ANNY, MHR

model = ANNY()  # Downloads automatically (CC0 license)
model = MHR()   # Downloads automatically (Apache 2.0)
```

### Registration-required models

SMPL, SMPL-X, and SKEL require registration. Download from:
- SMPL: https://smpl.is.tue.mpg.de/
- SMPL-X: https://smpl-x.is.tue.mpg.de/
- SKEL: https://skel.is.tue.mpg.de/

For SMPL, convert the pkl file to npz format (the pkl files use chumpy):

```bash
convert-smpl-pkl /path/to/model.pkl /path/to/model.npz
```

Then configure the paths:

```bash
body-models set smpl /path/to/smpl
body-models set smplx /path/to/smplx
body-models set skel /path/to/skel
```

Or pass paths directly:

```python
model = SMPLX("/path/to/smplx", gender="neutral")
```

### Configuration

View current settings:

```bash
$ body-models
Config file: /path/to/config.toml  # Platform-dependent location

Current settings:
  smpl: /data/models/smpl
  smplx: (not set)
  skel: (not set)
  anny: (not set)
  mhr: (not set)
```

Manage paths:

```bash
body-models set <model> <path>   # Set model path
body-models unset <model>        # Remove from config
```

## Quick Start

```python
import torch
from body_models import SMPL, SMPLX, SKEL, ANNY, MHR

# Load any model
model = SMPLX(gender="neutral")  # Uses configured path

# Get default parameters
params = model.get_rest_pose(batch_size=4)

# Generate mesh vertices
vertices = model.forward_vertices(**params)  # [4, V, 3]

# Get skeleton transforms
skeleton = model.forward_skeleton(**params)  # [4, J, 4, 4]
```

## Common Properties

All models inherit from `BodyModel` and share these properties:

| Property | Type | Description |
|----------|------|-------------|
| `num_joints` | `int` | Number of skeleton joints |
| `num_vertices` | `int` | Number of mesh vertices |
| `faces` | `[F, 3]` | Mesh face indices |
| `skin_weights` | `[V, J]` | Skinning weights |
| `rest_vertices` | `[V, 3]` | Vertices in rest pose |

### Common Methods

```python
# Get zero-initialized parameters (model-specific keys)
params = model.get_rest_pose(batch_size=1)

# Compute mesh vertices [B, V, 3] in meters
vertices = model.forward_vertices(**params)

# Compute joint transforms [B, J, 4, 4] in meters
transforms = model.forward_skeleton(**params)
```

## Supported Models

### SMPL

The original parametric body model with 6890 vertices and 24 joints.

```python
from body_models import SMPL

model = SMPL(gender="neutral")  # "neutral", "male", or "female"

vertices = model.forward_vertices(
    shape,               # [B, 10] body shape betas
    body_pose,           # [B, 23, 3] axis-angle per joint
    pelvis_rotation,     # [B, 3] root joint rotation (optional)
    pelvis_translation,  # [B, 3] root translation (optional)
    global_rotation,     # [B, 3] post-transform rotation (optional)
    global_translation,  # [B, 3] post-transform translation (optional)
)
```

Conversion functions for working with the official smplx library format:

```python
from body_models import smpl

# Convert flat tensors to API format
args = smpl.from_native_args(shape, body_pose, pelvis_rotation, pelvis_translation)
vertices = model.forward_vertices(**args)
transforms = model.forward_skeleton(**args)

# Convert outputs back to native format (removes feet offset, extracts joint positions)
result = smpl.to_native_outputs(vertices, transforms, model._feet_offset)
```

### SMPL-X

Expressive body model with articulated hands and facial expressions.

```python
from body_models import SMPLX

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
    pelvis_translation,  # [B, 3] root translation (optional)
    global_rotation,     # [B, 3] post-transform rotation (optional)
    global_translation,  # [B, 3] post-transform translation (optional)
)
```

Conversion functions for working with the official smplx library format:

```python
from body_models import smplx

# Convert flat tensors to API format
args = smplx.from_native_args(shape, expression, body_pose, hand_pose, head_pose,
                              pelvis_rotation, pelvis_translation)
vertices = model.forward_vertices(**args)
transforms = model.forward_skeleton(**args)

# Convert outputs back to native format
result = smplx.to_native_outputs(vertices, transforms, model._feet_offset)
```

### SKEL

Anatomically realistic skeletal articulation based on OpenSim. Only "male" and "female" genders are supported (no "neutral").

```python
from body_models import SKEL

model = SKEL(gender="male")  # "male" or "female" (no neutral)

vertices = model.forward_vertices(
    shape,               # [B, 10] body shape betas
    pose,                # [B, 46] anatomically constrained DOFs
    global_rotation,     # [B, 3] axis-angle (optional)
    global_translation,  # [B, 3] (optional)
)
```

### ANNY

Phenotype-based body model with intuitive shape parameters.

```python
from body_models import ANNY

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
from body_models import MHR

model = MHR(lod=1)  # Level of detail

vertices = model.forward_vertices(
    shape,               # [B, 45] identity blendshapes
    pose,                # [B, 204] pose parameters
    expression,          # [B, 72] facial expression (optional)
    global_rotation,     # [B, 3] axis-angle (optional)
    global_translation,  # [B, 3] (optional)
)
```

Conversion functions for working with the original MHR format (cm units):

```python
from body_models import mhr

# Convert native args (shape, expression, pose order) to API format
args = mhr.from_native_args(shape, expression, pose)
vertices = model.forward_vertices(**args)
transforms = model.forward_skeleton(**args)

# Convert outputs to native format (cm units, skeleton state [t, q, s])
result = mhr.to_native_outputs(vertices, transforms)
```

## Coordinate System

The unified API returns outputs in:
- **Y-up** coordinate system
- **Meters** as the unit
- Feet at floor level (Y=0)

Use the `to_native_outputs()` conversion functions to get outputs in the original library conventions.

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
- ANNY: CC0 (MakeHuman data)
- MHR: Apache 2.0 (Meta Platforms, Inc.)
