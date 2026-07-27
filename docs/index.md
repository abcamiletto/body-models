# body-models

`body-models` provides a shared interface for parametric and rigid articulated
models across PyTorch, NumPy, JAX, and optional Warp skinning kernels.

## Install

```bash
# Install the core package with the NumPy backend.
uv add body-models
```

Install optional differentiable backends when needed:

```bash
# Add PyTorch or JAX support only when your project needs it.
uv add "body-models[torch]"
uv add "body-models[jax]"
```

## Supported Models

### Full Bodies

| Model | Scope | Setup |
| --- | --- | --- |
| [SMPL](models/smpl.md) | body | registration required |
| [SMPL-H](models/smplh.md) | body and hands | registration required |
| [SMPL-X](models/smplx.md) | body, hands, face | registration required |
| [ANNY](models/anny.md) | phenotype-driven body | auto-download |
| [MHR](models/mhr.md) | expressive full body | auto-download |
| [SOMA](models/soma.md) | skinned body from SOMA-X assets | auto-download |
| [GarmentMeasurements](models/garment-measurements.md) | PCA body for garment measurements | auto-download |

### Anatomicals

| Model | Scope | Setup |
| --- | --- | --- |
| [SKEL](models/skel.md) | body with anatomical skeleton | registration required |
| [MyoFullBody](models/myofullbody.md) | MuJoCo-derived musculoskeletal full body | auto-download |

### Heads

| Model | Scope | Setup |
| --- | --- | --- |
| [FLAME](models/flame.md) | head and face | registration required |

### Hands

| Model | Scope | Setup |
| --- | --- | --- |
| [MANO](models/mano.md) | hand | registration required |

### Robots

| Model | Scope | Setup |
| --- | --- | --- |
| [BrainCo](models/brainco.md) | BrainCo Revo 2 robotic hand | auto-download |
| [G1](models/g1.md) | Unitree G1 rigid links | auto-download |
| [SmplHumanoid](models/smpl-humanoid.md) | SMPL-compatible humanoid MJCF variants | auto-download |

## Common Usage

Each model package exports one class. Select NumPy, Torch, or JAX with the
`runtime` argument; the class identity and model API stay the same. NumPy is the
default and does not require an optional framework dependency.

Names exported from `body_models` and model packages are the stable public API.
Underscore-prefixed modules are private implementation details and are not
covered by compatibility guarantees. See the
[architecture guide](architecture.md) for the runtime boundary and extension
rules.

```python
from body_models.smpl import SMPL

# Load the neutral SMPL model from the configured model path.
model = SMPL(gender="neutral", runtime="torch")

# Start from a batched rest pose.
params = model.get_rest_pose(batch_dims=(1,))

# Evaluate the mesh vertices and skeleton transforms with the same parameters.
vertices = model.forward_vertices(**params)
skeleton = model.forward_skeleton(**params)
```

Call `model.as_module()` when PyTorch module lifecycle behavior such as
`.to()`, `.cuda()`, or `state_dict()` is needed. Pass a configured runtime
object for runtime-specific behavior such as Warp skinning.

Skinned models share `faces`, `num_vertices`, `num_joints`, `joint_names`, `skin_weights`, `rest_vertices`, `forward_vertices`, `forward_skeleton`, and `get_rest_pose`. Rigid articulated models expose link metadata and `forward_links` instead of skinning weights.
