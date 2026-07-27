# body-models

`body-models` provides a shared interface for parametric and rigid articulated
models with NumPy, PyTorch, and JAX runtimes plus optional Warp acceleration.

## Install

```bash
uv add body-models
```

Install optional framework runtimes when needed:

```bash
uv add "body-models[torch]"
uv add "body-models[jax]"
uv add "body-models[torch,warp]"
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

### Robots and Humanoids

| Model | Scope | Setup |
| --- | --- | --- |
| [BrainCo](models/brainco.md) | BrainCo Revo 2 robotic hand | auto-download |
| [G1](models/g1.md) | Unitree G1 rigid links | auto-download |
| [SmplHumanoid](models/smpl-humanoid.md) | SMPL-compatible humanoid MJCF variants | auto-download |

## Common Usage

Each model has one public class shared by its NumPy, Torch, and JAX runtimes.
Select the runtime with the `runtime` argument. NumPy is the default and does
not require an optional framework dependency.

Names exported from `body_models` and model packages are the stable public API.
Underscore-prefixed modules are private implementation details and are not
covered by compatibility guarantees. See the
[architecture guide](architecture.md) for the runtime boundary and extension
rules.

```python
from body_models.smpl import SMPL

model = SMPL(gender="neutral", runtime="torch")
params = model.get_rest_pose(batch_dims=(1,))
vertices = model.forward_vertices(**params)
skeleton = model.forward_skeleton(**params)
```

Call `model.as_module()` when PyTorch module lifecycle behavior such as
`.to()`, `.cuda()`, or `state_dict()` is needed. Each model returns the same
cached module view on every call, and lifecycle mutations affect the model's
shared numeric state. Pass a configured runtime object for runtime-specific
behavior such as Warp skinning.

All models expose `parameter_spec`, `get_rest_pose`, `faces`, `num_vertices`,
`num_joints`, `joint_names`, and `forward_skeleton`. Skinned models additionally
share `skin_weights`, `rest_vertices`, and `forward_vertices`. Rigid articulated
models expose link metadata, `forward_links`, and `forward_meshes` instead of
skinning weights.
