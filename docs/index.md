# body-models

`body-models` provides a shared interface for parametric human body, head, hand, and measurement models across PyTorch, NumPy, and JAX.

## Install

```bash
uv add body-models
```

Install optional differentiable backends when needed:

```bash
uv add "body-models[torch]"
uv add "body-models[jax]"
```

## Supported Models

| Model | Scope | Setup |
| --- | --- | --- |
| [SMPL](models/smpl.md) | body | registration required |
| [SMPL-X](models/smplx.md) | body, hands, face | registration required |
| [FLAME](models/flame.md) | head and face | registration required |
| [SKEL](models/skel.md) | body with anatomical skeleton | registration required |
| [ANNY](models/anny.md) | phenotype-driven body | auto-download |
| [MHR](models/mhr.md) | expressive full body | auto-download |
| [SOMA](models/soma.md) | skinned body from SOMA-X assets | auto-download |
| [GarmentMeasurements](models/garment-measurements.md) | PCA body for garment measurements | auto-download |
| [G1](models/g1.md) | Unitree G1 rigid links | auto-download |

## Common Usage

Each model exposes backend modules under `body_models.<model>.torch`, `body_models.<model>.numpy`, and `body_models.<model>.jax` when that backend is supported. The model pages use the NumPy backend for API generation because it has the same public model interface without optional backend dependencies.

```python
from body_models.smpl.torch import SMPL

model = SMPL(gender="neutral")
params = model.get_rest_pose(batch_size=1)
vertices = model.forward_vertices(**params)
skeleton = model.forward_skeleton(**params)
```

Rigged models share `faces`, `num_vertices`, `num_joints`, `joint_names`, `skin_weights`, `rest_vertices`, `forward_vertices`, `forward_skeleton`, and `get_rest_pose`.
