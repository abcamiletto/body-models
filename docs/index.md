# body-models

Parametric body models for NumPy, PyTorch, and JAX, with optional Warp kernels.

## Install

Requires Python 3.11 or newer. NumPy support is included; extras add other
runtimes:

```bash
uv add body-models
uv add "body-models[torch]"
uv add "body-models[jax]"
uv add "body-models[torch,warp]"
```

## Quickstart

The import path selects the backend:

```python
from body_models.smpl.torch import SMPL

model = SMPL(gender="neutral")
params = model.get_rest_pose(batch_dims=(1,))
vertices = model.forward_vertices(**params)
skeleton = model.forward_skeleton(**params)
```

To select it at runtime instead, use
`create_model("smpl", runtime="torch", gender="neutral")`.

Torch models are `torch.nn.Module` instances, so `.to()`, `.cuda()`, and
`state_dict()` work as usual. Pass `kernel_backend="warp"` to run shared
operations on Warp while keeping Torch tensors.

## Models

| Model | Scope | Assets |
| --- | --- | --- |
| [SMPL](models/smpl.md) | Body | Registration |
| [SMPL-H](models/smplh.md) | Body and hands | Registration |
| [SMPL-X](models/smplx.md) | Body, hands, face | Registration |
| [ANNY](models/anny.md) | Phenotype-driven body | Auto-download |
| [MHR](models/mhr.md) | Body with facial expression | Auto-download |
| [SOMA](models/soma.md) | Body from SOMA-X assets | Auto-download |
| [GarmentMeasurements](models/garment-measurements.md) | PCA body for measurements | Auto-download |
| [SKEL](models/skel.md) | Body with anatomical skeleton | Registration |
| [FLAME](models/flame.md) | Head and face | Registration |
| [GNM Head](models/gnm.md) | Head, face, eyes, teeth, tongue | Auto-download |
| [MANO](models/mano.md) | Hand | Registration |

## Model assets

Public assets download to the user cache on first use. Licensed models need an
account on the upstream site: `body-models download <model>` asks for its
credentials, or reads them from `<ACCOUNT>_USERNAME` and `<ACCOUNT>_PASSWORD`
(for example `MANO_USERNAME`). Every download saves its path to the config.

```bash
body-models                                                 # show cache, config, and saved paths
body-models download anny                                   # prefetch into the cache
body-models download anny --output-dir /path/to/models/anny # download to a chosen directory
body-models download all --output-dir /path/to/models       # one subdirectory per model
body-models set smpl-neutral /path/to/SMPL_NEUTRAL.pkl      # use a file you already have
```

## Parameters and skeleton

- `parameter_spec` maps each parameter to its dimensions, role, and default;
  `get_rest_pose()` builds the defaults.
- Arrays accept any leading batch dimensions: `*batch J 4 4` covers unbatched,
  single, and multi-batch inputs.
- `joint_names` and `parents` describe the native skeleton in index order.
  `joint_index(Joint.LEFT_WRIST)` maps the shared `Joint` enum to a native index.
- `has_hands` and `has_face` report articulated hand and facial-expression
  controls, not mesh geometry.

Class constants give the fixed dimensions:

| Constants | Meaning |
| --- | --- |
| `NUM_JOINTS` | Skeleton size returned by `forward_skeleton()`. |
| `NUM_BODY_CONTROLS`, `NUM_HAND_CONTROLS`, `NUM_HEAD_CONTROLS` | Length of each pose argument's control axis. |
| `NUM_SHAPE_COEFFS`, `NUM_EXPR_COEFFS` | Identity and expression dimensions. |
| `NUM_POSE_COEFFS`, `NUM_*_POSE_COEFFS` | Compact pose dimensions. |

Controls and joints need not match: SMPL has 24 joints but 23 body controls,
plus a separate root rotation. Dimensions that depend on constructor arguments
are instance properties, such as SOMA's `num_shape_coeffs`.

The [API reference](api.md) documents the shared interface;
[architecture](architecture.md) explains how the code is organized.
