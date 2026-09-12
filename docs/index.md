# body-models

Parametric body models for NumPy, PyTorch, and JAX, with optional Warp kernels.

## Install

Requires Python 3.11 or newer. NumPy support is included; add extras for other
runtimes:

```bash
uv add body-models
uv add "body-models[torch]"
uv add "body-models[jax]"
uv add "body-models[torch,warp]"
```

## Model assets

Public assets download on first use when no model path is configured or passed.
Assets use the operating system's user cache. Run `body-models` to see the cache
and configuration paths.

To prefetch assets or save a custom destination:

```bash
body-models download anny
body-models download anny --output-dir /path/to/models/anny
```

`download all --output-dir /path/to/models` creates a subdirectory per family.
Licensed models require registration and accepted licenses; their download
commands prompt for credentials and save the asset path.

## Supported models

| Model | Scope | Setup |
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

## Common usage

Select the backend through the import path:

```python
from body_models.smpl.torch import SMPL

model = SMPL(gender="neutral")
params = model.get_rest_pose(batch_dims=(1,))
vertices = model.forward_vertices(**params)
skeleton = model.forward_skeleton(**params)
```

Torch models are `torch.nn.Module` instances supporting `.to()`, `.cuda()`, and
`state_dict()`. `kernel_backend="warp"` selects Warp implementations of shared
operations while keeping Torch tensors.

All models derive from `SkinnedModel`. Its [API reference](api.md) covers
parameter defaults, geometry, joints, prepared skinning, and mapped points.
Names exported from public packages are stable; underscore-prefixed modules
are private. See [architecture](architecture.md) for implementation boundaries.

### Parameters and joints

`parameter_spec` describes each parameter's dimensions, role, and default.
`has_face` indicates facial-expression controls and `has_hands` articulated
hand controls; these flags do not describe mesh geometry.

`joint_names` and `parents` describe the complete native skeleton in index
order. `common_joints` maps the shared `Joint` enum to native names;
`joint_index(Joint.LEFT_WRIST)` resolves a native index. `skin_weights` follows
this public skeleton, while `skinning_spec.skinning_weights` follows the render
rig and its prepared transforms.

Fixed dimensions use `NUM_*` class constants where applicable:

| Constants | Meaning |
| --- | --- |
| `NUM_JOINTS` | Skeleton size returned by `forward_skeleton()`. |
| `NUM_BODY_CONTROLS`, `NUM_HAND_CONTROLS`, `NUM_HEAD_CONTROLS` | Entries along each pose argument's control axis. |
| `NUM_SHAPE_COEFFS`, `NUM_EXPR_COEFFS` | Identity and expression dimensions. |
| `NUM_POSE_COEFFS`, `NUM_*_POSE_COEFFS` | Compact pose dimensions. |

Control counts can differ from joint counts: SMPL has 24 joints and 23 body
controls, with a separate root rotation. Dimensions fixed by the asset schema
remain class constants for custom paths. Constructor-dependent dimensions use
instance properties, such as SOMA's `num_shape_coeffs`.

Arrays accept arbitrary leading batch dimensions: `*batch J 4 4` includes
unbatched, single-batch, and multi-batch skeletons. Shared preparation types
include `LinearIdentity`, `SkinningIdentity`, `SkinningPose`, and `SkinningSpec`;
model packages export identity types when they need additional fields.
