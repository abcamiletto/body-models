# API reference

This page covers the interface every model shares. Model-specific options are on
each [model page](index.md#models).

## Model creation

::: body_models.create_model
    options:
      show_source: false

::: body_models.list_models
    options:
      show_source: false

## Models

Every model provides `get_rest_pose()`; whole-body models also provide
`get_tpose()` and `get_apose()`.

::: body_models.SkinnedModel
    options:
      show_source: false

## Mapped points

`forward_points()` evaluates a dense `[points, vertices]` mapping, such as a
marker or joint regressor, without building the posed mesh. The vertex count
must match the model, including any mesh simplification.

```python
import numpy as np
import torch

from body_models.smplx.torch import SMPLX

model = SMPLX(gender="neutral").cuda()
mapping = np.load("captury_J_regressor.npz")["J_regressor"]
regressor = model.prepare_point_regressor(mapping)
params = model.get_rest_pose(batch_dims=(2048,))

with torch.inference_mode():
    points = model.forward_points(**params, point_regressor=regressor)
# points.shape == (2048, 67, 3)
```

A prepared regressor stays on the device it was prepared on, so prepare it after
moving the model. Mapped points are positions; `forward_skeleton()` returns
joint transforms.

::: body_models.PointRegressor
    options:
      show_source: false

## Parameters and joints

`ParameterRole` is `"identity"`, `"pose"`, or `"transform"`. `RotationType` is
`"axis_angle"`, `"quat"`, `"sixd"`, `"matrix"` (any 3×3 transform), or
`"rotmat"` (a proper SO(3) rotation).

`pose_joint_indices` maps each pose parameter to the joints whose local
transforms it drives. Rotational controls follow the same order, so control `i`
(`[..., i, :]`, or `[..., i, :, :]` for matrices) drives joint `indices[i]`:

```python
hand_indices = model.pose_joint_indices["hand_pose"]
hand_skeleton = model.forward_skeleton(**params, joint_indices=hand_indices)
```

Indices refer to the full skeleton. Groups may overlap and omit fixed joints,
and a control also moves its joint's descendants outside the group.

`symmetric_joints` lists `(left, right)` index pairs over the whole native
skeleton, including joints outside `Joint`; unpaired joints lie on the midline.
Use it for symmetry losses or left/right swaps:

```python
order = list(range(model.num_joints))
for left, right in model.symmetric_joints:
    order[left], order[right] = right, left
swapped = model.forward_skeleton(**params)[..., order, :, :]
```

Swapping indices does not mirror a pose: the rotations must also be reflected in
the model's frame and parameterization.

::: body_models.ParameterSpec
    options:
      show_source: false

::: body_models.Joint
    options:
      show_source: false

## Runtimes

`RuntimeName` is `"numpy"`, `"torch"`, or `"jax"`. For Torch models,
`KernelBackend` selects `"torch"` or `"warp"` kernels.

::: body_models.ArrayRuntime
    options:
      show_source: false

## Prepared skinning

Each model's `prepare_identity()` and `prepare_pose()` return the intermediate
records behind `forward_vertices()`. Pass a prepared `identity` to reuse it
across poses. `skinning_spec` holds the static data used for skinning.

| Type | Contents |
| --- | --- |
| `SkinningIdentity` | Identity-dependent `rest_vertices`. |
| `LinearIdentity` | Rest vertices, rest joints, and local joint offsets. |
| `SkinningPose` | Skeleton and skinning transforms, and optional compact `pose_coefficients`. |
| `SkinningSpec` | Triangles, skinning weights, and an optional dense or sparse corrective basis. |

Identity and pose records are `TypedDict`s; `SkinningSpec` is a dataclass. All
keep the leading batch dimensions. Model packages export their own `*Identity`
type only when it adds fields; every model uses `SkinningPose`.

Pose coefficients are model-specific. Expand them with
`model.apply_pose_correctives(identity=identity, pose=pose)` rather than
depending on the basis representation.

`SkinningSpec.skinning_weights` follow the render rig, which can differ from
the public skeleton that `skin_weights` follows.

## Motion dictionaries

Each model has a `TypedDict` for motion parameters. Import it from
`body_models` and unpack it into a forward method.

```python
from body_models import SmplMotion

motion: SmplMotion = {
    "body_pose": body_pose,
    "global_translation": translation,
}
vertices = model.forward_vertices(**motion, shape=shape)
```

The available types are `AnnyMotion`, `FlameMotion`,
`GarmentMeasurementsMotion`, `GnmMotion`, `ManoMotion`, `MhrMotion`,
`SkelMotion`, `SmplMotion`, `SmplhMotion`, `SmplxMotion`, and `SomaMotion`.

::: body_models.LinearIdentity
    options:
      show_source: false

::: body_models.SkinningSpec
    options:
      show_source: false

::: body_models.DenseCorrectiveBasis
    options:
      show_source: false

::: body_models.SparseCorrectiveBasis
    options:
      show_source: false
