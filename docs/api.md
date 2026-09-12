# API reference

Public names are exported from `body_models`, model packages, and backend
modules. See [model pages](index.md#supported-models) for model-specific APIs.

## Model creation

::: body_models.create_model
    options:
      show_source: false

::: body_models.list_models
    options:
      show_source: false

## Model contracts

Every model provides `get_rest_pose()`. Models with whole-body presets also
expose `get_tpose()` and `get_apose()`.

::: body_models.SkinnedModel
    options:
      show_source: false

## Mapped points

`forward_points()` evaluates a dense `[points, vertices]` mapping without
producing the posed mesh. Its vertex dimension must match the model and mesh
simplification. Prepare the regressor after moving a Torch model to its final
device:

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

Mapped points contain positions; `forward_skeleton()` returns native joint
transforms. Prepared regressors do not follow later `.to()` calls.

::: body_models.PointRegressor
    options:
      show_source: false

## Parameter and joint metadata

`ParameterRole` is the literal `"identity"`, `"pose"`, or `"transform"`.
`RotationType` accepts `"axis_angle"`, `"quat"`, `"sixd"`, `"matrix"`, or
`"rotmat"`. `matrix` is an arbitrary 3×3 transform; `rotmat` is a proper
SO(3) rotation.

`pose_joint_indices` maps pose parameters to the distinct canonical joints whose
local transforms they drive. Use these tuples to select skeleton outputs:

```python
hand_indices = model.pose_joint_indices["hand_pose"]
hand_skeleton = model.forward_skeleton(**params, joint_indices=hand_indices)
```

Indices refer to the full skeleton. Groups may overlap and omit fixed joints.
Changing a local transform also moves descendants outside its group. Rotational
controls map one-to-one to indices in control order: `[..., i, :]` for vectors,
`[..., i, :, :]` for matrices.

`symmetric_joints` lists `(left_index, right_index)` pairs for symmetry losses
and left/right swaps:

```python
order = list(range(model.num_joints))
for left, right in model.symmetric_joints:
    order[left], order[right] = right, left
swapped = model.forward_skeleton(**params)[..., order, :, :]
```

Pairs cover the native skeleton, including joints outside `Joint`, such as
SMPL's collars. Unpaired joints lie on the midline. Swapping indices does not
mirror a pose; callers must also reflect rotations in the model's coordinate
frame and parameterization.

::: body_models.ParameterSpec
    options:
      show_source: false

::: body_models.Joint
    options:
      show_source: false

## Runtimes

`RuntimeName` accepts `"numpy"`, `"torch"`, or `"jax"`. `KernelBackend` selects
`"torch"` or `"warp"` kernels for Torch models.

::: body_models.ArrayRuntime
    options:
      show_source: false

## Prepared skinning

Identity and pose records are `TypedDict`s. `SkinningSpec` is a dataclass.

| Contract | Contents |
| --- | --- |
| `SkinningIdentity` | Identity-dependent `rest_vertices`. |
| `LinearIdentity` | Rest vertices, rest joints, and local joint offsets. |
| `SkinningPose` | `skeleton_transforms`, `skinning_transforms`, optional compact `pose_coefficients`. |
| `SkinningSpec` | Triangles, weights aligned with skinning transforms, optional dense/sparse corrective basis. |

Arrays retain arbitrary leading batch dimensions. Corrective bases implement
`pose_offsets = basis.apply(pose_coefficients)`. Coefficient meanings are
model-specific; use `model.apply_pose_correctives(identity=identity, pose=pose)`
to expand them without depending on the representation. Model packages export
`*Identity` types only for additional fields; all models share `SkinningPose`.

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
