# API Reference

Only names exported from `body_models`, its model packages, and backend modules
are public.
Model-specific classes and helpers are documented on their
[model pages](index.md#supported-models).

## Model creation

::: body_models.create_model
    options:
      show_source: false

::: body_models.list_models
    options:
      show_source: false

## Model contracts

Every model provides `get_rest_pose()`. Models expose `get_tpose()` and
`get_apose()` only when those whole-body presets are meaningful.

::: body_models.ArticulatedModel
    options:
      show_source: false

::: body_models.SkinnedModel
    options:
      show_source: false

::: body_models.RigidBodyModel
    options:
      show_source: false

## Mapped points

Every skinned model can evaluate positions defined by an arbitrary dense
`[points, vertices]` mapping without producing its posed mesh. The mapping is
topology-specific: its vertex dimension must match the selected model and mesh
simplification.

Prepare the regressor once after placing a Torch model on its final device,
then pass it to the model's explicit `forward_points()` method:

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

The result contains positions only. `forward_skeleton()` continues to return
the model's native rigid transforms. A prepared regressor does not follow later
`.to()` calls.

::: body_models.PointRegressor
    options:
      show_source: false

## Parameter and joint metadata

`ParameterRole` is the literal `"identity"`, `"pose"`, or `"transform"`.
`RotationType` accepts `"axis_angle"`, `"quat"`, `"sixd"`, `"matrix"`, or
`"rotmat"`. `matrix` is an arbitrary 3×3 transform; `rotmat` is a proper
SO(3) rotation.

::: body_models.ParameterSpec
    options:
      show_source: false

::: body_models.Joint
    options:
      show_source: false

## Runtimes

`RuntimeName` is the runtime-name literal (`"numpy"`, `"torch"`, or `"jax"`).
`KernelBackend` selects the Torch operation lowering (`"torch"` or `"warp"`).

::: body_models.ArrayRuntime
    options:
      show_source: false

::: body_models.NumpyRuntime
    options:
      show_source: false

::: body_models.TorchRuntime
    options:
      show_source: false

::: body_models.JaxRuntime
    options:
      show_source: false

## Prepared skinning

`SkinningIdentity` is a `TypedDict` containing identity-dependent
`rest_vertices`. `LinearIdentity` adds rest joints and local joint offsets.
`SkinningPose` contains `skeleton_transforms`, `skinning_transforms`, and
optional compact `pose_coefficients`. `SkinningSpec` contains model-static
triangles, weights aligned with `skinning_transforms`, and an optional dense or
sparse corrective basis. Shapes retain arbitrary leading batch dimensions.

The corrective contract is
`pose_offsets = corrective_basis.apply(pose_coefficients)`. Coefficient
semantics are model-local and must not be interpreted by consumers. Use
`model.apply_pose_correctives(identity=identity, pose=pose)` to expand them
without depending on the basis representation.
Model packages export `*Identity` types only when they add fields to the shared
identity contracts; all skinned models use the shared `SkinningPose`.

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
