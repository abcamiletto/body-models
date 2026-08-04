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

`RuntimeName` is the runtime-name literal (`"numpy"`, `"torch"`, or `"jax"`), and
`RuntimeLike` accepts either one of those names or an `ArrayRuntime` instance.

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
