# API Reference

Only names exported from `body_models` and its model packages are public.
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

`Backend` is the runtime-name literal (`"numpy"`, `"torch"`, or `"jax"`), and
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
optional `pose_offsets`. Shapes retain arbitrary leading batch dimensions.
Model packages export `*Identity` and `*Pose` types only when they add fields
to these shared contracts.

::: body_models.LinearIdentity
    options:
      show_source: false

::: body_models.SkinningPayload
    options:
      show_source: false
