# body-models

`body-models` provides a shared interface for parametric and rigid articulated
models with NumPy, PyTorch, and JAX runtimes plus optional GPU acceleration.

## Install

```bash
uv add body-models
```

Install optional framework runtimes when needed:

```bash
uv add "body-models[torch]"
uv add "body-models[triton]"  # Linux CUDA
uv add "body-models[jax]"
uv add "body-models[torch,warp]"
```

## Model assets

Public assets download automatically on first use when `model_path` is omitted.
They live in the operating system's private user cache, not beside the
configuration file or inside the Python environment. Run `body-models` to see
both locations.

Use the CLI to prefetch assets or choose an exact destination:

```bash
body-models download anny
body-models download anny --output-dir /path/to/models/anny
```

The custom path is saved as the model's configured override. With
`body-models download all --output-dir /path/to/models`, each family gets its
own subdirectory. Licensed models cannot download silently on first use because
they require accepted licenses and account credentials; their setup command
prompts for those credentials and stores the resulting private-cache path.

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

Each model exposes a class from its `numpy`, `torch`, and `jax` modules. Select
the array backend in the import path. NumPy does not require an optional
framework dependency.

Names exported from `body_models`, model packages, and backend modules are the
stable public API.
Underscore-prefixed modules are private implementation details and are not
covered by compatibility guarantees. See the [API reference](api.md) for the
shared contracts and the [architecture guide](architecture.md) for the runtime
boundary and extension rules.

```python
from body_models.smpl.torch import SMPL

model = SMPL(gender="neutral")
params = model.get_rest_pose(batch_dims=(1,))
vertices = model.forward_vertices(**params)
skeleton = model.forward_skeleton(**params)
```

Models imported from a `torch` module are `torch.nn.Module` instances, so
`.to()`, `.cuda()`, and `state_dict()` work directly. Torch models accept
`kernel_backend="triton"` for compiled CUDA float32 skinning and
`kernel_backend="warp"` for the optional Warp implementation. The array runtime
remains Torch; kernel backends only replace shared operations they implement.
Triton currently skins the full mesh before applying `vertex_indices`, keeping
its reusable backward plan outside compiled calls.

All models derive from `ArticulatedModel`; `SkinnedModel` and `RigidBodyModel`
define its two public specializations. The shared contract includes `runtime`,
`has_face`, `has_hands`, `parameter_spec`, `get_rest_pose`, `faces`,
`num_vertices`, `num_joints`, `joint_names`, `parents`, `common_joints`,
`joint_index`, and `forward_skeleton`.
`has_face` indicates facial-expression controls; `has_hands` indicates
articulated hand controls. Neither describes mesh geometry. Skinned models
additionally share `skin_weights`, `skinning_spec`, `rest_vertices`,
`apply_pose_correctives`, and `forward_vertices`. `skin_weights` follows the
public skeleton; `skinning_spec.skinning_weights` follows the complete render
rig and its prepared skinning transforms. Rigid articulated models expose link
metadata, cached link-local `link_meshes`, `forward_links`, and `forward_meshes`
instead of skinning weights.

`joint_names` and `parents` describe the complete native skeleton in joint
index order. The `Joint` enum names anatomical joints shared across models;
`common_joints` maps those names to the native skeleton, and
`joint_index(Joint.LEFT_WRIST)` resolves the corresponding native index.

Fixed public parameter dimensions use `NUM_*` class constants:
`NUM_JOINTS`, `NUM_BODY_JOINTS`, `NUM_HAND_JOINTS`, `NUM_HEAD_JOINTS`,
`NUM_SHAPE_COEFFS`, `NUM_EXPR_COEFFS`, and, for compact pose controls,
`NUM_POSE_COEFFS` and `NUM_*_POSE_COEFFS`. A class defines only the constants
that apply to that model. A dimension fixed by the supported checkpoint schema
is a class constant even when the checkpoint is loaded from a custom path.
Dimensions selected by a constructor option remain instance properties; for
example, SOMA exposes `num_shape_coeffs` because it depends on `model_type`.

Array shapes use arbitrary leading batch dimensions throughout. For example,
an annotated `*batch J 4 4` skeleton can be unbatched, singly batched, or have
several leading batch axes.

Skinned model packages export model-specific identity types when their schemas
are unique. Shared contracts are available as `LinearIdentity`,
`SkinningIdentity`, `SkinningPose`, and `SkinningSpec` from `body_models`.
