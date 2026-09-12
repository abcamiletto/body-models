# Architecture

Each model has one implementation, bound to NumPy, Torch, or JAX through a
shared runtime layer.

## Public API boundary

Names exported from `body_models`, model packages, and backend modules are
public. Underscore-prefixed modules are private and may change without a major
release.

## Model programs

Models live in `body_models/<name>/` and derive from `SkinnedModel`:

| File | Responsibility |
| --- | --- |
| `_io.py` | Resolve assets and load immutable NumPy data. |
| `_core.py` | Model mathematics with an explicit array namespace or runtime. |
| `_model.py` | Validation, state preparation, and forward methods. |
| `numpy.py`, `torch.py`, `jax.py` | Bind the model to an array backend. |
| `__init__.py` | Export shared model-specific types and helpers. |

Identity preparation computes rest vertices and joints. Pose preparation
computes transforms and compact corrective coefficients. `SkinningSpec` holds
triangles, render-rig weights, and an optional corrective basis.
`apply_pose_correctives()` expands coefficients; `forward_vertices()` skins the
result. `PointRegressor` projects this computation through vertex mappings,
while each model's `forward_points()` retains its explicit parameter signature.
Skeleton forwards have separate preparation paths and private state types.

Required numerical inputs may be positional. Optional arguments are
keyword-only, ordered as local pose options, identity, global transform, and
output selection.

SMPL, SMPL-H, SMPL-X, MANO, FLAME, and GNM share a private linear blendshape
engine. Model-local pose blocks define joint order and means; the engine handles
rotation conversion, root insertion, batch validation, forward kinematics,
bind-relative transforms, and corrective coefficients. It accepts arrays and
pose blocks, with no model names or feature flags.

These models also share linear identity preparation. Shape-only and
shape-plus-expression functions have separate signatures. Shape and expression
bases are evaluated separately to avoid copying concatenated bases on each call.

`parameter_spec` maps names to `ParameterSpec`, ordered by identity, pose, then
transform. Each entry records unbatched dimensions, role, numeric default, and
rotation representation. Dimensions reflect assets and configuration.
`get_rest_pose()` builds defaults from this mapping, including identity
rotations; model overrides apply named presets such as relaxed hands.

## Runtime boundary

`ArrayRuntime` owns the array namespace, device/dtype-aware construction, state
materialization, and shared operation implementations. `_state.py` converts
loader data; nested models remain models. Runtime-specific weights stay private,
with model properties exposing meshes, skeletons, and deformation bases.

Backend imports select the array runtime; `create_model()` selects it by name.
Torch models can select Warp kernels while retaining the Torch tensor API:

```python
from body_models.smpl.torch import SMPL

model = SMPL(gender="neutral", kernel_backend="warp")
```

Core functions that dispatch shared operations receive the runtime. Pure
numerical helpers receive its array namespace. Model code constructs local
transforms; the runtime composes the kinematic tree and executes skinning.

Reusable backend data is prepared during materialization. Torch/Warp compact
weights own a transform-gradient plan; per-call vertex subsets get temporary
plans. Sparse corrective bases likewise own their prepared state. A selected
kernel must execute or raise for unsupported inputs, never silently fall back.

Torch models inherit `torch.nn.Module`. Source arrays are persistent buffers,
so checkpoints are complete but may be large. Derived plans move with the model
and are rebuilt rather than serialized. Mapping values live in indexed child
modules to avoid collisions with Torch attributes. This changes paths in
exported `state_dict` snapshots containing mappings, such as SOMA's
`_assets.lods`. Hugging Face autodownload archives contain source assets and
upstream weights, so they need no regeneration.

JAX models implement the pytree protocol. Dataclass metadata is static; arrays
remain children even inside mixed containers. Reconstruction preserves model
and runtime configuration.

## Shared operations

| Module | Responsibility |
| --- | --- |
| `_common.skinning` | Dense/compact skinning, bind-relative transforms, global point and skeleton transforms. |
| `_common.deformation` | Linear blend shapes and dense/sparse corrective bases. |
| `_common.kinematics` | Affine transforms, rigid inversion, parent-relative offsets, forward kinematics. |

Shared operations know no model names, pose layouts, or asset formats. SOMA and
MHR compute corrective coefficients locally, then use the shared basis contract
to turn them into offsets.

## Adding a model

1. Load and validate assets in `_io.py`.
2. Implement mathematics in `_core.py` with an explicit namespace or runtime.
3. Define the `SkinnedModel` subclass in `_model.py`.
4. Bind and export NumPy, Torch, and JAX classes.
5. Add factory and asset metadata to `_catalog.py`.
6. Check cross-runtime results, batching, compilation, gradients, and reference
   outputs for supported operations.

Share code only when its meaning, inputs, outputs, batching, and gradients agree
across callers. Otherwise, keep it model-local.
