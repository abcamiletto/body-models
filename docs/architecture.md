# Architecture

`body-models` has one implementation of each model and a small execution layer
for array ownership and genuinely shared operations.

## Public API boundary

The stable public API is intentionally small:

- names exported from `body_models`;
- names explicitly exported from a model or backend package.

All underscore-prefixed modules are private implementation details. This
includes model programs and loaders such as `smpl._model` and `smpl._io`, and
shared infrastructure such as `_runtime`, `_state`, and `_common`. They may
change without a major release.

## Model programs

Each model family follows the same file roles:

| File | Responsibility |
| --- | --- |
| `_io.py` | Resolve assets and load immutable NumPy model data. |
| `_core.py` | Model-specific mathematics with an explicit array namespace. |
| `_model.py` | Define the model class, validation, state preparation, and forward orchestration. |
| `numpy.py`, `torch.py`, `jax.py` | Bind the shared model class to one public array backend. |
| `__init__.py` | Export shared model-specific types and helpers. |

Every model derives from the public `ArticulatedModel` base through either
`SkinnedModel` or `RigidBodyModel`. Models are self-contained in
`body_models/<name>/`; descriptive categories do not create a second package
tree. Thin public subclasses bind that implementation to each runtime. On skinned models,
identity preparation returns identity-dependent vertices and joints, while
pose preparation returns transforms and compact corrective coefficients.
`SkinningSpec` holds model-static triangles, render-rig skinning weights, and
the optional corrective basis. `apply_pose_correctives()` expands compact
coefficients without exposing the dense or sparse representation;
`forward_vertices()` then skins the surface. `PointRegressor` projects the same
contract through arbitrary vertex mappings, while explicit model-local
`forward_points()` methods retain each model's parameter signature. Skeleton
forwards use distinct model-local preparation paths.
Shared preparation and skinning contracts are exported from `body_models`;
skeleton-only preparation types remain private.
Required numerical inputs may be positional, while optional configuration,
state, transforms, and output selection are keyword-only. Forward signatures
order those groups as local pose options, identity, global transform, and
selection.

SMPL, SMPL-H, SMPL-X, MANO, and FLAME share one private family engine. Their
`_core.py` modules describe the ordered pose blocks and apply model-specific
means, while the engine owns rotation conversion, root insertion, batch
validation, forward kinematics, bind-relative transforms, and corrective
coefficient construction.
The public methods remain explicit per model. The engine accepts arrays and
pose blocks only; it has no model names, optional-feature flags, or knowledge of
hands and faces.

Linear identity preparation is shared within the family because each model
applies coefficients to vertex and joint bases in the same way. Shape-only and
shape-plus-expression paths remain separate so their signatures state their
requirements without mode flags.

Each instance exposes `parameter_spec`, an ordered mapping from public parameter
names to `ParameterSpec`. A specification records the unbatched array shape,
semantic role (`identity`, `pose`, or `transform`), numeric default, and rotation
representation where applicable. A rotation representation determines the
corresponding identity rotation. Parameters are ordered by role: identity, then
pose, then transform. Dimensions derived from assets or configuration are
therefore represented accurately. The shared base constructs `get_rest_pose()`
from this mapping; model-local overrides only apply named presets such as flat or
relaxed hands.

## Runtime boundary

`ArrayRuntime` owns the array namespace, device- and dtype-aware construction,
state materialization, and lowerings of stable shared operations. These include
compact linear blend skinning and skinned pose-tree composition.
Materialization delegates to the recursive converters in `_state.py`, which
accept loader data rather than model objects; models composed inside another
model remain models. Callers therefore cannot pair a runtime with the wrong
framework state. Materialized weights are private because their container
types are runtime-specific; stable model properties provide public access to
meshes, skeletons, and deformation bases. The runtime does not own model
semantics.

The public backend modules bind each model to one array runtime. Torch models
can additionally select a kernel backend without changing their tensor API:

```python
from body_models.smpl.torch import SMPL

model = SMPL(kernel_backend="warp")
```

The shared model implementation still receives an internal `ArrayRuntime`.
`create_model()` accepts a runtime name for callers that select a model and
backend dynamically.

Kernel dispatch follows the lifetime of the work. Operation execution is
lowered by the runtime; reusable derived inputs are created during state
materialization. Skinned pose programs construct local affine transforms but
delegate their parent-tree composition to the runtime, independent of pose
layout or rotation representation. Core entry points that execute a lowered
operation receive the runtime; pure numerical helpers receive only its array
namespace. Compact skinning follows the same boundary: every runtime executes
the same call contract, while Torch/Warp materialization augments the compact
weights with a transform-gradient plan. A vertex subset chosen during a call
gets a short-lived subset plan instead. Sparse corrective bases similarly own
their prepared representation as materialized state. Kernel backends are
additive operation lowerings; a selected lowering must execute or raise for an
unsupported input, never silently switch implementations. This keeps
`ArrayRuntime` independent of model semantics without hiding persistent work in
global caches.

Torch backend models inherit `torch.nn.Module`, and their materialized state is
registered directly as modules and persistent buffers. Source numeric model
state is persistent, so checkpoints are complete but may be large. Derived
backend plans move with their owning module but are rebuilt rather than
serialized. JAX backend models implement the pytree protocol. Pytree
reconstruction preserves both model configuration and runtime configuration.

The shared skinning module contains only operations whose signatures are stable
across model families: compact and dense linear blend skinning, bind-relative
transforms, global point transforms, and skeleton transforms. Model-specific
pose layouts remain beside their model; the family engine composes those layouts
with the generic kinematics and skinning operations.

The same rule applies below the runtime boundary. `_common.deformation` owns
linear blend shapes and dense or sparse corrective bases; `_common.kinematics`
owns affine transform assembly, rigid inversion, parent-relative offsets, and
generic forward kinematics. These functions operate on explicit arrays and do
not know model names, parameter layouts, or asset formats.

## Rigid articulated models

Rigid robots and anatomical models do not implement the skinning protocol.
They derive from `RigidBodyModel`, which shares metadata, link attachment, mesh
projection, link-local mesh access, and zero-control construction. A
`link_meshes[i]` surface is transformed by `forward_links(...)[i]`; packed
vertex and face ranges remain private storage details. Their kinematics remain
local: BrainCo retains coupled-joint polynomials, G1 retains hinge axes,
SmplHumanoid retains its Euler convention, and MyoFullBody retains mixed
hinge/slide joints.

## Specialized operations

An operation belongs in the runtime only when its contract is independent of a
particular model. SOMA and MHR compute their corrective coefficients locally;
their final coefficient-to-offset map uses the same public sparse-basis
contract as every other corrective model. Hiding coefficient generation in the
global runtime would make the runtime understand model semantics and create a
leaky abstraction.

## Adding a model

1. Add asset loading and validation in `_io.py`.
2. Put model-specific numerical functions in `_core.py` and pass the array
   namespace explicitly.
3. Define the shared implementation class in `_model.py` using `ArrayRuntime` and the
   appropriate model base.
4. Bind and export the class from the NumPy, Torch, and JAX backend modules.
5. Add its factory and asset metadata to `_catalog.py`.
6. Add cross-runtime, arbitrary-batch, compile, gradient, and reference tests
   in proportion to the operations it supports.

Before promoting repeated code into `_common/`, check that the candidate has the
same meaning, inputs, outputs, batching rules, and differentiation behavior in
every caller. If those differ, keeping a small amount of explicit duplication
is preferred to adding flags or model-name branches.
