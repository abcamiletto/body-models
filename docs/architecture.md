# Architecture

`body-models` has one implementation of each model and a small execution layer
for array ownership and genuinely shared operations. Framework support is no
longer organized as a matrix of model-specific backend modules.

## Public API boundary

The stable public API is intentionally small:

- names exported from `body_models`;
- names explicitly exported from a model package; and
- the `numpy`, `torch`, and `jax` modules within each model package.

All underscore-prefixed modules are private implementation details. This
includes model programs and loaders such as `smpl._model` and `smpl._io`, and
shared infrastructure such as `_runtime`, `_state`, and `_common`. They may
change without a major release. There are deliberately no compatibility aliases
for their pre-1.0 names.

## Model programs

Each model family follows the same file roles:

| File | Responsibility |
| --- | --- |
| `_io.py` | Resolve assets and load immutable NumPy model data. |
| `_core.py` | Model-specific mathematics with an explicit array namespace. |
| `_model.py` | Define validation, state preparation, and forward orchestration. |
| `body_models/<name>/numpy.py` | Construct the model program with `NumpyRuntime`. |
| `body_models/<name>/torch.py` | Add `nn.Module` storage and construct `TorchRuntime`. |
| `body_models/<name>/jax.py` | Construct `JaxRuntime` and define JAX pytree behavior. |

Every model is self-contained in `body_models/<name>/`; descriptive categories
do not create a second package tree. The wrappers are intentionally thin: a
signature or behavior change is made once in `_model.py`, so backends cannot
drift apart.
Public identity and pose preparation always returns complete mesh-ready state.
Skeleton forwards use distinct model-local preparation paths, so an optimization
cannot create a partial object that later fails in a mesh forward.

## Runtime boundary

`ArrayRuntime` owns the array namespace, device- and dtype-aware construction,
state materialization, and lowerings of stable shared operations such as
compact linear blend skinning. Materialization delegates to the recursive
converters in `_state.py`; callers therefore cannot pair a runtime with the
wrong framework state. The runtime does not own model semantics.

Warp is a Torch operation lowering, not a fourth copy of a model. Selecting
`skinning_backend="warp"` changes compact skinning while identity preparation, pose
semantics, correctives, and public outputs remain the same model program.

Linear identity preparation is shared by the SMPL family, MANO, and FLAME
because those models apply the same coefficients to vertex and joint bases.
Each model still assembles its own coefficient vector and bases; model-specific
pose construction remains local.

The shared skinning module contains only operations whose signatures are stable
across model families: compact and dense linear blend skinning, bind-relative
transforms, global point transforms, and skeleton transforms. Model-specific
pose assembly and bind construction remain beside their model.

The same rule applies below the runtime boundary. `_common.deformation` owns
linear blend shapes and rotation-deviation correctives; `_common.kinematics`
owns affine transform assembly, rigid inversion, parent-relative offsets, and
generic forward kinematics. These functions operate on explicit arrays and do
not know model names, parameter layouts, or asset formats.

## Rigid articulated models

Rigid robots and anatomical models do not implement the skinning protocol.
They derive from `RigidBodyModel`, which shares metadata, link attachment, mesh
projection, and zero-control construction. Their kinematics remain local:
BrainCo retains coupled-joint polynomials, G1 retains hinge axes, SmplHumanoid
retains its Euler convention, and MyoFullBody retains mixed hinge/slide joints.

## Specialized operations

An operation belongs in the runtime only when its contract is independent of a
particular model. SOMA's learned sparse corrective network is the deliberate
counterexample: it is a visible SOMA component with optimized NumPy/SciPy,
Torch sparse, and JAX scatter implementations. Hiding it in the global runtime
would make the runtime understand SOMA and create a leaky abstraction.

## Adding a model

1. Add asset loading and validation in `_io.py`.
2. Put model-specific numerical functions in `_core.py` and pass the array
   namespace explicitly.
3. Define the public program in `_model.py` using `ArrayRuntime` and the
   appropriate model base.
4. Add the three thin framework constructors that the model supports.
5. Add its factory and asset metadata to `_catalog.py`.
6. Add cross-framework, arbitrary-batch, compile, gradient, and reference tests
   in proportion to the operations it supports.

Before promoting repeated code into `_common/`, check that the candidate has the
same meaning, inputs, outputs, batching rules, and differentiation behavior in
every caller. If those differ, keeping a small amount of explicit duplication
is preferred to adding flags or model-name branches.
