# Architecture

`body-models` has one implementation of each model and a small execution layer
for array ownership and genuinely shared operations.

## Public API boundary

The stable public API is intentionally small:

- names exported from `body_models`;
- names explicitly exported from a model package.

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
| `_model.py` | Define the model class, validation, state preparation, and forward orchestration. |
| `__init__.py` | Export the model class and give it its stable public identity. |

Every model is self-contained in `body_models/<name>/`; descriptive categories
do not create a second package tree. There is one class per model, independent
of its runtime, so `isinstance`, error messages, and pickles all use the stable
package identity and signatures cannot drift across frameworks.
Public identity and pose preparation always returns complete mesh-ready state.
Skeleton forwards use distinct model-local preparation paths, so an optimization
cannot create a partial object that later fails in a mesh forward.

SMPL, SMPL-H, SMPL-X, MANO, and FLAME share one private family engine. Their
`_core.py` modules describe the ordered pose blocks and apply model-specific
means, while the engine owns rotation conversion, root insertion, batch
validation, forward kinematics, bind-relative transforms, and pose correctives.
The public methods remain explicit per model. The engine accepts arrays and
pose blocks only; it has no model names, optional-feature flags, or knowledge of
hands and faces.

Each instance exposes `parameter_spec`, an ordered mapping from public parameter
names to `ParameterSpec`. A specification records the unbatched array shape,
semantic role (`identity`, `pose`, or `transform`), canonical default, and
rotation representation where applicable. Dimensions derived from assets or
configuration are therefore represented accurately. The shared base constructs
`get_rest_pose()` from this mapping; model-local overrides only apply named
presets such as flat or relaxed hands.

## Runtime boundary

`ArrayRuntime` owns the array namespace, device- and dtype-aware construction,
state materialization, and lowerings of stable shared operations such as
compact linear blend skinning. Materialization delegates to the recursive
converters in `_state.py`; callers therefore cannot pair a runtime with the
wrong framework state. Materialized weights are private because their container
types are backend-specific; stable model properties provide public access to
meshes, skeletons, and deformation bases. The runtime does not own model
semantics.

Models accept either a runtime name or an `ArrayRuntime` instance. Runtime
options are configured once on that object, so adding an execution option does
not change every model constructor. Warp is a Torch operation lowering, not a
fourth model backend:

```python
from body_models import TorchRuntime
from body_models.smpl import SMPL

model = SMPL(runtime=TorchRuntime(skinning_backend="warp"))
```

Kernel dispatch follows the lifetime of its inputs. Stateless operations whose
inputs arrive with each call, such as skinning, are runtime methods. Operations
that own model-lifetime prepared data, such as sparse corrective multiplication,
are backend-materialized state objects. This keeps `ArrayRuntime` independent of
model state and prevents it from accumulating every accelerated operation.

Torch lifecycle behavior is orthogonal to model identity. `model.as_module()`
wraps Torch-backed state in `torch.nn.Module` semantics for `.to()`,
`state_dict()`, and buffer registration. All numeric model state is a persistent
buffer, so checkpoints are complete but may be large. JAX-backed instances of
the same model class implement the pytree protocol. Everything needed to
reconstruct a model is either a pytree child or static config; reconstruction
has no subclass hooks.

Linear identity preparation is shared by the SMPL family because those models
apply the same coefficients to vertex and joint bases. Shape-only and
shape-plus-expression paths are separate functions so their signatures state
their requirements without mode flags.

The shared skinning module contains only operations whose signatures are stable
across model families: compact and dense linear blend skinning, bind-relative
transforms, global point transforms, and skeleton transforms. Model-specific
pose layouts remain beside their model; the family engine composes those layouts
with the generic kinematics and skinning operations.

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
3. Define the public class in `_model.py` using `ArrayRuntime` and the
   appropriate model base.
4. Export the class from the model package.
5. Add its factory and asset metadata to `_catalog.py`.
6. Add cross-runtime, arbitrary-batch, compile, gradient, and reference tests
   in proportion to the operations it supports.

Before promoting repeated code into `_common/`, check that the candidate has the
same meaning, inputs, outputs, batching rules, and differentiation behavior in
every caller. If those differ, keeping a small amount of explicit duplication
is preferred to adding flags or model-name branches.
