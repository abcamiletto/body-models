# Architecture

`body-models` has one implementation of each model and a small execution layer
for array ownership and genuinely shared operations. Framework support is no
longer organized as a matrix of model-specific backend modules.

## Model programs

Each model family follows the same file roles:

| File | Responsibility |
| --- | --- |
| `io.py` | Resolve assets and load immutable NumPy model data. |
| `core.py` | Model-specific mathematics with an explicit array namespace. |
| `model.py` | Public signature, validation, state preparation, and forward orchestration. |
| `numpy.py` | Construct the model program with `NumpyRuntime`. |
| `torch.py` | Add `nn.Module` storage and construct `TorchRuntime`. |
| `jax.py` | Construct `JaxRuntime` and use shared pytree behavior for ordinary model state. |

The framework files are intentionally thin. A signature or behavior change is
made once in `model.py`; NumPy, Torch, and JAX cannot silently drift apart.

## Runtime boundary

`Runtime` owns three things:

1. conversion of loaded model data into framework-managed arrays;
2. device- and dtype-aware array creation;
3. lowerings of stable shared operations, currently compact linear blend
   skinning and dense skin-weight expansion.

Warp is a Torch operation lowering, not a fourth copy of a model. Selecting
`kernel="warp"` changes compact skinning while identity preparation, pose
semantics, correctives, and public outputs remain the same model program.

The shared skinning module contains only operations whose signatures are stable
across model families: compact and dense linear blend skinning, bind-relative
transforms, global point transforms, and skeleton transforms. Model-specific
pose assembly and bind construction remain beside their model.

The same rule applies below the runtime boundary. `common.deformation` owns
linear blend shapes and rotation-deviation correctives; `common.kinematics`
owns affine transform assembly, rigid inversion, parent-relative offsets, and
generic forward kinematics. These functions operate on explicit arrays and do
not know model names, parameter layouts, or asset formats.

## Rigid articulated models

Rigid robots and anatomical models do not implement the skinning protocol.
They derive from `RigidModel`, which shares metadata, link attachment, mesh
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

1. Add asset loading and validation in `io.py`.
2. Put model-specific numerical functions in `core.py` and pass the array
   namespace explicitly.
3. Define the public program in `model.py` using `Runtime` or `RigidModel`.
4. Add the three thin framework constructors that the model supports.
5. Add its factory and asset metadata to `catalog.py`.
6. Add cross-framework, arbitrary-batch, compile, gradient, and reference tests
   in proportion to the operations it supports.

Before promoting repeated code into `common/`, check that the candidate has the
same meaning, inputs, outputs, batching rules, and differentiation behavior in
every caller. If those differ, keeping a small amount of explicit duplication
is preferred to adding flags or model-name branches.
