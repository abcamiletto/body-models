# Architecture

Each model has one implementation, bound to NumPy, Torch, or JAX through a
shared runtime layer.

Names exported from `body_models`, model packages, and backend modules are
public. Underscore-prefixed modules are private and may change without a major
release.

## Model packages

Each model lives in `body_models/<name>/` and derives from `SkinnedModel`:

| File | Responsibility |
| --- | --- |
| `_io.py` | Find assets and load them as immutable NumPy data. |
| `_core.py` | Model mathematics, given an array namespace or runtime. |
| `_model.py` | Validation, state preparation, and forward methods. |
| `numpy.py`, `torch.py`, `jax.py` | Bind the model to an array backend. |
| `__init__.py` | Export model-specific types and helpers. |

`forward_vertices()` runs in stages. Identity preparation computes rest vertices
and joints; pose preparation computes transforms and corrective coefficients;
`apply_pose_correctives()` turns those coefficients into offsets; skinning
produces the posed mesh. `forward_points()` runs the same stages through a
vertex mapping. Skeleton forwards have their own preparation path.

`parameter_spec` lists parameters in identity, pose, transform order, each with
its unbatched dimensions, role, default, and rotation representation.
`get_rest_pose()` builds its defaults from this mapping; models override it to
add named presets such as relaxed hands.

Required numerical inputs may be positional. Optional arguments are
keyword-only, in this order: local pose options, identity, global transform,
output selection.

## Linear blendshape models

SMPL, SMPL-H, SMPL-X, MANO, FLAME, and GNM share a private engine. Each model
defines its pose blocks (joint order and means); the engine handles rotation
conversion, root insertion, batch validation, forward kinematics, bind-relative
transforms, and corrective coefficients. It sees only arrays and pose blocks,
never model names or feature flags.

These models also share identity preparation. Shape and expression bases are
applied separately, which avoids concatenating them on every call.

## Runtimes

`ArrayRuntime` owns the array namespace, device- and dtype-aware construction,
state materialization, and the shared operations. Model properties expose
meshes, skeletons, and deformation bases; runtime-specific arrays stay private.

Backend imports select the runtime; `create_model()` selects it by name. Core
functions that dispatch shared operations receive the runtime, and pure
numerical helpers receive only its array namespace. Models build local
transforms; the runtime composes the kinematic tree and skins.

Backend-specific data, such as Torch/Warp gradient plans for compact skinning
weights or prepared sparse corrective bases, is built once during
materialization. A selected kernel either runs or raises; it never silently
falls back to another backend.

**Torch.** Models are `torch.nn.Module`s with their source arrays as persistent
buffers, so checkpoints are complete but can be large. Derived plans move with
the model and are rebuilt, not serialized.

**JAX.** Models are pytrees. Dataclass metadata is static, arrays are children,
and reconstruction preserves model and runtime configuration.

## Shared operations

| Module | Responsibility |
| --- | --- |
| `_common.skinning` | Dense and compact skinning, bind-relative transforms, global transforms. |
| `_common.deformation` | Linear blend shapes and dense or sparse corrective bases. |
| `_common.kinematics` | Affine transforms, rigid inversion, parent-relative offsets, forward kinematics. |

Shared operations know nothing about model names, pose layouts, or asset
formats. SOMA and MHR, for example, compute corrective coefficients locally and
use the shared basis only to turn them into offsets.

## Adding a model

1. Load and validate assets in `_io.py`.
2. Implement the mathematics in `_core.py`.
3. Define the `SkinnedModel` subclass in `_model.py`.
4. Bind and export the NumPy, Torch, and JAX classes.
5. Register the factory and asset metadata in `_catalog.py`.
6. Test cross-runtime agreement, batching, compilation, gradients, and reference
   outputs.

Share code only when its meaning, inputs, outputs, batching, and gradients agree
across callers; otherwise keep it in the model.
