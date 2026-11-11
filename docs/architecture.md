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
| `_io.py` | Validate assets (`validate_path`), fetch public ones (`download_model`), and load them as immutable NumPy data. |
| `_core.py` | Model mathematics, given an array namespace or runtime. Linear blendshape models have none. |
| `_model.py` | Validation, state preparation, and forward methods. |
| `numpy.py`, `torch.py`, `jax.py` | Bind the model to an array backend. |
| `__init__.py` | Export model-specific types and helpers. |

`forward_vertices()` runs in stages. Identity preparation computes rest vertices
and joints; pose preparation computes transforms and corrective coefficients;
`apply_pose_correctives()` turns those coefficients into offsets; skinning
produces the posed mesh. `forward_points()` runs the same stages through a
vertex mapping. Skeleton forwards have their own preparation path.

Every forward takes either the raw identity parameters or a prepared
`identity`, never both. `SkinnedModel._identity_from()` makes that choice in
one place, and raw parameters broadcast against the pose batch rather than
being expanded to it.

`parameter_spec` lists parameters in identity, pose, transform order, each with
its unbatched dimensions, role, default, and rotation representation.
`get_rest_pose()` builds its defaults from this mapping; models override it to
add named presets such as relaxed hands.

Required numerical inputs may be positional. Optional arguments are
keyword-only, in this order: local pose options, identity, global transform,
output selection.

## Catalog and assets

`_catalog.FAMILIES` holds one `ModelFamily` per public model name: its package,
class name, `create_model()` defaults, and, for licensed models, a
`LicensedArchive` with the MPI URL, credential account, and the file each asset
config key maps to. Asset config keys, `create_model()`, `list_models()`, the
CLI, and the cross-model tests all derive from it. Entries are plain data and
import no model package.

Models resolve assets with `_config.resolve_model_path(name, model_path,
variant)`: an explicit path wins; otherwise `variant` picks the config key
`<name>-<variant>`. A missing public asset is downloaded into the cache; a
missing licensed asset raises with the exact `body-models download` and
`body-models set` commands. Every path is checked by the package's
`validate_path`, which single-file assets build from `_config.validate_file`.

`_download.download_licensed()` fetches any licensed archive from its catalog
entry. `download_public_assets()` fetches `<package>/assets.zip` from the
Hugging Face repo unless the package's `validate_path` already accepts the target
directory.

## Linear blendshape models

SMPL, SMPL-H, SMPL-X, MANO, FLAME, and GNM share `LinearBlendshapeModel`. Each
model supplies two things: its pose blocks (joint order and means), and its
`IdentityBasis` list, one per identity parameter, with vertex directions and,
when the parameter moves the skeleton, joint directions. The engine does the
rest: identity preparation, rotation conversion, root insertion, forward
kinematics, bind-relative transforms, corrective coefficients, skinning, and
point regression. It sees only arrays, bases, and pose blocks, never model
names or feature flags.

A prepared linear identity keeps its coefficients, so skeleton and point
forwards never need rest vertices: points come from bases projected through the
vertex mapping. Each basis is applied separately, which avoids concatenating
shape and expression on every call.

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

1. Add one `ModelFamily` entry to `_catalog.py`, with a `LicensedArchive` if
   the assets need credentials.
2. In `_io.py`, define `validate_path`, `download_model` for public assets, and
   the loader.
3. Implement the mathematics in `_core.py`, or, for a linear blendshape model,
   define its pose blocks and identity bases on `LinearBlendshapeModel`.
4. Define the `SkinnedModel` subclass in `_model.py`, resolving assets with
   `resolve_model_path`.
5. Bind and export the NumPy, Torch, and JAX classes.
6. Add test assets and parameter snapshots; the shared tests pick the model up
   from the catalog and check cross-runtime agreement, batching, compilation,
   gradients, and reference outputs.

Share code only when its meaning, inputs, outputs, batching, and gradients agree
across callers; otherwise keep it in the model.
