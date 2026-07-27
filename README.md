![Body model lineup](assets/readme/body-model-lineup.png)

# body-models

`body-models` provides a shared interface for parametric human body, head, hand,
anatomical, measurement, and robot models with NumPy, PyTorch, and JAX runtimes.

Documentation: https://abcamiletto.github.io/body-models/

## Features

- NumPy, PyTorch, and JAX runtimes
- Optional Warp acceleration for Torch skinning
- Separate mesh and skeleton forwards with `forward_vertices()` and `forward_skeleton()`
- Prepared identities for repeated poses with fixed shape/expression parameters
- Mesh simplification and vertex-subset forwards for supported mesh models
- Multiple rotation representations for supported pose models

## Install

```bash
uv add body-models
```

Install optional extras when needed:

```bash
uv add "body-models[torch]"
uv add "body-models[jax]"
uv add "body-models[torch,warp]"
```

## Quick Start

```python
import body_models

model = body_models.create_model("smpl", runtime="torch")
params = model.get_rest_pose(batch_dims=(1,))

vertices = model.forward_vertices(**params)
skeleton = model.forward_skeleton(**params)
```

Every model exposes a machine-readable description of those parameters:

```python
model.parameter_spec
# {
#     "shape": ParameterSpec(shape=(10,), role="identity", ...),
#     "body_pose": ParameterSpec(shape=(23, 3), role="pose", rotation_type="axis_angle", ...),
#     ...
# }
```

Each entry reports the unbatched array shape, semantic role, canonical default,
and rotation representation when applicable. `get_rest_pose()` constructs its
arrays directly from this specification.

Runtime-specific options stay in the runtime rather than every model signature:

```python
runtime = body_models.TorchRuntime(skinning_backend="warp")
model = body_models.create_model("smpl", runtime=runtime)
module = model.as_module().cuda()
```

`as_module()` returns one cached `torch.nn.Module` view per model. The view
shares numeric state with the model, so device and dtype changes apply to both,
without changing the underlying model class.

Discover available model names with `body_models.list_models()`. Model options
such as `gender="male"` or `side="left"` are passed as constructor kwargs.

## Public API

The stable API consists of names exported from `body_models` and each model
package. For example, `body_models.smpl.SMPL` is public; underscore-prefixed
modules such as `body_models.smpl._model` are implementation details and are
not covered by semantic-versioning compatibility guarantees.

When shape-dependent identity parameters stay fixed across many poses, prepare
them once and pass the returned dictionary back through `identity`. This avoids
recomputing rest joints, local offsets, and rest vertices on every forward pass.

```python
shape = params.pop("shape")
identity = model.prepare_identity(shape)

vertices = model.forward_vertices(**params, identity=identity)
skeleton = model.forward_skeleton(**params, identity=identity)
```

For models with expression-dependent rest state, such as SMPL-X and FLAME, pass
both identity controls to `prepare_identity(shape, expression)`. Prepared
identities and poses are always complete mesh-ready values; skeleton forwards
use separate lightweight internal preparation and never return partial state.

## Supported Models

- Full bodies: SMPL, SMPL-H, SMPL-X, ANNY, MHR, SOMA, GarmentMeasurements
- Anatomicals: SKEL, MyoFullBody
- Heads: FLAME
- Hands: MANO
- Robots and humanoids: BrainCo, G1, SmplHumanoid

See the [model docs](https://abcamiletto.github.io/body-models/#supported-models)
for setup, supported runtimes, inputs, and model-specific behavior.
The [architecture guide](https://abcamiletto.github.io/body-models/architecture/)
describes the model, runtime, and shared-operation boundaries.

## Development

```bash
uv run ruff format .
uv run ruff check .
uv run ty check
```

## License

See the documentation and upstream model projects for model-specific license
terms.
