![Body model lineup](assets/readme/body-model-lineup.png)

# body-models

`body-models` provides a shared interface for parametric human body, head, hand,
anatomical, measurement, and robot models across NumPy, PyTorch, and JAX.

Documentation: https://abcamiletto.github.io/body-models/

## Features

- Shared API across human, anatomical, hand, head, measurement, and robot models
- NumPy, PyTorch, and JAX backends
- Optional Warp skinning kernels for Torch models
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
```

## Quick Start

```python
import body_models

model = body_models.create_model("smpl", backend="torch")
params = model.get_rest_pose(batch_dims=(1,))

vertices = model.forward_vertices(params)
skeleton = model.forward_skeleton(params)
```

Use `skinning_backend="warp"` when constructing a supported Torch model to replace only
its compact skinning operation with the differentiable Warp lowering. The model
API and all model-specific preparation remain unchanged.

Discover available model names with `body_models.list_models()`. Model options
such as `gender="male"` or `side="left"` are passed as constructor kwargs.

Model inputs are immutable, model-specific parameter values. Identity controls
are grouped under `params.identity`; pose and world-placement controls are
direct fields. Use `_replace()` to derive a modified value.

```python
posed = params._replace(body_pose=body_pose)
vertices = model.forward_vertices(posed)
```

When identity parameters stay fixed across many poses, prepare them once. The
returned value has the same type, with its raw identity controls replaced by
prepared model state.

```python
person = model.prepare(params)
vertices = model.forward_vertices(person)
skeleton = model.forward_skeleton(person)
```

Prepared parameters can be reused with different poses using `_replace()`.

## Supported Models

- Full bodies: SMPL, SMPL-H, SMPL-X, ANNY, MHR, SOMA, GarmentMeasurements
- Anatomicals: SKEL, MyoFullBody
- Heads: FLAME
- Hands: MANO
- Robots: BrainCo, G1, SmplHumanoid

See the [model docs](https://abcamiletto.github.io/body-models/#supported-models)
for setup, supported backends, inputs, and model-specific behavior.
The [architecture guide](https://abcamiletto.github.io/body-models/architecture/)
describes the single-source model programs and the boundary for shared code.

## Development

```bash
uv run ruff format .
uv run ruff check .
uv run ty check
```

## License

See the documentation and upstream model projects for model-specific license
terms.
