![Body model lineup](assets/readme/body-model-lineup.png)

# body-models

`body-models` provides a shared interface for parametric human body, head, hand,
anatomical, measurement, and robot models across NumPy, PyTorch, and JAX.

Documentation: https://abcamiletto.github.io/body-models/

## Features

- Shared API across human, anatomical, hand, head, measurement, and robot models
- NumPy, PyTorch, and JAX backends
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
from body_models.smpl.torch import SMPL

model = SMPL(gender="neutral")
params = model.get_rest_pose(batch_dims=(1,))

vertices = model.forward_vertices(**params)
skeleton = model.forward_skeleton(**params)
```

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
both identity controls to `prepare_identity(shape, expression)`. Skeleton-only
work can use `skip_vertices=True` to avoid preparing rest vertices.

## Supported Models

- Full bodies: SMPL, SMPL-H, SMPL-X, ANNY, MHR, SOMA, GarmentMeasurements
- Anatomicals: SKEL, MyoFullBody
- Heads: FLAME
- Hands: MANO
- Robots: BrainCo, G1

See the [model docs](https://abcamiletto.github.io/body-models/#supported-models)
for setup, supported backends, inputs, and model-specific behavior.

## Development

```bash
uv run ruff format .
uv run ruff check .
uv run ty check
```

## License

See the documentation and upstream model projects for model-specific license
terms.
