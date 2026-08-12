# body-models

`body-models` provides a shared interface for parametric human body, head, hand,
anatomical, and measurement models across NumPy, PyTorch, and JAX.

Documentation: https://abcamiletto.github.io/body-models/

## Features

- Shared API across human, anatomical, hand, head, and measurement models
- NumPy, PyTorch, and JAX runtimes
- Separate mesh and skeleton forwards with `forward_vertices()` and `forward_skeleton()`
- Prepared identities for repeated poses with fixed shape/expression parameters
- Mesh simplification and vertex-subset forwards for supported mesh models
- Multiple rotation representations for supported pose models
- Optional Warp-accelerated skinning for Torch models

## Install

```bash
uv add body-models
```

Install optional extras when needed:

```bash
uv add "body-models[torch]"
uv add "body-models[jax]"
uv add "body-models[torch,warp]"
uv add "body-models[simplify]"
```

Public model assets download automatically on first use. Licensed assets use
`body-models download MODEL`, which prompts for credentials.

## Quick Start

```python
from body_models.smpl.torch import SMPL

model = SMPL(gender="neutral")
params = model.get_rest_pose(batch_dims=(1,))

vertices = model.forward_vertices(**params)
skeleton = model.forward_skeleton(**params)
```

The equivalent NumPy and JAX classes live in `body_models.smpl.numpy` and
`body_models.smpl.jax`. Torch models are `torch.nn.Module` instances, so
`.to()`, `.cuda()`, and `state_dict()` work directly.

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
both identity controls to `prepare_identity(shape, expression)`.

## Supported Models

- Full bodies: SMPL, SMPL-H, SMPL-X, ANNY, MHR, SOMA, GarmentMeasurements
- Anatomicals: SKEL
- Heads: FLAME
- Hands: MANO

See the [model docs](https://abcamiletto.github.io/body-models/#supported-models)
for setup, supported runtimes, inputs, and model-specific behavior.

## Development

```bash
uv run ruff format .
uv run ruff check .
uv run ty check
```

## License

See the documentation and upstream model projects for model-specific license
terms.
