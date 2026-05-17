![Body model lineup](assets/readme/body-model-lineup.png)

# body-models

`body-models` provides a shared interface for parametric human body, head, hand,
anatomical, measurement, and robot models across NumPy, PyTorch, and JAX.

Documentation: https://abcamiletto.github.io/body-models/

## Features

- Shared API across human, anatomical, hand, head, measurement, and robot models
- NumPy, PyTorch, and JAX backends
- Separate mesh and skeleton forwards with `forward_vertices()` and `forward_skeleton()`
- Skinned-mesh and rigid-body helpers for `viser`
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
uv add "body-models[viser]"
```

## Quick Start

```python
from body_models.smpl.torch import SMPL

model = SMPL(gender="neutral")
params = model.get_rest_pose(batch_dims=(1,))

vertices = model.forward_vertices(**params)
skeleton = model.forward_skeleton(**params)
```

## Supported Models

- Full bodies: SMPL, SMPL-H, SMPL-X, ANNY, MHR, SOMA, GarmentMeasurements
- Anatomicals: SKEL, MyoFullBody
- Heads: FLAME
- Hands: MANO
- Robots: BrainCo, G1

See the [model docs](https://abcamiletto.github.io/body-models/#supported-models)
for setup, supported backends, inputs, and model-specific behavior.

## Extras

Optional integrations live under `body_models.extras`, including the
[viser plugin](https://abcamiletto.github.io/body-models/extras/viser-plugin/).

```python
import viser
from body_models.extras import viser_plugin as vp
from body_models.smpl.numpy import SMPL

server = viser.ViserServer()
model = SMPL(gender="neutral")
handle = vp.add_body_model(server.scene, "/body", model)
handle.set_pose(**model.get_rest_pose())
```

## Development

```bash
uv run ruff format .
uv run ruff check .
uv run ty check
```

## License

See the documentation and upstream model projects for model-specific license
terms.
