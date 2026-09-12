# body-models

Parametric human models with a shared NumPy, PyTorch, and JAX API.

## Installation

Requires Python 3.11 or newer. NumPy is included; add extras for PyTorch or JAX.

```bash
pip install body-models
pip install "body-models[torch]"
pip install "body-models[jax]"
```

## Example

```python
from body_models.gnm.numpy import GNM

model = GNM()
params = model.get_rest_pose(batch_dims=(1,))

vertices = model.forward_vertices(**params)
skeleton = model.forward_skeleton(**params)
```

Public assets such as GNM Head download on first use. Models with restricted
assets require registration with their upstream project.

## Models

| Category | Models |
| --- | --- |
| Bodies | SMPL, SMPL-H, SMPL-X, ANNY, MHR, SOMA |
| Heads | FLAME, GNM Head |
| Hands | MANO |
| Anatomy | SKEL |
| Measurements | GarmentMeasurements |

See the [documentation](https://abcamiletto.github.io/body-models/) for setup,
parameters, and API details.

## License

The library uses Apache 2.0. Model assets retain their upstream licenses;
see each model's documentation.
