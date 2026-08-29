# body-models

Parametric human models for NumPy, PyTorch, and JAX behind a consistent Python
API.

## Installation

`body-models` requires Python 3.11 or newer. NumPy support is included by
default. Install an extra for PyTorch or JAX.

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

The [documentation](https://abcamiletto.github.io/body-models/) covers model
setup, parameters, supported runtimes, and the shared API.

## License

The library is licensed under Apache 2.0. Model assets retain their upstream
licenses; see the documentation for each model.
