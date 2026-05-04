# Multi-Backend Model Design

## Goal

Keep backend-independent math in one place, while letting each backend own its data representation and override only the operations that need to change.

## Layout

```text
body_models/<model>/
  io.py
  backend/
    core.py
    numpy.py
    scipy.py
    torch.py
    jax.py
```

## Responsibilities

- `io.py` loads files and returns a canonical NumPy `ModelData` dataclass.
- `backend/core.py` implements backend-independent math and takes `ModelData` instead of many individual arrays.
- Simple backends re-export `core.py`.
- Specialized backends re-export what they can and re-implement only what they must.
- Backend data classes own prepared backend-specific representation.
- Backend modules own backend-specific behavior.
- Public wrappers load IO data, prepare backend data, then call backend module functions.

## Data

`io.py` should return canonical data:

```python
@dataclass(frozen=True)
class ModelData:
    template: np.ndarray
    shapedirs: np.ndarray
    weights: np.ndarray
    faces: np.ndarray
```

Backends expose compatible data objects:

```python
NumpyModelData = core.ModelData

@dataclass(frozen=True)
class ScipyModelData(core.ModelData):
    pass
```

Backend subclasses only add fields when they need genuinely new prepared state. Often they only change the runtime representation of an existing field, such as storing `weights` as CSR.

For Torch/JAX, use the native structure that works with the framework (`nn.Module`, Flax/NNX, pytree dataclasses), while preserving the same field contract.

## Calls

Core functions take data objects:

```python
def forward_vertices(data: ModelData, ...):
    ...
```

Wrappers call backend module functions:

```python
raw_data = io.load_model_data(path)
self.data = backend.prepare_data(raw_data)

return backend.forward_vertices(self.data, ...)
```

## Constraints

- No backend branches in `core.py`.
- No object probing in shared core.
- No safety fallbacks for unsupported backend shapes.
- No repeated conversion of static data inside forward calls.
- No behavior hidden on data objects unless the backend data object is intentionally the execution object.

## SOMA Target

- `io.py` returns canonical NumPy `SomaData`.
- `backend/core.py` contains backend-independent SOMA math and accepts `SomaData`.
- `backend/numpy.py` mostly re-exports core.
- `backend/scipy.py` prepares sparse data and overrides sparse-heavy operations.
- `backend/torch.py` provides Torch-compatible data and Torch-specific scatter operations.
- `backend/jax.py` provides JAX-compatible data and JAX-specific update operations.
