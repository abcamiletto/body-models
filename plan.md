# Multi-Backend Body Models

## Architecture

Each model (SMPL, SMPL-X, FLAME, SKEL, ANNY, MHR) follows this structure:

```
src/body_models/<model>/
├── __init__.py      # Re-exports from torch.py (default backend)
├── core.py          # Backend-agnostic computation using array_api_compat
├── torch.py         # PyTorch nn.Module wrapper
├── numpy.py         # Plain Python class wrapper
├── jax.py           # Flax nnx.Module wrapper
└── io.py            # Model loading, path resolution, mesh simplification
```

## Implementation Pattern

### 1. core.py - Backend-Agnostic Computation

- Use `array_api_compat.get_namespace(array)` to get the array backend (`xp`)
- Use `xp.` for all array operations (zeros, stack, einsum, etc.)
- Use `common.set(array, np.index_exp[...], values)` for element assignment (handles JAX immutability)
- Use `nanomanifold.SO3` for rotation operations (already backend-agnostic)
- Type hints use `jaxtyping` with generic `Array = Any`

Example:
```python
from array_api_compat import get_namespace
from .. import common

def forward_vertices(..., shape: Float[Array, "B 10"], ...) -> Float[Array, "B V 3"]:
    xp = get_namespace(shape)
    # Use xp.zeros, xp.einsum, xp.stack, etc.
    # Use common.set(T, np.index_exp[..., :3, :3], R) for assignment
```

### 2. torch.py - PyTorch Wrapper

- Inherit from `nn.Module`
- Use `register_buffer()` for non-trainable tensors
- Use `nn.Parameter(..., requires_grad=False)` for blend shapes (better device handling)
- Add type declarations for buffers to satisfy type checkers:
  ```python
  class Model(nn.Module):
      v_template: Tensor  # Type declaration
      def __init__(self):
          self.register_buffer("v_template", ...)
  ```

### 3. numpy.py - NumPy Wrapper

- Plain Python class (no base class)
- Store arrays as instance attributes directly
- Simplest wrapper

### 4. jax.py - JAX/Flax Wrapper

- Inherit from `flax.nnx.Module`
- Store arrays as `nnx.Variable(jnp.asarray(...))`
- Access values with `variable[...]` syntax (NOT `.value` - deprecated)

### 5. io.py - Shared I/O

- `get_model_path(path, gender)` - Resolve model file path
- `load_model_data(path)` - Load .pkl or .npz files
- `compute_kinematic_fronts(parents)` - Precompute FK traversal order
- `simplify_mesh(vertices, faces, target)` - Quadric decimation

## Key Utilities

### common.set() - Backend-Independent Array Assignment

```python
# In common.py
def set(array, slices, values, *, copy=True):
    if hasattr(array, "at"):  # JAX
        return array.at[slices].set(values)
    # NumPy/PyTorch: classic assignment
    if copy:
        array = array.clone() if hasattr(array, "clone") else xp.asarray(array, copy=True)
    array[slices] = values
    return array

# Usage in core.py
T = common.set(T, np.index_exp[..., :3, :3], R)
```

## Testing

- `tests/test_<model>_backends.py` - Test all backends match reference outputs
- `tests/test_gradients.py` - Verify gradient flow (PyTorch only)

## Progress

- [x] SMPL - Complete (torch, numpy, jax)
- [x] SMPL-X - Complete (torch, numpy, jax)
- [x] FLAME - Complete (torch, numpy, jax)
- [x] SKEL - Complete (torch with skeleton mesh, numpy/jax without skeleton mesh)
- [x] ANNY - Complete (torch, numpy, jax)
- [x] MHR - Complete (torch with pose correctives, numpy/jax without pose correctives)
