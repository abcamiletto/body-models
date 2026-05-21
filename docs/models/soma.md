# SOMA

SOMA provides a native implementation for SOMA-X assets with identity, pose, and corrective controls.

## Setup

SOMA downloads automatically on first use. To prefetch and save the path:

```bash
# Download the SOMA-X asset used by the native SOMA implementation.
body-models download soma
```

## Notes

The native implementation does not require installing `py-soma-x`.

`cache_identity=True` can be passed to the constructor for interactive viewers that repeatedly evaluate the same identity with different poses. The default is `False`, which keeps training and JAX-transformed calls graph-safe unless caching is explicitly requested.

## API

::: body_models.soma.numpy.SOMA
