# SOMA

SOMA provides a native implementation for SOMA-X assets with identity, pose, and corrective controls.

## Setup

SOMA downloads automatically on first use from `https://huggingface.co/abcamiletto/body-models`.
The Hugging Face repo records the original SOMA-X Apache 2.0 provenance. To prefetch and save the path:

```bash
# Download the SOMA-X assets used by the native SOMA implementation.
body-models download soma
```

## Notes

The native implementation does not require installing `py-soma-x`.

`body-models` supports both the legacy SOMA-X NPZ rig asset layout and the SOMA-X 0.2 asset split, where rig data lives in `SOMA_template_rig.usda` and public-rig derivation metadata lives in `SOMA_procedural_transforms.json`. With 0.2 assets, the native backend keeps the expanded internal twist-joint rig for skinning while preserving the existing 77-joint public pose API.

The constructor accepts `lod="mid"`, `lod="low"`, or `lod="xlo"`. The hosted assets are preprocessed to keep runtime loading NPZ-only: `mid` has 18,056 vertices, `low` has 4,505 vertices, and `xlo` has 612 vertices.

`prepare_identity(..., repose=True)` matches the default SOMA-X bind-pose behavior. Pass `repose=False` to keep the fitted identity rest shape and fitted skeleton before reposing it to the bind pose.
Use `prepare_identity(..., bind_pose="fit")` for the default identity-dependent bind pose. Pass `bind_pose="fit_detached"` to fit from the current shape without gradients through the fit, or `bind_pose="canonical"` to use the model bind pose directly.

`cache_identity=True` can be passed to the constructor for interactive viewers that repeatedly evaluate the same identity with different poses. The default is `False`, which keeps training and JAX-transformed calls graph-safe unless caching is explicitly requested.

## API

::: body_models.bodies.soma.numpy.SOMA
