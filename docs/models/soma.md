# SOMA

SOMA provides a native implementation for SOMA-X assets with identity, pose,
and corrective controls.

## Setup

SOMA downloads automatically on first use from the
[`abcamiletto/body-models`](https://huggingface.co/abcamiletto/body-models)
Hugging Face repository, which records the original SOMA-X Apache 2.0
provenance. To prefetch the assets:

```bash
body-models download soma
```

## Notes

The native implementation does not require installing `py-soma-x`.

`body-models` supports both the original SOMA-X NPZ rig layout and the SOMA-X
0.2 split assets. With 0.2 assets, the implementation retains the internal
twist-joint rig for skinning while exposing the 77-joint public pose API.

The constructor accepts `lod="mid"`, `lod="low"`, or `lod="xlo"`. The hosted
assets contain 18,056, 4,505, and 612 vertices respectively.

`prepare_identity()` uses `repose=True` and `bind_pose="fit"` by default,
matching SOMA-X bind-pose behavior. Set `repose=False` to keep the fitted rest
shape and skeleton, `bind_pose="fit_detached"` to stop gradients through the
fit, or `bind_pose="canonical"` to use the canonical bind pose.

## API

::: body_models.soma.SOMA
