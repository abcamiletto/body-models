# SOMA

SOMA implements SOMA-X identity, pose, and corrective controls without depending
on `py-soma-x`.

## Setup

Assets download on first use from
[`abcamiletto/body-models`](https://huggingface.co/abcamiletto/body-models) on
Hugging Face (SOMA-X: Apache 2.0). To prefetch:

```bash
body-models download soma
```

The loader expects normalized assets. To normalize upstream SOMA-X 0.2.1 files
and save their path:

```bash
body-models preprocess-soma /path/to/upstream /path/to/processed
```

## Rig and resolution

SOMA exposes 77 joints and skins with an internal rig that adds twist joints.
`lod="mid"`, `"low"`, and `"xlo"` give 18,056, 4,505, and 612 vertices.

## Identity fitting

`prepare_identity()` defaults to `repose=True, bind_pose="fit"`, as in SOMA-X.
Use `repose=False` to keep the fitted rest shape and skeleton,
`bind_pose="fit_detached"` to stop gradients through the fit, or
`bind_pose="canonical"` for the canonical bind pose.

## API

::: body_models.soma.numpy.SOMA
