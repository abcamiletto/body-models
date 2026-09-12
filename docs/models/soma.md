# SOMA

SOMA implements SOMA-X identity, pose, and corrective controls without requiring
`py-soma-x`.

## Setup

SOMA downloads automatically on first use from the
[`abcamiletto/body-models`](https://huggingface.co/abcamiletto/body-models)
Hugging Face repository, which records the SOMA-X Apache 2.0 provenance.
To prefetch:

```bash
body-models download soma
```

The loader requires normalized assets, as provided by the hosted archive.
To normalize upstream SOMA-X 0.2.1 assets and save their path:

```bash
body-models preprocess-soma /path/to/upstream /path/to/processed
```

SOMA exposes 77 public joints and uses an internal twist-joint rig for skinning.
The `lod` options `"mid"`, `"low"`, and `"xlo"` have 18,056, 4,505, and 612
vertices, respectively.

`prepare_identity()` defaults to `repose=True, bind_pose="fit"`, matching
SOMA-X. Use `repose=False` to retain the fitted rest shape and skeleton,
`bind_pose="fit_detached"` to stop gradients through fitting, or
`bind_pose="canonical"` for the canonical bind pose.

## API

::: body_models.soma.numpy.SOMA
