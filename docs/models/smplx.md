# SMPL-X

SMPL-X extends SMPL with hands, facial expression, jaw, and eye controls.

## Setup

SMPL-X requires registration at
[smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/).

```bash
body-models download smplx
```

Or configure files by gender:

```bash
body-models set smplx-neutral /path/to/SMPLX_NEUTRAL.npz
body-models set smplx-male /path/to/SMPLX_MALE.npz
body-models set smplx-female /path/to/SMPLX_FEMALE.npz
```

For vertex mappings, see [mapped points](../api.md#mapped-points).

## API

::: body_models.smplx.numpy.SMPLX
