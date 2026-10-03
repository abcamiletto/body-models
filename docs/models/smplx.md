# SMPL-X

SMPL-X extends SMPL with hands, facial expression, jaw, and eye controls. To
evaluate marker or joint regressors, see
[mapped points](../api.md#mapped-points).

## Setup

SMPL-X requires registration at
[smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/).

```bash
body-models download smplx
```

Or point to existing files, one per gender:

```bash
body-models set smplx-neutral /path/to/SMPLX_NEUTRAL.npz
body-models set smplx-male /path/to/SMPLX_MALE.npz
body-models set smplx-female /path/to/SMPLX_FEMALE.npz
```

## API

::: body_models.smplx.numpy.SMPLX
