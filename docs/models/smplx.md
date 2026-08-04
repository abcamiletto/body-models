# SMPL-X

SMPL-X extends SMPL with articulated hands, face expression, jaw, and eye controls.

## Setup

SMPL-X requires registration at
[smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/).

```bash
body-models download smplx
```

Manual paths can also be configured per gender:

```bash
body-models set smplx-neutral /path/to/SMPLX_NEUTRAL.npz
body-models set smplx-male /path/to/SMPLX_MALE.npz
body-models set smplx-female /path/to/SMPLX_FEMALE.npz
```

Like every skinned model, SMPL-X supports arbitrary vertex mappings through
[`prepare_point_regressor()` and `forward_points()`](../api.md#mapped-points).

## API

::: body_models.smplx.SMPLX
