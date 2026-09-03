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

The vertex template can be replaced with any array of the same shape, for
example the toeless template shipped with
[BEDLAM2](https://bedlam2.is.tuebingen.mpg.de/). Register it once and enable it
with `toeless=True`, or pass the vertices directly through `v_template`:

```bash
body-models set template-smplx-neutral-toeless /path/to/smplx_neutral-lh_vtemplate_toeless.obj
```

```python
from body_models.smplx.numpy import SMPLX

model = SMPLX(gender="neutral", toeless=True)
```

Like every skinned model, SMPL-X supports arbitrary vertex mappings through
[`prepare_point_regressor()` and `forward_points()`](../api.md#mapped-points).

## API

::: body_models.smplx.numpy.SMPLX
