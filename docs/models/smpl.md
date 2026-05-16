# SMPL

SMPL is a skinned human body model with shape coefficients and 24 articulated joints.

## Setup

SMPL requires registration at https://smpl.is.tue.mpg.de/.

```bash
body-models download smpl
```

Manual paths can also be configured per gender:

```bash
body-models set smpl-neutral /path/to/SMPL_NEUTRAL.pkl
body-models set smpl-male /path/to/SMPL_MALE.pkl
body-models set smpl-female /path/to/SMPL_FEMALE.pkl
```

## API

::: body_models.smpl.numpy.SMPL
