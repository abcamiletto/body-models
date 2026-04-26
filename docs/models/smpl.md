# SMPL

SMPL is a skinned human body model with shape coefficients and 24 articulated joints.

Download SMPL from the official registration site, then configure the gendered model files:

```bash
body-models set smpl-neutral /path/to/SMPL_NEUTRAL.pkl
body-models set smpl-male /path/to/SMPL_MALE.pkl
body-models set smpl-female /path/to/SMPL_FEMALE.pkl
```

::: body_models.smpl.numpy.SMPL
