# SMPL-X

SMPL-X extends SMPL with articulated hands, face expression, jaw, and eye controls.

## Setup

SMPL-X requires registration at https://smpl-x.is.tue.mpg.de/.

```bash
# Download SMPL-X after configuring credentials for the upstream site.
body-models download smplx
```

Manual paths can also be configured per gender:

```bash
# Configure local SMPL-X files when you already have the assets on disk.
body-models set smplx-neutral /path/to/SMPLX_NEUTRAL.npz
body-models set smplx-male /path/to/SMPLX_MALE.npz
body-models set smplx-female /path/to/SMPLX_FEMALE.npz
```

## API

::: body_models.smplx.numpy.SMPLX
