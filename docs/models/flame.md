# FLAME

FLAME is a head model with shape, expression, neck, jaw, and eye controls.

## Setup

FLAME requires registration at https://flame.is.tue.mpg.de/.

```bash
# Download FLAME after configuring credentials for the upstream site.
body-models download flame
```

Manual paths can also be configured:

```bash
# Point body-models at an already downloaded neutral FLAME model.
body-models set flame /path/to/FLAME_NEUTRAL.pkl
```

## API

::: body_models.parts.flame.numpy.FLAME
