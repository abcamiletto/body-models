# SKEL

SKEL is a body model with an anatomical skeleton, in `male` and `female`
variants.

## Setup

SKEL requires registration at [skel.is.tue.mpg.de](https://skel.is.tue.mpg.de/).

```bash
body-models download skel
```

Or point to existing files, one per gender:

```bash
body-models set skel-male /path/to/skel_male.pkl
body-models set skel-female /path/to/skel_female.pkl
```

## API

::: body_models.skel.numpy.SKEL
