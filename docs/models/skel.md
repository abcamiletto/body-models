# SKEL

SKEL is a body model with anatomical articulation, available in `male` and
`female` variants.

## Setup

SKEL requires registration at [skel.is.tue.mpg.de](https://skel.is.tue.mpg.de/).

```bash
body-models download skel
```

Or configure files by gender:

```bash
body-models set skel-male /path/to/skel_male.pkl
body-models set skel-female /path/to/skel_female.pkl
```

## API

::: body_models.skel.numpy.SKEL
