# SKEL

SKEL is a human body model with anatomically motivated skeletal articulation.

## Setup

SKEL requires registration at https://skel.is.tue.mpg.de/.

```bash
body-models download skel
```

Manual paths can also be configured per gender:

```bash
body-models set skel-male /path/to/skel_male.pkl
body-models set skel-female /path/to/skel_female.pkl
```

## Notes

SKEL supports `male` and `female` genders.

## API

::: body_models.skel.numpy.SKEL
