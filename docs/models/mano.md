# MANO

MANO is a skinned hand model with shape and finger-pose controls.

## Setup

MANO requires registration at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/).

```bash
body-models download mano
```

Or configure files by side:

```bash
body-models set mano-right /path/to/MANO_RIGHT.pkl
body-models set mano-left /path/to/MANO_LEFT.pkl
```

## API

::: body_models.mano.numpy.MANO
