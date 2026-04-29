# MANO

MANO is a skinned hand model with shape and articulated finger pose parameters.

## Setup

MANO requires registration at https://mano.is.tue.mpg.de/.

```bash
body-models download mano
```

Manual paths can also be configured per side:

```bash
body-models set mano-right /path/to/MANO_RIGHT.pkl
body-models set mano-left /path/to/MANO_LEFT.pkl
```

## API

::: body_models.mano.numpy.MANO
