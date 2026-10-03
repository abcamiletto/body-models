# MANO

MANO is a skinned hand model with shape and finger-pose controls.

## Setup

MANO requires registration at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/).

```bash
body-models download mano
```

Or point to existing files, one per side:

```bash
body-models set mano-right /path/to/MANO_RIGHT.pkl
body-models set mano-left /path/to/MANO_LEFT.pkl
```

## Left-hand shape space

The official left model reuses the right model's shape blend shapes, although
its mesh is mirrored. A positive shape coefficient therefore moves vertices the
same way in x on both hands
([smplx#48](https://github.com/vchoutas/smplx/issues/48)). Pass
`flip_shapedirs=True` to mirror the left shape space, so that equal coefficients
give mirror-image hands:

```python
from body_models.mano.numpy import MANO

left = MANO(side="left", flip_shapedirs=True)
```

The default keeps the official behavior, which existing left-hand fits rely on.

## API

::: body_models.mano.numpy.MANO
