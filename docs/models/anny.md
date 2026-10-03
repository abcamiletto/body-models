# ANNY

ANNY is a phenotype-driven body model with configurable rigs and topology.

## Setup

Assets download on first use from
[`abcamiletto/body-models`](https://huggingface.co/abcamiletto/body-models) on
Hugging Face (ANNY: Apache 2.0; MPFB2: CC0). To prefetch:

```bash
body-models download anny
```

## Fitted poses

Pose values depend on `rotation_type`, so store it with fitted parameters and
convert when loading them into a model that uses another representation:

```python
from body_models.anny import convert_pose
from body_models.anny.torch import ANNY

model = ANNY(rotation_type="sixd")
parameters = convert_pose(cached_parameters, src=cached_rotation_type, dst=model.rotation_type)
vertices = model.forward_vertices(**parameters)
```

## API

::: body_models.anny.numpy.ANNY

::: body_models.anny.convert_pose
