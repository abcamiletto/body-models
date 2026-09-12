# ANNY

ANNY is a phenotype-driven body model with configurable rigs and topology.

## Setup

ANNY downloads automatically on first use from the
[`abcamiletto/body-models`](https://huggingface.co/abcamiletto/body-models)
Hugging Face repository (ANNY: Apache 2.0; MPFB2: CC0). To prefetch:

```bash
body-models download anny
```

## API

### Portable fitted poses

Save `rotation_type` with fitted parameters. Convert the parameters when loading
into a model with a different representation:

```python
from body_models.anny import convert_pose
from body_models.anny.torch import ANNY

model = ANNY(rotation_type="sixd")
parameters = convert_pose(cached_parameters, src=cached_rotation_type, dst=model.rotation_type)
vertices = model.forward_vertices(**parameters)
```

::: body_models.anny.numpy.ANNY

::: body_models.anny.convert_pose
