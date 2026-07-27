# ANNY

ANNY is a phenotype-driven body model with configurable rig and topology variants.

## Setup

ANNY downloads automatically on first use from the
[`abcamiletto/body-models`](https://huggingface.co/abcamiletto/body-models)
Hugging Face repository, which records the original ANNY Apache 2.0 and MPFB2
CC0 provenance. To prefetch the assets:

```bash
body-models download anny
```

## API

### Portable fitted poses

Store `rotation_type` with cached fitted parameters, then convert them when
loading into a model configured with another representation:

```python
from body_models.anny import ANNY, convert_pose

model = ANNY(rotation_type="sixd", runtime="torch")
parameters = convert_pose(cached_parameters, src=cached_rotation_type, dst=model.rotation_type)
vertices = model.forward_vertices(**parameters)
```

::: body_models.anny.ANNY
