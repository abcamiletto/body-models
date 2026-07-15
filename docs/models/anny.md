# ANNY

ANNY is a phenotype-driven body model with configurable rig and topology variants.

## Setup

ANNY downloads automatically on first use from `https://huggingface.co/abcamiletto/body-models`.
The Hugging Face repo records the original ANNY Apache 2.0 and MPFB2 CC0 provenance. To prefetch
and save the path:

```bash
# Download the ANNY assets and store their path in the body-models config.
body-models download anny
```

## API

### Portable fitted poses

Store ``rotation_type`` with cached fitted parameters, then convert them when
loading into a model configured with another representation:

```python
from body_models.anny import convert_pose
from body_models.anny.torch import ANNY

model = ANNY(rotation_type="sixd")
parameters = convert_pose(cached_parameters, src=cached_rotation_type, dst=model.rotation_type)
vertices = model.forward_vertices(**parameters)
```

::: body_models.anny.numpy.ANNY
