# MHR

MHR is an expressive full-body model with neural pose correctives.

## Setup

MHR downloads from the public [`abcamiletto/body-models`](https://huggingface.co/abcamiletto/body-models) Hugging Face repository on first use. The hosted package keeps the original MHR checkpoint for the default LOD 1 path and adds preprocessed FBX-derived mesh assets for LODs 0 through 6.

To prefetch and save the path:

```bash
# Download the MHR assets and store their path in the body-models config.
body-models download mhr
```

The original MHR license is included with the hosted assets.

## API

::: body_models.bodies.mhr.numpy.MHR
