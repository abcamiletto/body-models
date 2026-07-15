# SmplHumanoid

SmplHumanoid is a rigid articulated humanoid model loaded from SMPL-compatible MJCF XML variants.

## Setup

SmplHumanoid downloads its XML assets from the public [`abcamiletto/body-models`](https://huggingface.co/abcamiletto/body-models) Hugging Face repository. To prefetch and save the hosted path:

```bash
# Download the SmplHumanoid MJCF XML assets.
body-models download smpl-humanoid
```

The hosted folder includes license/provenance notes for the XML variants.

## API

::: body_models.smpl_humanoid.numpy.SmplHumanoid
