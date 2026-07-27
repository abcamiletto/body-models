# SmplHumanoid

SmplHumanoid is a rigid articulated humanoid model loaded from SMPL-compatible
MJCF variants.

## Setup

SmplHumanoid downloads its XML assets from the public
[`abcamiletto/body-models`](https://huggingface.co/abcamiletto/body-models)
Hugging Face repository. To prefetch all variants:

```bash
body-models download smpl-humanoid
```

The hosted folder includes license/provenance notes for the XML variants.

Select the `humenv`, `phc`, or `smplsim` variant with `source`; `humenv` is the
default.

## API

::: body_models.smpl_humanoid.SmplHumanoid
