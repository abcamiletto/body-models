# SmplHumanoid

SmplHumanoid is a rigid articulated humanoid model loaded from SMPL-compatible
MJCF variants.

## Setup

SmplHumanoid downloads its XML assets on first use from the public
[`abcamiletto/body-models`](https://huggingface.co/abcamiletto/body-models)
Hugging Face repository. To prefetch all variants:

```bash
body-models download smpl-humanoid
```

The hosted folder includes license/provenance notes for the XML variants.

Select the `humenv`, `phc`, or `smplsim` variant with `variant`; `humenv` is
the default. Pass a custom MJCF file with `model_path`. The hosted variants
are also available as factory names:
`create_model("humenv")`, `create_model("phc")`, and
`create_model("smplsim")`.

## API

::: body_models.smpl_humanoid.SmplHumanoid
