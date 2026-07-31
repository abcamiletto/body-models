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

Select the `humenv`, `phc`, or `smplsim` variant with `source`; `humenv` is the
default. The same variants are available as factory names:
`create_model("humenv")`, `create_model("phc")`, and
`create_model("smplsim")`.

## API

::: body_models.smpl_humanoid.SmplHumanoid
