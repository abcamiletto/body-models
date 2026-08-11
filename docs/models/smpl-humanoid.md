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
the default. Pass a custom MJCF file with `model_path`.

```python
model = body_models.create_model("smpl-humanoid", variant="phc")
```

## API

::: body_models.smpl_humanoid.numpy.SmplHumanoid
