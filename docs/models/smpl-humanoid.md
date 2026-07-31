# SMPL humanoids and mannequins

This package provides three rigid, non-skinned humanoid APIs:

- `SmplHumanoid` loads the HumEnv, PHC, and SMPLSim MJCF variants.
- `SmplMannequin` loads the authored mannequin with a 54-joint body-and-hand hierarchy.
- `SmplxMannequin` accepts SMPL-X motion and shape parameters while keeping every mesh part rigid.

The mannequin is available at three exactly mirrored levels of detail: `mannequin`
(about 40k vertices), `mannequin_lod1` (about 15k), and
`mannequin_lod2` (under 5k). Shape coefficients change symmetric bone lengths
and move the rigid meshes; they do not change link thickness or introduce skinning.

## Setup

Assets download on first use from the public
[Hugging Face repository](https://huggingface.co/abcamiletto/body-models).
To prefetch every humanoid variant:

```bash
body-models download smpl-humanoid
```

Pass a custom MJCF file with `model_path`. Factory names include `humenv`,
`phc`, `smplsim`, `mannequin`, `mannequin-lod1`, `mannequin-lod2`,
and `smplx-mannequin`.

Select the `humenv`, `phc`, or `smplsim` variant with `variant`; `humenv` is
the default. Pass a custom MJCF file with `model_path`.

```python
model = body_models.create_model("smpl-humanoid", variant="phc")
```

Prepare a SMPL-X shape once and reuse it for a motion sequence:

```python
import numpy as np

from body_models.smpl_humanoid import SmplxMannequin

model = SmplxMannequin()
identity = model.prepare_identity(np.zeros(10, dtype=np.float32))
params = model.get_tpose()
params.pop("shape")
params.pop("expression")

vertices = model.forward_vertices(**params, identity=identity)
joints = model.forward_skeleton(**params, identity=identity)
```

## API

::: body_models.smpl_humanoid.SmplHumanoid

::: body_models.smpl_humanoid.SmplMannequin

::: body_models.smpl_humanoid.SmplxMannequin
