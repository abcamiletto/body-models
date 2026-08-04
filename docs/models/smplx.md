# SMPL-X

SMPL-X extends SMPL with articulated hands, face expression, jaw, and eye controls.

## Setup

SMPL-X requires registration at
[smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/).

```bash
body-models download smplx
```

Manual paths can also be configured per gender:

```bash
body-models set smplx-neutral /path/to/SMPLX_NEUTRAL.npz
body-models set smplx-male /path/to/SMPLX_MALE.npz
body-models set smplx-female /path/to/SMPLX_FEMALE.npz
```

## Vertex-to-joint mappings

SMPL-X can evaluate joint positions defined by any dense `[joints, vertices]`
mapping. Prepare the regressor once after placing a Torch model on its target
device, then reuse it across forwards:

```python
import numpy as np
import torch

from body_models.smplx import SMPLX

model = SMPLX(gender="neutral", runtime="torch").as_module().cuda()
mapping = np.load("captury_J_regressor.npz")["J_regressor"]
regressor = model.prepare_joint_regressor(mapping)
params = model.get_rest_pose(batch_dims=(2048,))

with torch.inference_mode():
    positions = model.forward_joint_positions(**params, joint_regressor=regressor)
# positions.shape == (2048, 67, 3)
```

The result contains positions only; `forward_skeleton()` continues to return
SMPL-X's native 55 rigid transforms. Preparation avoids materializing the posed
mesh and should happen after the Torch module is moved to its final device,
because the returned regressor does not follow later `.to()` calls.

## API

::: body_models.smplx.SMPLX
