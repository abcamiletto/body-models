# MyoFullBody

MyoFullBody is a MuJoCo-derived musculoskeletal full-body model from `amathislab/musclemimic_models`. It exposes rigid STL link meshes, body-frame skeleton transforms, site positions, and tendon metadata through the same NumPy, PyTorch, and JAX backend layout as the other models.

## Setup

MyoFullBody downloads automatically on first use from `https://huggingface.co/abcamiletto/body-models`.
The Hugging Face repo records the original MuscleMimic Apache 2.0 provenance. To prefetch and save the path:

```bash
# Download the MyoFullBody MJCF and referenced mesh assets.
body-models download myofullbody
```

When passed manually, `model_path` should contain `body/myofullbody.xml` and the referenced mesh assets from the upstream `musclemimic_models/model/` tree.

## Usage

```python
from body_models.myofullbody.numpy import MyoFullBody

# Load the rigid articulated model and start from its bundled A-pose.
model = MyoFullBody()
params = model.get_apose(batch_dims=(1,))

# Evaluate the concatenated link meshes, body-frame skeleton, and link transforms.
vertices = model.forward_vertices(**params)
skeleton = model.forward_skeleton(**params)
links = model.forward_links(**params)

# Lift MuJoCo site positions from local body frames into world coordinates.
sites = model.world_sites(skeleton)
```

## Notes

MyoFullBody is a rigid articulated model, so it does not define `skin_weights`. Use `forward_links`, `link_mesh`, or `joint_meshes` when rendering or inspecting individual STL links.

## API

::: body_models.myofullbody.numpy.MyoFullBody
