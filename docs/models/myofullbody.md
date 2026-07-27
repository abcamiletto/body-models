# MyoFullBody

MyoFullBody is a MuJoCo-derived musculoskeletal full-body model from
`amathislab/musclemimic_models`. It exposes rigid STL link meshes, body
transforms, muscle sites, and tendon metadata.

## Setup

MyoFullBody downloads automatically on first use from the
[`abcamiletto/body-models`](https://huggingface.co/abcamiletto/body-models)
Hugging Face repository, which records the original MuscleMimic Apache 2.0
provenance. To prefetch the assets:

```bash
body-models download myofullbody
```

When passed manually, `model_path` should contain `body/myofullbody.xml` and
the referenced mesh assets from the upstream `musclemimic_models/model/` tree.

## Usage

```python
from body_models.myofullbody import MyoFullBody

model = MyoFullBody()
params = model.get_apose(batch_dims=(1,))

skeleton = model.forward_skeleton(**params)
links = model.forward_links(**params)
meshes = model.forward_meshes(**params)

sites = model.world_sites(skeleton)
```

## Notes

MyoFullBody does not define `skin_weights`. Use `forward_links()` for link
transforms and `forward_meshes()` for renderable meshes.

## API

::: body_models.myofullbody.MyoFullBody
