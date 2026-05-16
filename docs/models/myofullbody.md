# MyoFullBody

MyoFullBody is a MuJoCo-derived musculoskeletal full-body model from `amathislab/musclemimic_models`. It exposes rigid STL link meshes, body-frame skeleton transforms, site positions, and tendon metadata through the same NumPy, PyTorch, and JAX backend layout as the other models.

MyoFullBody downloads automatically on first use. To prefetch and save the path:

```bash
body-models download myofullbody
```

When passed manually, `model_path` should contain `body/myofullbody.xml` and the referenced mesh assets from the upstream `musclemimic_models/model/` tree.

```python
from body_models.myofullbody.numpy import MyoFullBody

model = MyoFullBody()
params = model.get_apose(batch_dims=(1,))
vertices = model.forward_vertices(**params)
skeleton = model.forward_skeleton(**params)
links = model.forward_links(**params)
sites = model.world_sites(skeleton)
```

MyoFullBody is a rigid articulated model, so it does not define `skin_weights`. Use `forward_links`, `link_mesh`, or `joint_meshes` when rendering or inspecting individual STL links.

::: body_models.myofullbody.numpy.MyoFullBody
