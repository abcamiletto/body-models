# viser Plugin

`body_models.extras.viser_plugin` provides small composite handles for rendering body models in `viser`.

```python
import viser
from body_models.extras import viser_plugin as vp
from body_models.smpl.numpy import SMPL

server = viser.ViserServer()

model = SMPL(gender="neutral")
skeleton = vp.add_skeleton(server.scene, "/smpl/skeleton", model)
body = vp.add_body_model(server.scene, "/smpl", model)

posed = model.get_apose()
skeleton.skeleton = model.forward_skeleton(**posed)
body.body_pose = posed["body_pose"]
body.global_translation = posed["global_translation"]
```

`vp.add_skeleton()` renders joint positions and clickable parent-child cylinder bones from `forward_skeleton()`.

`vp.add_body_model()` renders non-rigid models as skinned meshes when possible, with a simple mesh fallback.

The returned handles follow `viser` conventions: scene transforms such as `position`, `wxyz`, and `visible` are assignable properties, and resources are removed with `remove()`. Body handles expose common model parameters as explicit assignable attributes, including `body_pose`, `hand_pose`, `head_pose`, `wrist_rotation`, `shape`, `identity`, `expression`, `global_rotation`, and `global_translation`. Accessing a parameter that the model does not support raises an `AttributeError`.

::: body_models.extras.viser_plugin
