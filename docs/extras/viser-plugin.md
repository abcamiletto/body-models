# viser Plugin

`body_models.extras.viser_plugin` provides small composite handles for rendering body models in `viser`.

```python
import viser
from body_models.extras import viser_plugin as vp
from body_models.g1.numpy import G1
from body_models.smpl.numpy import SMPL

server = viser.ViserServer()

model = SMPL(gender="neutral")
skeleton = vp.add_skeleton(server.scene, "/smpl/skeleton", model)
body = vp.add_body_model(server.scene, "/smpl", model)

posed = model.get_apose()
skeleton.skeleton = model.forward_skeleton(**posed)
body.set_pose(body_pose=posed["body_pose"], global_translation=posed["global_translation"])

robot = G1()
robot_pose = robot.get_rest_pose()
rigid_body = vp.add_rigid_body_model(server.scene, "/g1", robot)
rigid_body.set_pose(body_pose=robot_pose["body_pose"])
rigid_body.position = (1.0, 0.0, 0.0)
```

`vp.add_skeleton()` renders joint positions and clickable parent-child cylinder bones from `forward_skeleton()`.

`vp.add_body_model()` renders non-rigid models as skinned meshes.

`vp.add_rigid_body_model()` renders rigid articulated models from `forward_links()` and one static mesh per link.

The returned handles follow `viser` conventions: scene transforms such as `position`, `wxyz`, and `visible` are assignable properties, and resources are removed with `remove()`. Body handles expose common model parameters as explicit assignable attributes and accept bulk pose updates with `set_pose(**forward_kwargs)`. Non-rigid body handles support `shape`, `body_pose`, `hand_pose`, `head_pose`, `expression`, `global_rotation`, and `global_translation`; rigid body handles support `body_pose`, `hand_pose`, `global_rotation`, and `global_translation`.

::: body_models.extras.viser_plugin
