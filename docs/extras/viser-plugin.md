# viser

`body_models.extras.viser_plugin` provides small composite handles for rendering body models in `viser`.

Install the optional dependency before using it:

```bash
# Install the optional viser dependency for the visualization helpers.
uv add "body-models[viser]"
```

```python
import viser
from body_models.extras import viser_plugin as vp
from body_models.g1.numpy import G1
from body_models.smpl.numpy import SMPL

# Create a viser server and keep all objects in its scene tree.
server = viser.ViserServer()

# Add a skinned body model and a clickable skeleton for the same SMPL instance.
model = SMPL(gender="neutral")
rest_pose = model.get_rest_pose()
rest_joints = model.forward_skeleton(**rest_pose)[:, :3, 3]
skeleton = vp.add_skeleton(
    server.scene,
    "/smpl/skeleton",
    rest_joints,
    model.parents,
    joint_names=model.joint_names,
)
body = vp.add_body_model(server.scene, "/smpl", model)

# Update both handles from the same model pose.
posed = model.get_apose()
skeleton.joint_positions = model.forward_skeleton(**posed)[:, :3, 3]
body.set_pose(body_pose=posed["body_pose"], global_translation=posed["global_translation"])

# Rigid articulated models use one mesh per link instead of skinning weights.
robot = G1()
robot_pose = robot.get_rest_pose()
rigid_body = vp.add_rigid_body_model(server.scene, "/g1", robot)
rigid_body.set_pose(body_pose=robot_pose["body_pose"])

# Scene transform properties follow viser handle conventions.
rigid_body.position = (1.0, 0.0, 0.0)
```

`vp.add_skeleton()` renders joint positions and clickable parent-child cylinder bones from a parent list.

`vp.add_body_model()` renders non-rigid models as skinned meshes.
Pass `joint_handles=True` to add transform controls for editable joints exposed by `model.kinematic_chain`.

`vp.add_rigid_body_model()` renders rigid articulated models from `forward_links()` and one static mesh per link.

The returned handles follow `viser` conventions: scene transforms such as `position`, `wxyz`, and `visible` are assignable properties, and resources are removed with `remove()`. Body handles expose common model parameters as explicit assignable attributes and accept bulk pose updates with `set_pose(**forward_kwargs)`. Non-rigid body handles support `shape`, `body_pose`, `hand_pose`, `head_pose`, `expression`, `global_rotation`, and `global_translation`; rigid body handles support `body_pose`, `hand_pose`, `global_rotation`, and `global_translation`.

::: body_models.extras.viser_plugin
