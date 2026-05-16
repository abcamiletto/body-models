# viser Plugin

`body_models.extras.viser_plugin` provides small composite handles for rendering body models in `viser`.

```python
import viser
from body_models.extras import viser_plugin as vp
from body_models.smpl.numpy import SMPL

server = viser.ViserServer()

model = SMPL(gender="neutral")
skeleton = vp.add_skeleton(server.scene, "/smpl/skeleton", model)

posed = model.get_apose()
skeleton.skeleton = model.forward_skeleton(**posed)
```

`vp.add_skeleton()` renders joint positions and clickable parent-child cylinder bones from `forward_skeleton()`.

The returned handle follows `viser` conventions: scene transforms such as `position`, `wxyz`, and `visible` are assignable properties, and resources are removed with `remove()`.

::: body_models.extras.viser_plugin
