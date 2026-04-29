# BrainCo

BrainCo is a rigid articulated model of the BrainCo Revo 2 robotic hand using the official MuJoCo XML and STL assets.

BrainCo downloads automatically on first use. To prefetch and save the path:

```bash
body-models download brainco
```

When passed manually, `model_path` should contain `left.xml`, `right.xml`, and `meshes/{left,right}/*.STL`.

```python
from body_models.brainco.numpy import BrainCoHand

hand = BrainCoHand(side="right", rotation_type="hinge")
```

The model exposes the six active Revo 2 joints for each hand: thumb metacarpal, thumb proximal, and the proximal joints for index, middle, ring, and pinky. Passive distal joints are included in the skeleton and meshes.

::: body_models.brainco.numpy.BrainCoHand
