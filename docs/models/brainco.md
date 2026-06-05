# BrainCo

BrainCo is a rigid articulated model of the BrainCo Revo 2 robotic hand using the official MuJoCo XML and STL assets.

## Setup

BrainCo downloads automatically on first use. To prefetch and save the path:

```bash
# Download the BrainCo MuJoCo XML and STL assets.
body-models download brainco
```

When passed manually, `model_path` should contain `left.xml`, `right.xml`, and `meshes/{left,right}/*.STL`.

## Usage

```python
from body_models.brainco.numpy import BrainCoHand

# Load the right hand with scalar hinge coordinates for the active joints.
hand = BrainCoHand(side="right", rotation_type="hinge")
```

## Notes

The model exposes the six active Revo 2 joints for each hand: thumb metacarpal, thumb proximal, and the proximal joints for index, middle, ring, and pinky. Passive distal joints are included in the skeleton and meshes.

## API

::: body_models.robots.brainco.numpy.BrainCoHand
