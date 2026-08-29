# GNM Head

[GNM Head](https://github.com/google/GNM) is Google's parametric statistical
head model. It controls identity, expression, neck and head pose, and both
eyes. The mesh also contains teeth and tongue geometry.

## Setup

GNM Head downloads automatically on first use from the public
[`abcamiletto/body-models`](https://huggingface.co/abcamiletto/body-models)
Hugging Face repository. To prefetch GNM Head v3.0:

```bash
body-models download gnm
```

Google releases the model and its data under the Apache License 2.0. The
hosted archive includes Google's license. See the
[upstream repository](https://github.com/google/GNM) and
[technical report](https://arxiv.org/abs/2607.23687) for source details and
citation information.

## API

`shape` has 253 identity coefficients. `expression` has 383 coefficients.
`head_rotation` controls the root neck joint, while the three entries in
`head_pose` control the head, left eye, and right eye in that order. GNM's
native geometry is already measured in meters.

The `identity_names` and `expression_names` properties expose Google's name
for every coefficient.

::: body_models.gnm.numpy.GNM
