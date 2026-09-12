# GNM Head

[GNM Head](https://github.com/google/GNM) controls identity, expression, neck,
head, and eyes. Its mesh includes teeth and tongue geometry.

## Setup

GNM Head downloads automatically on first use from the public
[`abcamiletto/body-models`](https://huggingface.co/abcamiletto/body-models)
Hugging Face repository. To prefetch GNM Head v3.0:

```bash
body-models download gnm
```

The model and data use Apache 2.0; the hosted archive includes Google's license.
See the [source](https://github.com/google/GNM) and
[technical report](https://arxiv.org/abs/2607.23687) for citation details.

## API

`shape` has 253 coefficients and `expression` has 383, named by `identity_names`
and `expression_names`. `head_rotation` controls the root neck joint;
`head_pose` controls the head, left eye, and right eye, in that order.
Geometry is in meters.

::: body_models.gnm.numpy.GNM
