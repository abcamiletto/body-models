# GNM Head

[GNM Head](https://github.com/google/GNM) is a head model with identity,
expression, neck, head, and eye controls. Its mesh includes teeth and tongue.

## Setup

GNM Head v3.0 downloads on first use from
[`abcamiletto/body-models`](https://huggingface.co/abcamiletto/body-models) on
Hugging Face. The model and data are Apache 2.0, and the archive includes
Google's license. To prefetch:

```bash
body-models download gnm
```

For citation details, see the [source](https://github.com/google/GNM) and the
[technical report](https://arxiv.org/abs/2607.23687).

## Parameters

`shape` has 253 coefficients and `expression` has 383, named by `identity_names`
and `expression_names`. `head_rotation` drives the root neck joint; `head_pose`
drives the head, left eye, and right eye, in that order. Units are meters.

## API

::: body_models.gnm.numpy.GNM
