# MHR

MHR is an expressive full-body model with facial expression controls and neural
pose correctives.

## Setup

MHR downloads from the public
[`abcamiletto/body-models`](https://huggingface.co/abcamiletto/body-models)
Hugging Face repository on first use. The hosted assets include the original
MHR checkpoint for LOD 1 and preprocessed FBX-derived meshes for LODs 0–6.

To prefetch the assets:

```bash
body-models download mhr
```

The original MHR license is included with the hosted assets.

## API

::: body_models.mhr.numpy.MHR
