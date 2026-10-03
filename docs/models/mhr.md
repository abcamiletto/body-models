# MHR

MHR is a full-body model with facial expression controls and neural pose
correctives.

## Setup

Assets download on first use from
[`abcamiletto/body-models`](https://huggingface.co/abcamiletto/body-models) on
Hugging Face, together with the original MHR license. They include the original
checkpoint for LOD 1 and preprocessed meshes for LODs 0–6. To prefetch:

```bash
body-models download mhr
```

## API

::: body_models.mhr.numpy.MHR
