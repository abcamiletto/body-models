# SMPL-H

SMPL-H extends SMPL with articulated MANO hands.

## Setup

SMPL-H requires registration at https://mano.is.tue.mpg.de/.

```bash
body-models download smplh
```

Manual paths can also be configured per gender:

```bash
body-models set smplh-neutral /path/to/smplh/neutral/model.npz
body-models set smplh-male /path/to/smplh/male/model.npz
body-models set smplh-female /path/to/smplh/female/model.npz
```

## Notes

The downloader uses the "Extended SMPL+H model (used in AMASS project)" archive.

## API

::: body_models.smplh.numpy.SMPLH
