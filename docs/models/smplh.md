# SMPL-H

SMPL-H extends SMPL with articulated MANO hands.

Download SMPL-H from the official MANO registration site, then configure the gendered model files:

```bash
body-models download smplh
```

This downloads the "Extended SMPL+H model (used in AMASS project)" archive and configures:

```bash
body-models set smplh-neutral /path/to/smplh/neutral/model.npz
body-models set smplh-male /path/to/smplh/male/model.npz
body-models set smplh-female /path/to/smplh/female/model.npz
```

::: body_models.smplh.numpy.SMPLH
