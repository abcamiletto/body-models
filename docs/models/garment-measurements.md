# GarmentMeasurements

GarmentMeasurements is a PCA body model with an FBX-derived skeleton and
skinning for garment measurement workflows.

## Setup

GarmentMeasurements downloads its preprocessed assets from
the [`abcamiletto/body-models`](https://huggingface.co/abcamiletto/body-models)
Hugging Face repository on first use. The repository records the original
SOMA-X Apache 2.0 provenance for the source asset. To prefetch the assets:

```bash
body-models download garment-measurements
```

## API

::: body_models.garment_measurements.GarmentMeasurements
