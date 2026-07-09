# GarmentMeasurements

GarmentMeasurements is a PCA body model with an FBX-derived skeleton and skinning, intended for garment measurement workflows.

## Setup

GarmentMeasurements downloads its preprocessed assets from
`https://huggingface.co/abcamiletto/body-models` on first use. The Hugging Face repo records
the original SOMA-X Apache 2.0 provenance for the source asset. To prefetch and save the path:

```bash
# Download the preprocessed GarmentMeasurements body-model asset.
body-models download garment-measurements
```

## API

::: body_models.bodies.garment_measurements.numpy.GarmentMeasurements
