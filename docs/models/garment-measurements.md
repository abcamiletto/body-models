# GarmentMeasurements

GarmentMeasurements is a PCA body model with an FBX-derived skeleton and skinning, intended for garment measurement workflows.

## Setup

GarmentMeasurements downloads its preprocessed asset from
`https://huggingface.co/datasets/abcamiletto/body-models-assets` on first use. To prefetch and save the path:

```bash
# Download the preprocessed GarmentMeasurements body-model asset.
body-models download garment-measurements
```

## API

::: body_models.bodies.garment_measurements.numpy.GarmentMeasurements
