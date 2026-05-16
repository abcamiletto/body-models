# GarmentMeasurements

GarmentMeasurements is a PCA body model with an FBX-derived skeleton and skinning, intended for garment measurement workflows.

## Setup

GarmentMeasurements downloads its preprocessed asset from
`https://huggingface.co/datasets/abcamiletto/body-models-assets` on first use. To prefetch and save the path:

```bash
body-models download garment-measurements
```

## API

::: body_models.garment_measurements.numpy.GarmentMeasurements
