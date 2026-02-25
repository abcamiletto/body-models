# Benchmark Results

- **Runs per measurement**: 20 (outliers removed via IQR)
- **Warmup runs**: 5
- **PyTorch version**: 2.10.0+cu126
- **CUDA available**: True

## CPU

### `forward_skeleton` (ms)

| Model | B=256 | B=512 | B=1024 | B=2048 | B=4096 |
|---|---:|---:|---:|---:|---:|
| SMPL | 2.41 | 3.37 | 4.43 | 6.17 | 13.00 |
| SMPLX | 3.11 | 4.16 | 5.56 | 10.60 | 19.84 |

### `forward_vertices` (ms)

| Model | B=64 | B=128 | B=256 | B=512 |
|---|---:|---:|---:|---:|
| SMPL | 10.50 | 18.34 | 44.45 | 119.59 |
| SMPLX | 16.67 | 46.26 | 79.63 | 192.34 |

## CPU (torch.compile)

### `forward_skeleton` (ms)

| Model | B=256 | B=512 | B=1024 | B=2048 | B=4096 |
|---|---:|---:|---:|---:|---:|
| SMPL | 1.23 | 1.30 | 1.81 | 2.55 | 6.51 |
| SMPLX | 1.58 | 2.18 | 4.18 | 8.15 | 18.00 |

### `forward_vertices` (ms)

| Model | B=64 | B=128 | B=256 | B=512 |
|---|---:|---:|---:|---:|
| SMPL | 6.27 | 9.48 | 23.89 | 65.82 |
| SMPLX | 11.34 | 27.78 | 54.73 | 136.25 |

## CUDA

### `forward_skeleton` (ms)

| Model | B=256 | B=512 | B=1024 | B=2048 | B=4096 |
|---|---:|---:|---:|---:|---:|
| SMPL | 1.65 | 1.48 | 1.47 | 1.55 | 1.61 |
| SMPLX | 1.75 | 1.76 | 1.75 | 1.81 | 2.37 |

### `forward_vertices` (ms)

| Model | B=64 | B=128 | B=256 | B=512 |
|---|---:|---:|---:|---:|
| SMPL | 2.27 | 2.92 | 4.21 | 6.64 |
| SMPLX | 2.88 | 3.93 | 6.00 | 10.28 |

## CUDA (torch.compile)

### `forward_skeleton` (ms)

| Model | B=256 | B=512 | B=1024 | B=2048 | B=4096 |
|---|---:|---:|---:|---:|---:|
| SMPL | 0.64 | 0.65 | 0.64 | 0.65 | 0.81 |
| SMPLX | 0.95 | 0.94 | 0.95 | 0.96 | 1.52 |

### `forward_vertices` (ms)

| Model | B=64 | B=128 | B=256 | B=512 |
|---|---:|---:|---:|---:|
| SMPL | 1.38 | 2.08 | 3.46 | 5.79 |
| SMPLX | 2.04 | 3.04 | 4.86 | 8.77 |
