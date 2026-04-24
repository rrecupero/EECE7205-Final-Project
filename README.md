# Physics-Informed Linear Transformer for Robot Trajectory Prediction

**EECE 7205 — High Performance Computing Final Project**
Northeastern University | Spring 2026

## Overview

This project compares four neural network architectures for predicting robot trajectories using the [NCLT dataset](http://robots.engin.umich.edu/nclt/) (University of Michigan North Campus Long-Term Vision and Lidar Dataset). Given 20 observed timesteps (2 seconds at 10 Hz), each model predicts the next 10 timesteps (1 second) of position and heading.

The key contribution is a **physics-informed linear transformer** that predicts velocity commands (v, ω) instead of positions directly, then integrates them through a unicycle kinematic model. This guarantees physically plausible trajectories while achieving significantly lower prediction error.

## Models

| Model | Description |
|-------|-------------|
| **LSTM** | 2-layer LSTM baseline (hidden_dim=64) |
| **Transformer** | Standard transformer encoder with softmax attention |
| **Linear Transformer** | Transformer with O(n) linear attention (Katharopoulos et al., 2020) |
| **Physics Linear Transformer** | Linear transformer + unicycle kinematic integrator + physics-informed loss |

## Results

| Model | ADE (m) | FDE (m) | Inference (ms) | Parameters |
|-------|---------|---------|----------------|------------|
| LSTM | 6.81 | 6.81 | — | — |
| Transformer | 8.34 | 8.07 | 0.75 | — |
| Linear Transformer | 2.93 | 3.09 | 2.11 | 102,942 |
| Physics Linear Transformer | **0.085** | **0.156** | 3.23 | 102,292 |

## Repository Structure

```
├── preprocessor.py                      # NCLT data preprocessing and normalization
├── dataexplorer.py                      # Dataset exploration and visualization
├── linear_transformer.py                # Linear attention transformer architecture
├── physics_linear_transformer.py        # Physics-informed variant with unicycle model
├── train_lstm.py                        # LSTM training script
├── train_transformer.py                 # Standard transformer training script
├── train_linear_transformer.py          # Linear transformer training script
├── train_physics_linear_transformer.py  # Physics-informed model training script
├── benchmark_scaling.py                 # Inference latency vs sequence length benchmark
└── README.md
```

## Usage

### Prerequisites

```
torch >= 2.0
numpy
matplotlib
```

### Data Preprocessing

```bash
python preprocessor.py
```

This reads raw NCLT data, creates sliding windows (20 observe / 10 predict), applies z-score normalization, and saves train/val/test splits as `.pt` files.

### Training

Run all models via SLURM:

```bash
sbatch compare_models.slurm
```

Or individually:

```bash
python train_lstm.py
python train_transformer.py
python train_linear_transformer.py
python train_physics_linear_transformer.py
```

All scripts auto-detect CUDA and save results (model checkpoints, metrics JSON, training curves, sample prediction plots) to `Results/`.

### Inference Scaling Benchmark

```bash
python benchmark_scaling.py
```

Measures inference latency across sequence lengths to demonstrate O(n) vs O(n²) scaling behavior of linear vs standard attention.

## Key Ideas

**Linear Attention**: Replaces softmax attention with a kernel feature map φ(x) = elu(x) + 1, enabling the associativity trick: instead of computing the T×T attention matrix first (O(T²)), we compute a d×d summary matrix first (O(T·d²)), making attention linear in sequence length.

**Physics-Informed Output**: Instead of directly regressing (x, y, θ) positions, the model predicts (v, ω) velocity commands and integrates them through the unicycle kinematic model. This structurally prevents physically impossible trajectories (no lateral sliding, smooth motion) and reduces the effective search space.

**Physics Loss**: Combines trajectory MSE with soft velocity bound penalties and smoothness regularization to encourage realistic acceleration profiles.

## References

- Katharopoulos et al., "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" (ICML 2020)
- Carlevaris-Bianco et al., "University of Michigan North Campus Long-Term Vision and Lidar Dataset" (IJRR 2016)
