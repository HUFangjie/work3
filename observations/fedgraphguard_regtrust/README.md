# FedGraphGuard / ReG-Trust Observation-1 (Two-stage)

## Scope
This directory is self-contained for Observation-1.
- Stage-1: `export_real_logits.py` (real benign logits export)
- Stage-2: `observation1_lowrank.py` (analysis + plotting only; no training)

## Data
Default data root: `analysis/data`.
Supported datasets:
- `cifar10` (expects `analysis/data/cifar-10-batches-py/`)
- `mnist` (expects `analysis/data/MNIST/`)
- `tinyimagenet` (expects `analysis/data/tiny-imagenet-200/`)

No automatic download is used. Missing data raises clear error.

## Stage-1: export logits
```bash
python -m observations.fedgraphguard_regtrust.export_real_logits \
  --dataset cifar10 \
  --data-root analysis/data \
  --out-dir observations/real_inputs/obs1 \
  --num-clients 20 \
  --num-public 1000 \
  --betas 0.1 0.3 0.5 1.0 \
  --model-modes homogeneous heterogeneous \
  --seeds 0 1 2 \
  --device cuda
```

Outputs (unique run id, never fixed filename):
- `<run_id>_logits.npz` with keys:
  - `dataset_cifar10_mode_homogeneous_beta_0.1_seed_0` ...
- `<run_id>_metadata.json`

Value shape for every key: `(K, N_pub, C)`.

## Stage-2: analyze + plot
```bash
python -m observations.fedgraphguard_regtrust.observation1_lowrank \
  --logits-npz observations/real_inputs/obs1/<generated_logits_file>.npz \
  --out-dir observations/outputs/obs1 \
  --dataset cifar10 \
  --kappa 5
```

This script only reads logits and computes:
- Top-k Jaccard similarity matrix
- SVD + cumulative spectral energy
- effective rank, r90, r95
- CIFAR-10 2x2 main figure (beta = 0.1/0.3/0.5/1.0, homogeneous vs heterogeneous)

Outputs (unique run id):
- `<run_id>_metrics.csv`
- `<run_id>_metrics.json`
- `<run_id>_cumulative_energy.png`
- `<run_id>_cumulative_energy.pdf`

## Notes
- CIFAR-10 is the main-text figure.
- MNIST and Tiny-ImageNet can be exported/analyzed with same commands for appendix figures.
- Dirichlet parameter is named **beta** consistently.
