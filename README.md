# PhaseConeGNN

[中文说明](README_zh.md)

PhaseConeGNN is a phase-aware logic-cone graph neural network for And-Inverter
Graph (AIG) representation learning. The implementation supports signal
probability prediction, equivalent gate identification, and truth-table-distance
based representation learning on signed AIG graphs.

This repository is prepared as an anonymous research artifact. It intentionally
does not include author identities, institution names, or non-anonymous project
links.

## Overview

The model takes a signed AIG as input and learns node-level embeddings together
with signal probability predictions. Signed edges are used to distinguish
phase-preserving and phase-inverting logic propagation. The main implementation
contains:

- phase-state initialization for original and inverse node phases;
- signed phase-transition message passing over positive and complemented edges;
- graph-context-conditioned normalization and gated residual propagation;
- adaptive fusion over representations from multiple propagation depths;
- logic-aware signal probability readout with a signed-fanin prior.

The learned node embeddings can be used for equivalent gate identification by
computing the cosine similarity between gate-pair embeddings.

## Repository Structure

```text
PhaseConeGNN/
  README.md            # Usage and reproduction instructions
  environment.yml      # Conda environment specification
  data/
    demo_AIGDataset/   # Tiny processed dataset for smoke testing
  train.py             # Training, validation, checkpointing, and testing
  model.py             # PhaseConeGNN model definition
  layers.py            # Phase transition, residual, fusion, and readout layers
  normalization.py     # Graph-context-conditioned normalization
  load_data.py         # Signed AIG dataset loader
  utils.py             # Edge splitting, graph context, and utility functions
  __init__.py          # Package export
```

## Environment

The code is implemented in Python with PyTorch. We recommend using Conda to
create an isolated environment:

```bash
conda env create -f environment.yml
conda activate phaseconegnn
```

If you prefer to create the environment manually:

```bash
conda create -n phaseconegnn python=3.10 -y
conda activate phaseconegnn
conda install pytorch numpy tqdm -c pytorch -c conda-forge -y
```

For CUDA-enabled training, install the PyTorch build that matches your local
CUDA driver. For example:

```bash
conda install pytorch pytorch-cuda=12.1 numpy tqdm -c pytorch -c nvidia -c conda-forge -y
```

Optional dependencies are only needed when using spectral input features:

```bash
conda install scipy scikit-learn -c conda-forge -y
```

## Dataset Access

The training script uses processed signed-AIG samples. The full processed
dataset is distributed separately because it contains large `.npz` files and
many per-graph CSV files:

[Noah0519/AIGdataset](https://huggingface.co/datasets/Noah0519/AIGdataset)

After downloading the full processed dataset, pass its root directory with
`--data_root_path`. If it is placed under the repository root, use:

```bash
--data_root_path ./AIGDataset/ForgeEDA_proxy_processed
```

## Demo Processed Dataset

This repository includes a tiny processed dataset for checking that the data
loader and training entry point work end to end:

```text
data/demo_AIGDataset/ForgeEDA_proxy_demo/
```

The demo contains a few small graph folders and a `demo` split. It is only for
smoke testing and is not intended to reproduce the full experimental results.
Run this quick check from the `PhaseConeGNN` directory:

```bash
python train.py \
  --task_type prob \
  --data_root_path ./data/demo_AIGDataset/ForgeEDA_proxy_demo \
  --split_file demo \
  --feature_type one-hot \
  --in_dim 3 \
  --out_dim 32 \
  --layer_num 2 \
  --batch_size 2 \
  --device_backend cpu \
  --epochs 1 \
  --name_others demo_smoke
```

For full experiments, use the full processed dataset from Hugging Face and pass
its root directory with `--data_root_path`.

## Processed Dataset Layout

The loader expects a processed signed-AIG dataset with the following structure:

```text
AIGDataset/ForgeEDA_proxy_processed/
  split/
    0.05-0.05-0.9__full/
      train.txt
      valid.txt
      test.txt
  npz/
    labels.npz
  <graph_name>/
    raw/
      signed_edge.csv
      node-feat.csv
      prob.csv
```

Each split file lists graph folders relative to the dataset root or as absolute
paths. Each graph folder must contain:

- `raw/signed_edge.csv`: signed directed edges in `(src, dst, sign)` format,
  where positive signs denote phase-preserving edges and negative signs denote
  complemented edges;
- `raw/node-feat.csv`: node-level input features;
- `raw/prob.csv`: node-level signal probability labels.

For equivalent gate identification and truth-table distance learning,
`npz/labels.npz` stores gate-pair labels. The loader reads `tt_pair_index` and
`tt_dis`, and converts truth-table distance to equivalent-gate similarity as
`eq_sim = 1 - tt_dis`.

## Running Experiments

Run commands from the `PhaseConeGNN` directory.

### Signal Probability Prediction

```bash
python train.py \
  --task_type prob \
  --data_root_path ./AIGDataset/ForgeEDA_proxy_processed \
  --split_file 0.05-0.05-0.9__full \
  --feature_type one-hot \
  --in_dim 3 \
  --out_dim 256 \
  --layer_num 5 \
  --batch_size 128 \
  --device_backend cuda \
  --device 0 \
  --epochs 40 \
  --name_others forgeeda_proxy_full
```

### Equivalent Gate Identification

```bash
python train.py \
  --task_type eq \
  --data_root_path ./AIGDataset/ForgeEDA_proxy_processed \
  --split_file 0.05-0.05-0.9__full \
  --feature_type one-hot \
  --in_dim 3 \
  --out_dim 256 \
  --layer_num 5 \
  --batch_size 128 \
  --device_backend cuda \
  --device 0 \
  --epochs 40 \
  --name_others forgeeda_proxy_full
```

### Truth-Table Distance Learning

```bash
python train.py \
  --task_type tt \
  --data_root_path ./AIGDataset/ForgeEDA_proxy_processed \
  --split_file 0.05-0.05-0.9__full \
  --feature_type one-hot \
  --in_dim 3 \
  --out_dim 256 \
  --layer_num 5 \
  --batch_size 128 \
  --device_backend cuda \
  --device 0 \
  --epochs 40 \
  --name_others forgeeda_proxy_full
```

Use CPU-only execution by replacing the device arguments with:

```bash
--device_backend cpu --device -1
```

## Main Arguments

- `--task_type`: task to train, selected from `prob`, `eq`, and `tt`.
- `--data_root_path`: processed dataset root.
- `--split_file`: split folder name under `split/`.
- `--feature_type`: input feature type, selected from `one-hot`, `raw`, and
  `spectral`.
- `--in_dim`: input feature dimension. For `one-hot`, this is the number of
  gate types.
- `--out_dim`: node embedding dimension. It must be even because the model
  maintains original-phase and inverse-phase states.
- `--layer_num`: number of propagation layers.
- `--use_backward`: `-1` uses the default task-dependent setting, `0` disables
  backward fanout propagation, and `1` enables it.
- `--use_layer_moe`: whether to enable adaptive multi-depth representation
  fusion.
- `--lambda_not`, `--lambda_and`: weights for the logic consistency losses used
  in signal probability prediction.

## Outputs

Training creates the following folders under the current working directory:

```text
ft_saved/   # best model checkpoints
results/    # experiment logs and metric summaries
```

The script reports validation and test metrics for the selected task. For
equivalent gate identification, gate-pair similarity is computed as the cosine
similarity between the learned embeddings of the two gates.
