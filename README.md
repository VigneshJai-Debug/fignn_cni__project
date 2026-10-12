# FIGNN: EEG Workload Representation and Frozen-Encoder Calibration

Frequency-Informed Graph Neural Network (FIGNN) for cross-subject EEG workload representation and subject-specific calibration using the COG-BCI dataset.

## Dataset

Download the COG-BCI dataset from [Zenodo](https://zenodo.org/records/7413650) and extract the subject folders into `data/raw/`.

The expected layout is:

```text
fignn_cni__project/
├── data/
│   |── data/
│   |   ├── raw/
│   |   │   ├── sub-01/
│   |   │   │   ├── ses-S1/eeg/
│   |   │   │   ├── ses-S2/eeg/
│   |   │   │   └── ses-S3/eeg/
│   |   │   ├── sub-02/
│   |   │   └── ...
│   |   ├── processed/
│   └── outputs/
├── notebooks/
│   ├── FIGNN_Final_Demo.ipynb
│   ├── FIGNN_Research_Validation_set1.ipynb
│   ├── FIGNN_Research_Validation_set2.ipynb
│   └── FIGNN_Review_2.ipynb
├── results/
│   ├── ablation_summary.csv
│   ├── architecture_ablation_29fold_subject_results.csv
│   ├── loso_all29_subject_metrics.csv
│   ├── stage2_calibration_subject_results.csv
│   └── threshold_summary.csv
├── scripts/
│   ├── ablation.py
│   ├── ablation_train.py
│   ├── baseline.py
│   ├── build_graphs.py
│   ├── full_train.py
│   ├── loso.py
│   ├── make_pre_graph_data.py
│   ├── parallel_ablation.py
│   ├── plv_threshold_sensitivity.py
│   └── stage2.py
├── src/
│   ├── evaluation/
│   │   └── evaluate.py
│   ├── models/
│   │   └── model.py
│   ├── preprocessing/
│   │   ├── graph_builder.py
│   │   ├── plv.py
│   │   └── preprocess.py
│   └── training/
│       └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

Each session's `eeg` directory should contain the EEGLAB files `zeroBACK.set`, `oneBACK.set`, `twoBACK.set`, and `RS_Beg_EC.set`. The subject folders should be directly inside `data/raw/`, without an extra nested subject or dataset directory.

Task labels are:

| File | Label | Task |
|---|---:|---|
| `zeroBACK.set` | 0 | 0-back |
| `oneBACK.set` | 1 | 1-back |
| `twoBACK.set` | 2 | 2-back |
| `RS_Beg_EC.set` | 3 | Rest |

Keep the raw dataset out of version control.

## Pipeline

The preprocessing pipeline reads the EEGLAB recordings, creates two-second epochs, filters the EEG into delta, theta, alpha, and beta bands, and computes phase-locking value (PLV) connectivity.

Each graph represents one epoch and contains 62 EEG-channel nodes. Each node has four relative band-power features. Edges are selected using the mean PLV across the four bands with a threshold of 0.5, and each edge stores the four band-specific PLV values.

The main graph dataset includes the three N-back tasks and rest across 29 participants and three sessions. Stage 1 workload-trajectory evaluation removes rest; Stage 2 uses rest as a subject-specific anchor.

## Setup

Create and activate a Python environment with the dependencies required by the project. Install compatible versions of PyTorch, PyTorch Geometric, MNE, NumPy, pandas, SciPy, and `pytorch-metric-learning`. Use a PyTorch/PyG installation compatible with the machine's CPU or CUDA setup.

Run commands from the repository root after configuring the scripts to use repository-relative paths.

## Build the datasets

Build the main graph dataset:

```bash
python scripts/build_graphs.py
```

Output:

```text
data/processed/all_graphs_29sub.pkl
```

Build the pre-graph cache used by threshold sensitivity:

```bash
python scripts/make_pre_graph_data.py
```

Output:

```text
data/processed/pre_graph_data_reconstructed.pkl
```

The pre-graph cache contains N-back epochs, node features, per-band PLV matrices, and subject/session/epoch metadata. The threshold-sensitivity script must point to the same cache filename.

## Experiments

### Stage 1: Leave-one-subject-out evaluation

```bash
python scripts/loso.py
```

The launcher trains a separate fold for each of the 29 held-out participants. The held-out participant is excluded from encoder training. Stage 1 metrics are computed by the accompanying evaluation notebook or validation script using the saved fold checkpoints.

### Stage 2: Frozen-encoder calibration

Stage 2 uses the fold-specific trained encoder and normalization statistics to represent the held-out participant in the 64-dimensional latent space. The encoder remains frozen during calibration. Rest, 0-back, 1-back, and 2-back centroids are computed; the held-out 1-back centroid is used only to evaluate prediction error.

Run `FIGNN_Review_2.ipynb` after the Stage 1 outputs are available. The notebook compares five methods: Population-only, Midpoint, Translation, Ridge regression, and Frozen geometric. Translation is implemented in this notebook.

### PLV threshold sensitivity

The threshold-sensitivity experiment evaluates thresholds 0.3, 0.4, 0.5, 0.6, and 0.7 using the existing fold-specific checkpoints and training normalization. It rebuilds graph connectivity at each threshold without retraining the encoder. Run the experiment through the corresponding notebook or driver that calls the threshold graph construction and evaluation functions.

### Controlled component ablation

```bash
python scripts/parallel_ablation.py
```

The controlled four-fold experiment uses held-out participants 7, 15, 22, and 29. It compares the full model with variants that remove PLV, remove band weighting, replace the graph-attention encoder with a GCN, replace it with a node-feature MLP, remove supervised contrastive loss, or use band power alone.

### Controlled graph-encoder benchmark

```bash
python scripts/baseline.py
```

The benchmark compares GCN and GAT baselines on held-out participants 7, 15, 22, and 29. By default, the script reuses the existing controlled four-fold FIGNN result rather than retraining FIGNN.

### Architecture sensitivity

```bash
python scripts/ablation.py
```

This experiment compares the earlier 64-hidden-unit, four-head, 32-dimensional configuration with the final 128-hidden-unit, eight-head, 64-dimensional configuration across 29 leave-one-subject-out folds.

### Full-population training

```bash
python scripts/full_train.py
```

This trains a model using all 29 participants. It is intended for a final population model or demonstration, not for reporting unseen-subject evaluation.

## Reproducibility

- Use the same preprocessed graph dataset for the main LOSO evaluation and controlled comparisons.
- Keep held-out participants out of encoder training and model selection.
- Use the training-only normalization statistics saved for each LOSO fold.
- Keep the encoder frozen during Stage 2 calibration.
- For threshold sensitivity, reuse the fold checkpoints and normalization statistics without retraining.
- Report the 29-fold LOSO results separately from the controlled four-fold experiments.
- Record the environment, package versions, random seeds, and hardware used for each final run.

## Citation

FIGNN Manuscript: Access-2026-51817

Hinss, M. F., Jahanpour, E. S., Somon, B., Pluchon, L., Dehais, F., and Roy, R. N. *COG-BCI database: A multi-session and multi-task EEG cognitive dataset for passive brain-computer interfaces*. Zenodo. https://doi.org/10.5281/zenodo.7413650
