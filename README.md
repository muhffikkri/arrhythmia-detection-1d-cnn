# Arrhythmia Detection using 1D-CNN

A structured, research-grade pipeline for detecting cardiac arrhythmias from 12-lead (and 3-lead) Electrocardiogram (ECG) signals using custom 1D Convolutional Neural Networks (CNNs). The pipeline supports both **multiclass** (using Softmax classification head) and **multi-label** (using Sigmoid classification heads) arrhythmia classification across the PTB-XL and Chapman datasets.

> [!NOTE]
> **250Hz Resampling Specification**: For research consistency and seamless alignment with physical deployments, all training, testing, and external validations in this repository are unified at **250 Hz**. This frequency specifically matches the sampling configuration of the hardware analog front-end (AFE) **ADS1293 sensor**, which has been set to **250 Hz** for real-time acquisition.

---

## 🚀 Key Features

- **Advanced DSP Preprocessing**: Polyphase FIR resampling, Wavelet `db4` adaptive denoising, median filtering for baseline wander removal, and z-score signal clipping.
- **Dynamic 1D-CNN Factory**: High-performance architecture supporting Stochastic Depth (residual branch dropping), Squeeze-and-Excitation (SE) channel attention, and multiple temporal heads (LSTM, Bidirectional LSTM, and Multi-Head Attention).
- **Comprehensive Experiment Tracker**: Auto-generated experiment outputs containing JSON configurations, training CSV logs, model summary cards, confusion matrices, and detailed metrics.
- **Medical-Grade Visualization**: Clinical-style vertical multi-lead plots of misclassified samples alongside raw vs. cleaned ECG morphology steps.
- **Grad-CAM 1D Explainability**: Attention heatmaps overlaid directly onto ECG signals to highlight clinically significant QRS complexes and morphological features.
- **Statistical Significance Analysis**: Automatic generation of Wilcoxon signed-rank significance tests and 95% Confidence Intervals (CI) comparing F1 scores between experimental runs.

---

## 📂 Project Directory Structure

```directory
├── dataset/                    # Dataset storage (excluded from git tracking)
│   ├── Chapman/                # Chapman ECG raw dataset
│   ├── PTBXL/                  # PTB-XL ECG raw dataset
│   └── resample/               # Preprocessed signals (.npy) and dataset manifests
├── docs/                       # Research documentation and figures
│   ├── dataset-label-map.md    # Label mapping guide (mapped/native schemes)
│   └── training-configs/       # Best-config reference cards for each scheme
├── models/                     # Saved model files (.keras / .h5)
├── output/                     # Generated experiment results, logs, and plots
├── standalone/                 # Standalone scripts (copyright / external tools)
├── src/                        # Main source code package
│   ├── analysis/               # EDA and preprocessing quality audits
│   ├── config/                 # Configurations (paths, hyperparameters, label mappings)
│   ├── evaluation/             # Metrics, cross-dataset tests, explainability, statistics
│   ├── experiments/            # Unified training runner entry points
│   ├── models/                 # Model structures and loss functions
│   ├── preprocessing/          # Signal cleaning, filtering, and resampling pipelines
│   ├── training/               # Data loaders, augmentation, mixup, experiment tracker
│   └── __init__.py
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites

Ensure you have Python 3.10+ installed.

### 2. Virtual Environment Setup

Clone the repository and initialize a virtual environment:

```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows Powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 🏃 Pipeline Execution Workflow

### Step 1: Preprocess Datasets

Process raw signals into standardized resampled (.npy) representations. This will automatically create the validation and training manifests under `dataset/resample/`.

```powershell
# Preprocess PTB-XL (Internal train/val/test data)
python src/preprocessing/proccess_ptbxl.py

# Preprocess Chapman (External validation data)
python src/preprocessing/proccess_chapman.py
```

### Step 2: Run Exploratory Data Analysis (EDA)

Run EDA scripts to audit the signal distributions and quality:

```powershell
# Run basic dataset analysis & quality check
python src/analysis/eda_visualization.py

# Run advanced quantitative DSP audit (SNR, spectral entropy, HRV)
python src/analysis/eda_quantitative_audit.py
```

### Step 3: Run Research Experiments

Execute training with the unified runner. All runs are configured in `src/config/experiment_configs.py`:

| Config field | Values | Default | Meaning |
|---|---|---|---|
| `Config.SCHEME` | `"softmax"` \| `"sigmoid"` | `"softmax"` | Classification head (multiclass vs. multi-label) |
| `Config.LABEL_SCHEME` | `"mapped"` \| `"native"` | `"mapped"` | Shared 4-class labels vs. dataset-native labels |
| `Config.TRAIN_DATASET` | `"PTBXL"` \| `"CHAPMAN"` | `"PTBXL"` | Training dataset |
| `Config.TEST_DATASET` | `"PTBXL"` \| `"CHAPMAN"` | `"PTBXL"` | Test/eval dataset |

```powershell
# Multiclass experiment (Softmax head, mapped 4-class labels)
python src/experiments/run_experiment.py

# Multi-label experiment (Sigmoid head)
# -> set Config.SCHEME = "sigmoid" in experiment_configs.py

# Cross-dataset training (e.g. train PTB-XL, test Chapman)
# -> set Config.TEST_DATASET = "CHAPMAN" (LABEL_SCHEME must stay "mapped")
```

Splits are handled centrally by `src/training/dataset_loader.py`: PTB-XL uses its native `strat_fold` (folds 1-8 train / 9 val / 10 test), Chapman uses a stratified 70/15/15 split. Results (with JSON config, classification report, confusion matrix, and metrics) land under `output/research_experiments/`.

### Step 4: Cross-Dataset Evaluation

Evaluate trained models from `models/` against the other dataset's domain (registry-driven, bidirectional):

```powershell
# Default: PTB-XL trained -> Chapman test, and Chapman trained -> PTB-XL test
python src/evaluation/cross_dataset_test.py

# Also include per-dataset sanity runs (PTB-XL->PTB-XL, Chapman->Chapman)
python src/evaluation/cross_dataset_test.py --per-dataset
```

### Step 5: Statistical Significance Evaluation

Compare F1 scores across completed experiment runs to identify statistically significant performance changes:

```powershell
python src/evaluation/run_statistical_tests.py
```

This consumes the unified tracker (`output/research_experiments/master_experiment_tracker.csv`) and produces confidence intervals and Wilcoxon pairwise comparison matrices saved in `output/statistical_tests/`. See [`docs/pipeline.md`](docs/pipeline.md) for the full workflow.

---

## 📊 Core Configurations

All execution paths, label hierarchies, and hyperparameters are managed under the `src/config/` folder:

- [config.py](arrhythmia-detection-1d-cnn/src/config/config.py): Contains base system paths, auto-directory creation setups, target sampling rates, lead indices, and the canonical 4-class order via `TARGET_CLASSES = ["Normal", "AF", "Takikardia", "Bradikardia"]`.
- [config_labels.py](arrhythmia-detection-1d-cnn/src/config/config_labels.py): Consolidates label dictionaries for both datasets (`PTBXL_TO_TARGET_MAPPING` and `CHAPMAN_TO_TARGET_MAPPING`) and re-exports the target class order from `config.py`.
- [experiment_configs.py](arrhythmia-detection-1d-cnn/src/config/experiment_configs.py): Defines the active run (`SCHEME`, `LABEL_SCHEME`, `TRAIN_DATASET`, `TEST_DATASET`) plus grid-search spaces for filters, kernel sizes, dilations, Mixup augmentation strategies, epochs, and learning rates.
- [model_registry.py](arrhythmia-detection-1d-cnn/src/config/model_registry.py): Central registry of trained models for cross-dataset evaluation.

See [`docs/dataset-label-map.md`](docs/dataset-label-map.md) for the label mapping guide and [`docs/training-configs/`](docs/training-configs/) for the best-config reference cards.
