# Arrhythmia Detection using 1D-CNN

A structured, research-grade pipeline for detecting cardiac arrhythmias from 12-lead (and 3-lead) Electrocardiogram (ECG) signals using custom 1D Convolutional Neural Networks (CNNs). The pipeline supports both **multiclass** (using Softmax classification head) and **multi-label** (using Sigmoid classification heads) arrhythmia classification across the PTB-XL and Chapman datasets.

> [!NOTE]
> **Active sampling scheme (100/500 Hz)**: All training, testing, and external validations run on the
> **raw ("murni") and preprocessed** tensor folders at **100 Hz and 500 Hz** (`dataset/resample/`).
> Preprocessing is config-driven via `src/preprocessing/preprocessing.py::CLEANING_FLAGS` (wavelet `db4`
> baseline-wander removal + median baseline + bandpass using the existing 0.5–45 Hz bounds; **z-score
> normalization is currently disabled** — code kept, stage turned off). The unified **250 Hz** scheme
> (hardware-aligned to the ADS1293 AFE) is **deferred to a future experiment**.

---

## 📑 Documentation & Changelog

| Resource                                                         | Description                                                             |
| :--------------------------------------------------------------- | :---------------------------------------------------------------------- |
| [CHANGELOG.md](CHANGELOG.md) | Project changelog (Keep a Changelog format) |
| [docs/pipeline.md](docs/pipeline.md) | End-to-end experiment pipeline (ingest → training → cross-eval → stats) |
| [docs/preprocessing.md](docs/preprocessing.md) | DSP specification & active 100/500 Hz folder scheme |
| [docs/architecture.md](docs/architecture.md) | 1D-CNN architecture reference |
| [docs/research-progress.md](docs/research-progress.md) | Research roadmap, experiment IDs & reproducibility spec |
| [docs/training-env.md](docs/training-env.md) | Environment setup & GPU training on Windows / Linux |
| [docs/dataset-label-map.md](docs/dataset-label-map.md) | Mapped vs. native label schemes |
| [docs/training-configs/](docs/training-configs/) | Best-config cards (legacy 250 Hz baseline) |
| [kaggle/train_all_schemes.ipynb](kaggle/train_all_schemes.ipynb) | Phase 1 & 2 baseline notebook (RAW vs CLEANED, folder-driven) on Kaggle (GPU) |

---

## 🚀 Key Features

- **Advanced DSP Preprocessing**: Polyphase FIR resampling plus config-driven cleaning stages (`CLEANING_FLAGS`): Wavelet `db4` adaptive denoising and median filtering for baseline wander removal, and a bandpass high-frequency filter using the existing 0.5–45 Hz bounds. z-score clipping is available but disabled in the current schedule.
- **Dynamic 1D-CNN Factory**: High-performance architecture supporting Stochastic Depth (residual branch dropping), Squeeze-and-Excitation (SE) channel attention, and multiple temporal heads (LSTM, Bidirectional LSTM, and Multi-Head Attention).
- **Comprehensive Experiment Tracker**: Auto-generated experiment outputs containing JSON configurations, training CSV logs, model summary cards, confusion matrices, and detailed metrics.
- **Reproducible Experiment Records**: every run writes an 18-field `experiment_metadata.json` with an environment snapshot (OS / Python / TF / GPU / CUDA) plus prediction and confidence distributions, appended to a master `experiment_registry.csv` (see [docs/research-progress.md](docs/research-progress.md)).
- **Medical-Grade Visualization**: Clinical-style vertical multi-lead plots of misclassified samples alongside raw vs. cleaned ECG morphology steps.
- **Grad-CAM 1D Explainability**: Attention heatmaps overlaid directly onto ECG signals to highlight clinically significant QRS complexes and morphological features.
- **Statistical Significance Analysis**: Automatic generation of Wilcoxon signed-rank significance tests and 95% Confidence Intervals (CI) comparing F1 scores between experimental runs.

---

## 📂 Project Directory Structure

```directory
├── dataset/                    # Dataset storage (excluded from git tracking)
│   ├── Chapman/                # Chapman ECG raw dataset
│   ├── PTBXL/                  # PTB-XL ECG raw dataset
│   └── resample/               # Preprocessed signals (.npy) + manifests
│       ├── ptbxl_{raw,clean}_{100hz,500hz}/
│       └── chapman_{raw,clean}_{100hz,500hz}/
├── docs/                       # Research documentation and figures
│   ├── architecture.md         # 1D-CNN model architecture reference
│   ├── pipeline.md             # End-to-end experiment pipeline
│   ├── preprocessing.md        # DSP preprocessing specification (100/500 Hz scheme)
│   ├── dataset-label-map.md    # Label mapping guide (mapped/native schemes)
│   └── training-configs/       # Best-config reference cards (legacy 250 Hz baseline)
├── kaggle/                     # Kaggle notebook for the full scheme matrix
│   └── train_all_schemes.ipynb # Single notebook (DATA_SELECTION -> 100/500 Hz folder)
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
├── CHANGELOG.md                # Project changelog
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## ⚙️ Installation & Setup

The repo runs identically on **Windows (PowerShell)** and **Linux (bash)**. Full detail:
[`docs/training-env.md`](docs/training-env.md).

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> `pip tensorflow` on native Windows is **CPU-only**. For **GPU** speed, train inside
> **WSL2 + Ubuntu** (see `docs/training-env.md`) — the code needs no changes.

### Linux (bash) — recommended for GPU

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Verify GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

`run_experiment.py` auto-detects CPU/GPU, prints the device banner, and enables GPU
memory growth — no flags required.

---

## 🏃 Pipeline Execution Workflow

### Step 1: Preprocess Datasets

Process raw signals into resampled (`.npy`) tensors. Both scripts write six folders per dataset
(**raw** and **cleaned** at **100 Hz** and **500 Hz**) plus a manifests CSV under `dataset/resample/`.
Cleaning is **config-driven** via `CLEANING_FLAGS` in `src/preprocessing/preprocessing.py`
(current schedule: wavelet `db4` + median baseline + bandpass 0.5–45 Hz; z-score clip is **off**).

```powershell
# Preprocess PTB-XL (Internal train/val/test data)
python src/preprocessing/proccess_ptbxl.py

# Preprocess Chapman (External validation data)
python src/preprocessing/proccess_chapman.py
```

The active folder per dataset is selected by `Config.FOLDER_PTBXL` / `Config.FOLDER_CHAPMAN`
in `src/config/experiment_configs.py` (e.g. `ptbxl_clean_500hz`, `chapman_raw_100hz`). Input shape
and sampling rate are derived from the selected folder automatically.

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

| Config field            | Values                     | Default                 | Meaning                                          |
| ----------------------- | -------------------------- | ----------------------- | ------------------------------------------------ |
| `Config.SCHEME`         | `"softmax"` \| `"sigmoid"` | `"softmax"`             | Classification head (multiclass vs. multi-label) |
| `Config.LABEL_SCHEME`   | `"mapped"` \| `"native"`   | `"mapped"`              | Shared 4-class labels vs. dataset-native labels  |
| `Config.TRAIN_DATASET`  | `"PTBXL"` \| `"CHAPMAN"`   | `"PTBXL"`               | Training dataset                                 |
| `Config.TEST_DATASET`   | `"PTBXL"` \| `"CHAPMAN"`   | `"PTBXL"`               | Test/eval dataset                                |
| `Config.FOLDER_PTBXL`   | any `SUB_FOLDERS` key      | `"ptbxl_clean_500hz"`   | Active PTB-XL folder (raw/clean × 100/500 Hz)    |
| `Config.FOLDER_CHAPMAN` | any `SUB_FOLDERS` key      | `"chapman_clean_500hz"` | Active Chapman folder (raw/clean × 100/500 Hz)   |

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

### Step 6: Run the Full Scheme Matrix on Kaggle

[kaggle/train_all_schemes.ipynb](kaggle/train_all_schemes.ipynb) trains **all 12 plans** of the
scheme matrix end-to-end and packs the results history, confusion matrices, cross-dataset evaluation,
and statistical tests into one ZIP:

| Head    | Label scheme | Valid dataset pairs                                        |
| ------- | ------------ | ---------------------------------------------------------- |
| softmax | mapped       | PTBXL→PTBXL, PTBXL→CHAPMAN, CHAPMAN→PTBXL, CHAPMAN→CHAPMAN |
| softmax | native       | PTBXL→PTBXL, CHAPMAN→CHAPMAN                               |
| sigmoid | mapped       | PTBXL→PTBXL, PTBXL→CHAPMAN, CHAPMAN→PTBXL, CHAPMAN→CHAPMAN |
| sigmoid | native       | PTBXL→PTBXL, CHAPMAN→CHAPMAN                               |

A single **`DATA_SELECTION`** cell switches the folder used by all plans
(`500Hz Cleaned` / `500Hz Raw` / `100Hz Cleaned` / `100Hz Raw`), so the "pure dataset vs. preprocessed
dataset" experiment can be reproduced by changing just that one variable. Set `RUN_EDA = False` and
`FAST_MODE = True` for a quick end-to-end smoke test.

---

## 🧪 Current Research Phases

All experimentation uses **3 leads** (I, II, III). The phases map 1-to-1 onto the
existing configuration knobs — no code changes required:

| Phase | Description | Config (experiment_configs.py) |
| :---- | :---------- | :------------------------------ |
| 1. Baseline (raw, all classes) | Train PTB-XL & Chapman separately on the raw tensors with `LABEL_SCHEME="native"` (all available classes) | `FOLDER_*=*_raw_500hz`, `LABEL_SCHEME="native"`, `TRAIN=TEST` |
| 2. Cleaned comparison | Same, but on cleaned tensors (`wavelet db4 + median baseline + bandpass`, z-score off) | `FOLDER_*=*_clean_500hz`, `LABEL_SCHEME="native"`, `TRAIN=TEST` |
| 3. In-domain + cross-dataset (mapped) | E1 PTBXL→PTBXL, E2 CHAPMAN→CHAPMAN, E3 PTBXL→CHAPMAN, E4 CHAPMAN→PTBXL on cleaned folders | `FOLDER_*=*_clean_500hz`, `LABEL_SCHEME="mapped"`, `TRAIN`/`TEST` per experiment |
| 4. Clinically-detectable subset | Retrain on cleaned with only the classes reliably detectable with 3 leads | class subset filtering (planned) |
| 5. Verification & derived leads | Rule-based verification for FP/FN, plus aVR/aVL/aVF as auxiliary input | planned |

Research questions (from the plan): Is the failure caused by **domain shift**? Is the
**3-lead** configuration robust? Do **derived leads** help? Does the same preprocessing fit both
datasets? Can **rule-based verification** correct specific false positives/negatives? Is the model
learning **physiological vs. dataset-specific** features?

> The Kaggle notebook runs every phase-1/2/3 combination via `DATA_SELECTION` (raw/clean × 100/500 Hz)
> + the scheme matrix; currently `LABEL_SCHEME="mapped"` is required for cross-dataset plans.

---

## 📊 Core Configurations

All execution paths, label hierarchies, and hyperparameters are managed under the `src/config/` folder:

- [config.py](src/config/config.py): Contains base system paths, auto-directory creation setups, target sampling rates, lead indices, the folder map (`SUB_FOLDERS`, `FOLDER_FS`), and the canonical 4-class order via `TARGET_CLASSES = ["Normal", "AF", "Takikardia", "Bradikardia"]`.
- [config_labels.py](src/config/config_labels.py): Consolidates label dictionaries for both datasets (`PTBXL_TO_TARGET_MAPPING` and `CHAPMAN_TO_TARGET_MAPPING`) and re-exports the target class order from `config.py`.
- [experiment_configs.py](src/config/experiment_configs.py): Defines the active run (`SCHEME`, `LABEL_SCHEME`, `TRAIN_DATASET`, `TEST_DATASET`, `FOLDER_PTBXL`, `FOLDER_CHAPMAN`) plus grid-search spaces for filters, kernel sizes, dilations, Mixup augmentation strategies, epochs, and learning rates.
- [model_registry.py](src/config/model_registry.py): Central registry of trained models for cross-dataset evaluation.

See [`docs/dataset-label-map.md`](docs/dataset-label-map.md) for the label mapping guide and [`docs/training-configs/`](docs/training-configs/) for the best-config reference cards.
