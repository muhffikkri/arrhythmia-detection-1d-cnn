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
├── models/                     # Saved model files (.keras / .h5)
├── output/                     # Generated experiment results, logs, and plots
├── src/                        # Main source code package
│   ├── analysis/               # EDA and preprocessing quality audits
│   ├── config/                 # Configurations (paths, hyperparameters, and label mappings)
│   ├── evaluation/             # Metrics, explainability (Grad-CAM), and statistical tests
│   ├── experiments/            # Multiclass and multi-label run entry points
│   ├── models/                 # Model structures and loss functions
│   ├── preprocessing/          # Signal cleaning, filtering, and resampling pipelines
│   ├── training/               # Data loaders, augmentation, mixup, and experiment tracker
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

# Run advanced quantitative DSP audit (SNR, spectral entropy)
python src/analysis/audit_preprocessing.py
```

### Step 3: Run Research Experiments

Execute training scripts for either classification scheme. The hyperparameters can be configured dynamically in `src/config/experiment_configs.py`.

```powershell
# Run multiclass experiment (Softmax classification head)
python src/experiments/run_experiment_softmax.py

# Run multi-label experiment (Sigmoid classification head)
python src/experiments/run_experiment_sigmoid.py
```

### Step 4: Statistical Significance Evaluation

Compare F1 scores across completed experiment phases to identify statistically significant performance changes:

```powershell
python src/evaluation/run_statistical_tests.py
```

This produces confidence intervals and Wilcoxon pairwise comparison matrices saved in `output/statistical_tests/`.

---

## 📊 Core Configurations

All execution paths, label hierarchies, and hyperparameters are managed under the `src/config/` folder:

- [config.py](arrhythmia-detection-1d-cnn/src/config/config.py): Contains base system paths, auto-directory creation setups, target sampling rates, and lead indices.
- [config_labels.py](arrhythmia-detection-1d-cnn/src/config/config_labels.py): Consolidates label dictionaries for both datasets (`PTBXL_TO_TARGET_MAPPING` and `CHAPMAN_TO_TARGET_MAPPING`) along with active validation target labels (`["Normal", "AF", "Takikardia", "Bradikardia"]`).
- [experiment_configs.py](arrhythmia-detection-1d-cnn/src/config/experiment_configs.py): Defines grid-search spaces for filters, kernel sizes, dilations, Mixup augmentation strategies, epochs, and learning rates.
