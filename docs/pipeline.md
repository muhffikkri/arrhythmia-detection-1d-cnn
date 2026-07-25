# End-to-End Experiment Pipeline

This document describes the workflow of the training and evaluation phases used to validate model modifications and research developments in this repository.

---

## 🏃 Pipeline Workflow Overview

The pipeline consists of five key phases, running from raw signal ingest to statistical validation:

```mermaid
graph LR
    Ingest[1. Data Ingest & In-place DSP] --> Manifest[2. Manifest Auditing]
    Manifest --> Train[3. CNN Training & Grid Search]
    Train --> Tune[4. Threshold Optimization]
    Tune --> Stats[5. Statistical Significance Tests]
```

---

## 📁 1. Data Ingest & Preprocessing
Raw files from the PTB-XL and Chapman datasets are preprocessed to the unified **250 Hz** target frequency. The output `.npy` signals and diagnostic target mappings are tracked via metadata CSV manifests:
*   **PTB-XL manifest**: Generated in `dataset/resample/manifest_ptbxl.csv`.
*   **Chapman manifest**: Generated in `dataset/resample/manifest_chapman.csv`.

### Training Resolutions Study
To evaluate the impact of source recording resolution on downstream model performance, the training pipeline isolates two distinct resampling paths:
1.  **500 Hz to 250 Hz path (Downsampled)**: Preserves maximum high-frequency features. Signals are cleaned at 500 Hz, downsampled to 250 Hz, and standardized.
2.  **100 Hz to 250 Hz path (Upsampled)**: Cleaned at 100 Hz, then upsampled to 250 Hz. This represents hardware configurations that operate at lower power/frequency baselines.

---

## 🔍 2. Manifest & Preprocessing Audits
Before training, files are run through quantitative and visual exploratory data analysis (EDA) to ensure signal integrity:
*   **EDA Analysis**: `src/analysis/eda_visualization.py` plots class balances and age/gender demographic breakdowns.
*   **Preprocessing Quality Audits**: `src/analysis/audit_preprocessing.py` evaluates the performance of the denoising filters by calculating Pseudo-SNR metrics and spectral entropy shifts.

---

## 🏋️ 3. CNN Model Training & Grid Search
The model training process is run via `src/experiments/run_experiment_softmax.py` (multiclass Softmax) and `src/experiments/run_experiment_sigmoid.py` (multi-label Sigmoid):

*   **Dataset Partitioning**: PTB-XL splits are handled via stratified folds. Folds 1-8 are dedicated to training, Fold 9 to validation/optimization, and Fold 10 is isolated exclusively as the test set.
*   **Grid Search Parameters**: Fits combinations of filter depths (Small vs. Medium), kernel spaces (Balanced, Large Receptive), dilations, and temporal models (Pure CNN, CNN-BiLSTM, CNN-Attention) as configured in `experiment_configs.py`.
*   **Augmentation Options**: Features on-the-fly Mixup regularization and random physiological noise injection (baseline wander, Gaussian noise, gain scaling).

---

## 🎯 4. Post-Training Tuning & Test Set Isolation
*   **Multi-label Threshold Tuning**: In multi-label setups, predicting class probabilities via Sigmoid activation does not assume a uniform `0.5` threshold. Instead, the model outputs for the validation set (Fold 9) are scanned across threshold intervals `[0.1, 0.9]` to assign optimal decision boundaries per-class that maximize class-specific F1 scores.
*   **Independent Test Evaluation**: The tuned models and thresholds are evaluated against the held-out Test Set (Fold 10) to obtain clean, unbiased performance metrics.
*   **External Validation**: To verify domain robustness, Chapman's dataset acts as an out-of-domain external validation check.

---

## 📈 5. Evaluation, Visualizations & Statistical Significance
All results are compiled and saved to the experiment's directory:
*   **Exported Artifacts**: Generates training history CSV curves, a classification report, model architecture summaries, and confusion matrices.
*   **Error Logging**: Generates medical-grade stacked lead plots showing misclassified waveforms with confidence estimates.
*   **Explainability**: Grad-CAM overlays identify the precise time regions influencing predictions.
*   **Statistical Tests**: Wilcoxon signed-rank tests compare pairwise F1 scores between models to evaluate if modifications yield statistically significant improvements.
