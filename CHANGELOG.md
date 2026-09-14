# Changelog

All notable changes to **Arrhythmia Detection using 1D-CNN** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Upload the preprocessed tensors (raw + cleaned, 100/500 Hz) to Kaggle so every scheme can run
  without re-running DSP on the raw data.
- Reproduce the **pure dataset vs. preprocessed dataset** training comparison by toggling only the
  `DATA_SELECTION` cell in `kaggle/train_all_schemes.ipynb`.
- Re-enable the unified **250 Hz** scheme (`GENERATE_SCHEMES["250hz"] = True`) once the 100/500 Hz
  experiments are finalized (hardware-aligned to the ADS1293 AFE).

---

## [1.2.0] - 2026-09-14

### Changed
- **Config-driven cleaning**: preprocessing stages are now toggled via `CLEANING_FLAGS`
  (`src/preprocessing/preprocessing.py`). Active schedule = wavelet `db4` + median baseline + bandpass
  0.5–45 Hz; **z-score normalization disabled** (code kept, stage turned off).
- **Preprocessing scripts** (`proccess_ptbxl.py`, `proccess_chapman.py`) now emit **raw ("murni") and
  cleaned** tensors at **100 Hz and 500 Hz** per dataset; the 250 Hz scheme is deferred
  (`GENERATE_SCHEMES["250hz"] = False`).
- **Folder selection**: single per-dataset selector `Config.FOLDER_PTBXL / Config.FOLDER_CHAPMAN`
  (default `*_clean_500hz`); `INPUT_SHAPE` and sampling rate are derived from the selected folder
  (`config.FOLDER_FS` + `TARGET_LEN`).
- **Kaggle notebook** (`kaggle/train_all_schemes.ipynb`): each of the 12 plans now runs on a single
  user-selectable folder via the `DATA_SELECTION` cell (`500Hz Cleaned`, `500Hz Raw`, `100Hz Cleaned`,
  `100Hz Raw`); readiness/preprocessing cells updated to the 100/500 Hz layout.
- **Docs**: README, `docs/pipeline.md`, `docs/preprocessing.md`, `docs/architecture.md` refreshed for
  the 100/500 Hz scheme; legacy 250 Hz cards in `docs/training-configs/` marked historical.

### Fixed
- `plot_distribution_ptbxl.py`: `KeyError "Nonaritmia"` — target label now maps to the canonical
  `"Normal"` class.
- `eda_visualization.py`: Chapman header parsing reduced from O(N×M) repeated scans to a single walk
  with a dictionary lookup (practically ~4× faster); UTF-8 stdout reconfiguration to avoid cp1252
  crashes on Windows.

### Added
- `eda_waveform_characteristics.py` — QRS-aligned mean-waveform morphology profiling (cross-dataset
  Pearson/DTW similarity of R-aligned leads I–III).

---

## [1.1.0] - 2026-09-13

### Added
- **Data-driven `DatasetLoader`** (`src/training/dataset_loader.py`) supporting `mapped` (shared 4-class:
  `Normal`, `AF`, `Takikardia`, `Bradikardia`) and `native` (dataset codes) label schemes for PTB-XL and
  Chapman; PTB-XL uses official `strat_fold` splits, Chapman uses a stratified 70/15/15 split (seed 42).
- **Unified experiment runner** (`src/experiments/run_experiment.py`) replacing the legacy
  softmax/sigmoid runners — covers `softmax`/`sigmoid` × `mapped`/`native` × cross-dataset pairs.
- **Registry-driven bidirectional cross-dataset evaluation** (`src/evaluation/cross_dataset_test.py`)
  with self-describing class order for legacy models.
- **Statistical tests** now consume the unified master tracker and derive comparisons from data.

### Changed
- `src/config/config.py`: deduplicated `SUB_FOLDERS`, canonical 4-class order centralized in
  `TARGET_CLASSES`; data-driven label-scheme config.
- Documentation refresh for the unified runner/loader/cross-dataset flow (`README.md`,
  `docs/pipeline.md`, `docs/training-configs/`).

---

## [1.0.0] - 2026-07-27

Baseline research release (git tag `v1.0.0`).

### Added
- **Preprocessing DSP engine** (`src/preprocessing/preprocessing.py`): sanitization, 3-lead extraction
  (I/II/III), polyphase FIR resampling (to a unified 250 Hz at the time), wavelet `db4` adaptive
  denoising, median baseline-wander removal, length standardization, z-score + amplitude clipping.
- **Preprocessing scripts** for PTB-XL and Chapman producing `.npy` tensors + metadata manifests under
  `dataset/resample/`.
- **1D-CNN model factory** with residual blocks, Squeeze-and-Excitation, Stochastic Depth, Spatial
  Dropout, and temporal heads (Pure CNN, CNN+BiLSTM, CNN+Multi-Head Attention); multiclass softmax and
  multi-label sigmoid heads; focal loss and binary crossentropy.
- **Experiment runners** for softmax and sigmoid heads with multi-label threshold tuning; per-class
  F1-optimized decision thresholds on validation splits.
- **Class balancing**: undersampling and SMOTE-Tomek oversampling for the mapped scheme.
- **Experiment tracker** (`src/training/tracker.py`) and **data augmentation** (Mixup, physiological
  noise injection).
- **EDA & audits**: dataset distribution plots, preprocessing quality audit (SNR, spectral entropy,
  HRV, morphology preservation).
- **Evaluation suite**: statistical significance tests (Wilcoxon + 95% CI), model confusion-matrix
  report generation, model registry, cross-dataset evaluation baseline.
- **Documentation**: architecture, DSP preprocessing, and end-to-end pipeline specs.