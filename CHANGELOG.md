# Changelog

All notable changes to **Arrhythmia Detection using 1D-CNN** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.6.0] - 2026-09-19

### Added
- **Self-contained Phase 1 & 2 training notebook** (`kaggle/train_phase1-2.ipynb`), replacing
  `kaggle/train_all_schemes.ipynb`: no `src/` dependency. Auto-detects the dataset root under
  `/kaggle/input`, trains one **softmax + native (all-class)** model per dataset × active folder, and
  saves `best_model.keras` + `metrics.csv` + `classification_report.txt` + `training_history.csv` +
  `predictions.npz` + `experiment_metadata.json` per run. Folder picking stays copy-paste driven via
  `DATA_SELECTION` (`500Hz`/`100Hz`) and `RUN_PHASES` (`Phase1_Raw`/`Phase2_Cleaned`); empty folders
  are skipped. A smoke harness (`FAST_MODE=True`) exercises the full path
  (data → training → reload → confusion/misclassified plots → zip).
- **Inline Kaggle evaluation**: right after each run the notebook **saves and renders** the
  classification report, the confusion matrix (counts + normalized) and the first misclassified
  ECG samples (3-lead clinical panels) under each `<experiment_id>/` folder, so results can be
  assessed immediately on Kaggle.

### Changed
- **Stress test and (macro/per-class) AUROC removed from the pipeline**:
  `src/evaluation/evaluator.py` loses `evaluate_stress_test()` and the AUROC metric blocks;
  `src/experiments/run_experiment.py` no longer writes `AUROC_*` columns; and
  `src/evaluation/run_statistical_tests.py` guards the `Macro_AUROC` column (NaN when absent).
- **README**: Step 6 rewritten (training-only notebook + local post-analysis for statistical tests /
  Grad-CAM / cross-dataset evaluation, Keras 3 `compile=False` + `custom_objects` note); docs table
  and directory tree updated.
- **Portable manifests**: the `path_*` columns in `manifest_ptbxl.csv` / `manifest_chapman.csv` now
  store paths **relative to the project root** (`config.BASE_DIR`) via `config.to_relative_path()`,
  so `dataset/resample` can be uploaded to Kaggle and mounted elsewhere without rewriting the CSVs;
  `config.resolve_path()` reads them back (EDA resolves automatically). The deferred **250 Hz**
  resampling scheme is retained behind `GENERATE_SCHEMES["250hz"] = False`.
- **Config-driven native class selection added**: `USE_NATIVE_ALL_CLASSES=True` (default) makes
  `LABEL_SCHEME="native"` use **every** class in the manifest (3 leads); `NATIVE_CLASS_SELECTION`
  (`all`/`allowlist`/`map`), `NATIVE_CLASS_ALLOWLIST`, `NATIVE_CLASS_MAPPING` and `MIN_CLASS_COUNT`
  are the future hooks to restrict/remap classes from config alone. `DatasetLoader.resolve_label`
  applies the selection; `eda_dataset_profiles.py` reuses it so EDA shows the **full** class
  distribution under native (per-folder figures + `class_distribution_by_folder.csv`).
- **Preprocessing no longer filters by the 4-class mapping**: `proccess_ptbxl.py` / `proccess_chapman.py`
  now store **every** record in the 100/500 Hz folders. `native_label` (original SCP / SNOMED code) is
  kept for `LABEL_SCHEME="native"` (no mapping applied), and the derived `target_class` remains only
  for the legacy / cross-dataset `mapped` scheme. `PIPELINE_VERSION` bumped to `v6.1_100_500hz`.
  Docs updated (`docs/training-configs/label-schemes.md`).

---

## [Unreleased]

### Planned
- Run the research phases: (3) in-domain & direct cross-dataset experiments E1–E4 on cleaned tensors,
  (4) re-train on the subset of classes that are clinically detectable with 3 leads,
  (5) rule-based verification for false positives/negatives and derived leads (aVR/aVL/aVF) as auxiliary input.
- Upload the **raw** and **cleaned** tensor datasets as two separate Kaggle datasets; run the notebook
  "Save Version" against the relevant dataset URL.
- Re-enable the unified **250 Hz** scheme (`GENERATE_SCHEMES["250hz"] = True`) once the 100/500 Hz
  experiments are finalized (hardware-aligned to the ADS1293 AFE).

---

## [1.5.0] - 2026-09-17

### Added
- **Phase 1 & 2 baseline notebook** (`kaggle/train_all_schemes.ipynb`): the notebook is now the
  folder-driven **Phase 1 (RAW baselines)** + **Phase 2 (CLEANED comparison)** runner. Picking a folder
  (`DATA_SELECTION` = `500Hz`/`100Hz`, `RUN_PHASES` toggles) remaps every phase; empty folders are
  auto-skipped, so attaching only the cleaned dataset still runs Phase 2 and only raw runs Phase 1.
- **RAW vs CLEANED phase comparison**: a new post-processing cell writes `phase_comparison.csv` + PNG
  (best Macro-F1 / balanced accuracy per dataset and sampling rate, with `Delta_Macro_F1`).

### Changed
- **Notebook plans**: narrowed to the softmax + native (all-class) in-domain baselines for Phases 1–2;
  sigmoid and cross-dataset plans remain gated for later phases. Per-phase folder selection replaces the
  old copy-paste scheme matrix.
- **README** and `docs/research-progress.md`: Phase 1–2 rows marked *notebook ready*.

---

## [1.4.0] - 2026-09-16

### Added
- **Experiment metadata infrastructure** (`src/training/experiment_metadata.py`): `get_environment_metadata()`
  (OS / Python / TF / CUDA / GPU snapshot), `infer_experiment_id()` / `build_experiment_id()`,
  `build_experiment_record()`, `write_experiment_metadata()`, `append_experiment_registry()`,
  `save_prediction_distributions()`.
- **Experiment records in the runner**: `run_experiment.py` now emits `experiment_metadata.json`
  (18-field schema + environment) per run, writes per-class `prediction_distribution.csv` and
  `confidence_distribution.csv`, and appends a flattened row to
  `output/research_experiments/experiment_registry.csv`.
- **Research roadmap document** (`docs/research-progress.md`): Phase 0–7 status table, experiment-ID
  spec, record schema, reproducibility contract, experiment flow diagram, versioning policy.

### Changed
- **Config**: new `EXPERIMENT_ID`, `MODEL_VERSION`, `EXPERIMENT_NOTE`, `EXPERIMENT_INTERPRETATION`,
  and `THRESHOLD_TUNING` fields. Experiment ID is auto-inferred from dataset + folder when unset.
- **Multi-label threshold tuning gated off**: `THRESHOLD_TUNING=False` (sigmoid track now uses a flat
  0.5 threshold); the tuning code path is preserved and re-enabled by the flag. Research focuses on the
  softmax multiclass track.

---

## [1.3.0] - 2026-09-16

### Added
- **Cross-platform training support** (`docs/training-env.md`): full setup for native Windows
  (CPU), WSL2/Ubuntu (GPU), and Linux (GPU). Since TF 2.11 the pip `tensorflow` wheel is
  self-contained on Linux; native Windows remains CPU-only.
- **GPU auto-detection in the runner**: `run_experiment.py` prints an `[Env]` device banner
  (OS / TF version / GPU list) and enables `set_memory_growth` on every GPU — single command works
  on both Windows and Linux.
- **Self-contained native labels**: the preprocessing scripts now embed a `native_label` column in
  `manifest_{ptbxl,chapman}.csv` (primary SCP code / primary SNOMED-CT code). `DatasetLoader` uses it
  for `LABEL_SCHEME="native"` with a fallback to the legacy database merge — so the cleaned-only
  Kaggle upload can run the all-classes native baseline without `ptbxl_database.csv`.

### Changed
- **Kaggle notebook**: the mount cell now merges **every** attached `/kaggle/input/*` bundle, so the
  **raw** and **cleaned** datasets can be released as two separate Kaggle datasets and attached
  independently ("Save Version" per dataset URL). Readiness auto-detects cleaned-only / raw-only /
  both and skips or runs preprocessing accordingly. How-to and intro cells document the split uploads.

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