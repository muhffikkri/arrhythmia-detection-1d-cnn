# Stable Best Configuration — Multiclass Softmax Head

This is the reference configuration that produced the artifacts inside
`models/Pure CNN Softmax Head/`. It is the **recommended stable baseline** for the
4-class arrhythmia classification scheme (Normal, AF, Takikardia, Bradikardia).

---

## 1. Scheme Overview

* **Task**: single-label multiclass classification (mutually exclusive).
* **Head**: `Dense(4, activation='softmax')`.
* **Decision rule**: `argmax` over the 4 output probabilities.
* **Loss**: **Stable Focal Loss** (`gamma = 2.0`, `alpha = 0.5`, `label_smoothing = 0.0`).
* **Metrics**: Balanced Accuracy, Macro Precision / Recall / F1, Macro AUROC
  (per-class sensitivity / F1 / AUROC exported to `metrics.csv`).

---

## 2. Model Architecture

Built by `src/models/model_factory.py::build_dynamic_cnn(scheme="multiclass")`.

| Hyperparameter | Value |
| :------------- | :---- |
| Filter space | `Medium` = `[64, 128, 256, 256, 512]` |
| Kernel space | `Balanced` = `[15, 11, 7, 5, 3]` |
| Dilation space | `Progressive_Dilation` = `[1, 2, 4, 8, 16]` |
| Temporal head | `Pure_CNN` |
| Residual blocks | 5 (SE channel attention + stochastic depth) |
| MaxPooling | After blocks 0–2 (`pool_size = 2`) |
| Classifier | `GAP → Dense(128, BN, ReLU) → Dropout(0.2) → Dense(4, softmax)` |
| Weight decay | L2 = `3e-4` |
| Dropout | `SpatialDropout1D(0.15)` per residual block |
| Separable conv | `False` (`Config.USE_SEPARABLE_CONV`) |
| Stochastic depth rate | `0.0` |

```python
# experiment_configs.py (equivalent snippet)
FILTER_SPACES   = {"Medium": [64, 128, 256, 256, 512]}
KERNEL_SPACES   = {"Balanced": [15, 11, 7, 5, 3]}
DILATION_SPACES = {"Progressive_Dilation": [1, 2, 4, 8, 16]}
TEMPORAL_MODELS = ["Pure_CNN"]
CLASSES         = 4
INPUT_SHAPE     = (2500, 3)
```

---

## 3. Training Hyperparameters

| Parameter | Value | Source |
| :-------- | :---- | :----- |
| Batch size | 64 | `experiment_config.json` |
| Epochs | 35 | `experiment_config.json` |
| Learning rate | `3e-4` (CosineDecay) | `experiment_config.json` |
| Optimizer | Adam | `experiment_config.json` |
| Loss | Focal Loss | `experiment_config.json` |
| Label smoothing | `0.0` | `experiment_config.json` |
| Augmentation | `false` | `experiment_config.json` |
| Mixup alpha | `0.1` (disabled) | `experiment_config.json` |
| Early stopping | patience 10 on `val_loss`, restore best weights | `run_experiment` callbacks |
| Best checkpoint | `best_model.keras` (best `val_loss`) | ModelCheckpoint |

### Data balancing (optional, applied to the train split only)

The newer experiments (see `output/research_experiments/balanced_baseline/`)
also applied rebalancing:

| Setting | Value |
| :------ | :---- |
| Undersampling | `UNDERSAMPLE_RATIO = 10` (Normal : Bradikardia target ratio) |
| Oversampling | `OVERSAMPLE_METHOD = "smote_tomek"` (also ran `"smote"`) |
| Oversample strategy | `{"AF": 3000, "Bradikardia": 3000, "Takikardia": 3000}` |

---

## 4. Results (reference: `models/Pure CNN Softmax Head/<dir>/metrics.csv`)

Metrics reported on the isolated PTB-XL test fold (fold 10).

| Model dir | Dataset path | Bal. Acc | Macro Prec | Macro Rec | Macro F1 | Macro AUROC | Params | Size |
| :-------- | :------------ | :------- | :--------- | :-------- | :------- | :---------- | :----- | :--- |
| `softmax_filtered_100to250_cnn` | `E2_clean_100_to_250` | 0.8469 | 0.9025 | 0.8469 | **0.8676** | 0.9755 | 3,113,540 | 35.82 MB |
| `softmax_raw_100to250_cnn` | `E2_100_to_250` | **0.8520** | 0.8920 | 0.8520 | **0.8678** | 0.9781 | 3,113,540 | 35.82 MB |
| `softmax_filtered_500to250_cnn` | `E3_clean_500_to_250` | 0.8427 | 0.8953 | 0.8427 | 0.8630 | 0.9754 | 3,113,540 | 35.82 MB |
| `softmax_raw_500to250_cnn` | `E3_500_to_250` | 0.8495 | 0.8924 | 0.8495 | 0.8663 | **0.9784** | 3,113,540 | 35.82 MB |
| `softmax_filtered_100to250_cnnattention` | `E2_clean_100_to_250` | 0.8234 | 0.9082 | 0.8234 | 0.8563 | 0.9753 | 3,377,604 | 38.87 MB |
| `softmax_filtered_500to250_cnnattention` | `E3_clean_500_to_250` | 0.8359 | 0.9046 | 0.8359 | 0.8631 | 0.9763 | 3,377,604 | 38.87 MB |
| `softmax_filtered_100to250_lstm` | `E2_clean_100_to_250` | 0.7541 | 0.8114 | 0.7541 | 0.7795 | 0.9706 | 317,700 | 3.72 MB |
| `softmax_filtered_500to250_lstm` | `E3_clean_500_to_250` | 0.7547 | 0.8454 | 0.7547 | 0.7908 | 0.9702 | 317,700 | 3.72 MB |

### Per-class detail for the recommended model `softmax_filtered_100to250_cnn`

| Class | Sensitivity | Precision | F1 | AUROC |
| :---- | :---------- | :-------- | :-- | :---- |
| Normal | 0.9172 | 0.9057 | 0.9114 | 0.9801 |
| AF | 0.5781 | 0.8605 | 0.6916 | 0.9623 |
| Takikardia | 0.9854 | 0.9771 | 0.9812 | 0.9715 |
| Bradikardia | 0.9070 | 0.8667 | 0.8864 | 0.9879 |

> AF remains the hardest class (typical for class imbalance); the Focal loss and the
> balancing strategies above specifically target this class.

---

## 5. Reproducing this run

With the unified runner (after the repo restructure):

```powershell
# Edit src/config/experiment_configs.py first:
#   SCHEME = "softmax"
#   LABEL_SCHEME = "mapped"
#   TRAIN_DATASET = "PTBXL" ; TEST_DATASET = "PTBXL"
#   TRAIN_FOLDER_KEY = "E2_clean_100_to_250"   # or E3_clean_500_to_250
python -m src.experiments.run_experiment
```

Or equivalently with the legacy scripts (still present in git history):

```powershell
python src/experiments/run_experiment_softmax.py   # multiclass
```