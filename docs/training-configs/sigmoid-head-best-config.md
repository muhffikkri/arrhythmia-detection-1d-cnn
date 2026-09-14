# Stable Best Configuration — Multi-label Sigmoid Head

> [!NOTE]
> **Legacy baseline**: measured under the deferred **250 Hz scheme**
> (`E2_clean_100_to_250` / `E3_clean_500_to_250`). Keep as a historical
> architecture/hyperparameter reference; the active scheme uses 100/500 Hz folders.

This is the reference configuration for the **multi-label** arrhythmia scheme
(Sigmoid head + per-class threshold tuning). The artifacts live inside
`models/Pure CNN Sigmoid Head/`.

> The class order for the Sigmoid scheme is `["Normal", "AF", "Takikardia", "Bradikardia"]`
> (indices 0–3). Every threshold vector below follows that order.

---

## 1. Scheme Overview

* **Task**: multi-label classification (a recording may belong to > 1 class).
* **Head**: `Dense(4, activation='sigmoid')` (`name="multilabel_output"`).
* **Loss**: `BinaryCrossentropy` (`metrics = ['binary_accuracy']`).
* **Decision rule**: per-class adaptive threshold (F1-optimized on validation set).
* **Threshold sweep**: `np.linspace(0.1, 0.9, 81)` per class, maximizing class F1.

---

## 2. Model Architecture

Identical backbone as the Softmax scheme, built with
`build_dynamic_cnn(..., scheme="multilabel")`:

| Hyperparameter | Value |
| :------------- | :---- |
| Filter space | `Medium` = `[64, 128, 256, 256, 512]` |
| Kernel space | `Balanced` = `[15, 11, 7, 5, 3]` |
| Dilation space | `Progressive_Dilation` = `[1, 2, 4, 8, 16]` |
| Temporal head | `Pure_CNN` |
| Classifier | `GAP → Dense(128, BN, ReLU) → Dropout(0.2) → Dense(4, sigmoid)` |
| Parameters | 3,113,540 (35.82 MB `.keras`) |

---

## 3. Training Hyperparameters

| Parameter | Value | Source |
| :-------- | :---- | :----- |
| Batch size | 64 | `experiment_config.json` |
| Epochs | 35 | `experiment_config.json` |
| Learning rate | `3e-4` (CosineDecay) | `experiment_config.json` |
| Optimizer | Adam | `experiment_config.json` |
| Loss | Binary Crossentropy | `experiment_config.json` |
| Early stopping | patience 10 on `val_loss` | callbacks |
| Checkpoint | `best_model.keras` | ModelCheckpoint |

---

## 4. Results & Tuned Thresholds

Metrics reported on the isolated PTB-XL test fold (fold 10) using the tuned thresholds.

### `multilabel_100to250` (`E2_clean_100_to_250` → 100→250 Hz path)

Thresholds: `[0.46, 0.82, 0.19, 0.42]` (Normal, AF, Takikardia, Bradikardia)

| Class | F1 | AUROC |
| :---- | :-- | :---- |
| Normal | 0.9810 | 0.9704 |
| AF | 0.9020 | 0.9740 |
| Takikardia | 0.8876 | 0.9869 |
| Bradikardia | 0.6667 | 0.9585 |

### `multilabel_500to250` (`E3_clean_500_to_250` → 500→250 Hz path)

Thresholds: `[0.42, 0.52, 0.20, 0.37]` (Normal, AF, Takikardia, Bradikardia)

| Class | F1 | AUROC |
| :---- | :-- | :---- |
| Normal | 0.9791 | 0.9706 |
| AF | 0.8987 | 0.9794 |
| Takikardia | 0.8778 | 0.9867 |
| Bradikardia | 0.6545 | 0.9593 |

> Bradikardia has the lowest F1 in the multi-label setting, matching the multiclass
> observation that this class is the least numerous and hardest to separate.

---

## 5. Reproducing this run

```powershell
# Edit src/config/experiment_configs.py first:
#   SCHEME = "sigmoid"
#   LABEL_SCHEME = "mapped"
#   TRAIN_DATASET = "PTBXL" ; TEST_DATASET = "PTBXL"
#   TRAIN_FOLDER_KEY = "E2_clean_100_to_250"
python -m src.experiments.run_experiment
```

The runner saves `metrics.csv` with `Thresholds_Assigned` so the tuned thresholds can
be reused later during cross-dataset evaluation.