# Training Configurations & Best Results

This folder records the **stable / best-performing training configurations** of this
repository together with their measured results, so any experiment can be reproduced
(and reported) without reverse-engineering the code.

> [!NOTE]
> **Legacy 250 Hz reference**: the configs and metrics below were produced under the
> unified **250 Hz scheme** (folders `E2_clean_100_to_250` / `E3_clean_500_to_250`).
> The 250 Hz scheme is currently **deferred** (see [CHANGELOG.md](../../CHANGELOG.md));
> the active scheme runs on the 100/500 Hz folders. These cards are kept as the historical
> architecture/hyperparameter baseline ("Pure CNN Softmax Head" / "Pure CNN Sigmoid Head").

> Reference artifacts live in `models/Pure CNN Softmax Head/` (multiclass Softmax head)
> and `models/Pure CNN Sigmoid Head/` (multi-label Sigmoid head). Every `.md` file below
> is derived from the `experiment_config.json` + `metrics.csv` saved next to those models.

---

## Documents

| Document | Scope |
| :------- | :---- |
| [softmax-head-best-config.md](softmax-head-best-config.md) | Stable **multiclass Softmax** scheme (the "Pure CNN Softmax Head" reference), hyperparameters, architecture, data balancing, and per-model results. |
| [sigmoid-head-best-config.md](sigmoid-head-best-config.md) | Stable **multi-label Sigmoid** scheme, per-class threshold tuning, and results. |
| [label-schemes.md](label-schemes.md) | How labels are defined ("shared 4-class" vs dataset-native), and how cross-dataset training/evaluation works without per-script mappings. |

---

## Common Base Configuration

All experiments share the following base settings (see also
`src/config/experiment_configs.py` and `src/config/config.py`):

| Setting | Value |
| :------ | :---- |
| Input shape | `(2500, 3)` — 10 s at 250 Hz, Leads I / II / III *(legacy 250 Hz; current scheme derives `INPUT_SHAPE` from the selected 100/500 Hz folder)* |
| Target sampling rate | 250 Hz (hardware-aligned to the ADS1293 AFE) — *legacy; current scheme: 100 / 500 Hz* |
| Signal length | 2500 samples |
| Batch size | 64 |
| Epochs | 35 (EarlyStopping patience = 10 on `val_loss`) |
| Optimizer | Adam with CosineDecay (`initial_lr = 3e-4`, `decay_steps = 3000`, `alpha = 1e-2`) |
| Loss | Focal Loss (softmax) / Binary Crossentropy (sigmoid) |
| Reproducibility | Global seed = 42 |
| Split (PTB-XL) | Stratified folds 1–8 train, fold 9 validation, fold 10 isolated test |
| Active label space | `["Normal", "AF", "Takikardia", "Bradikardia"]` (mapped scheme) |

> Depending on the `LABEL_SCHEME` selected in `src/config/experiment_configs.py`,
> training can also run directly on the **dataset-native label names** without any
> per-script mapping — see [label-schemes.md](label-schemes.md).