# Label Schemes & Cross-Dataset Training

## 1. Motivation

The repository supports two complementary ways to label a recording:

1. **Shared 4-class scheme (mapped)** — `["Normal", "AF", "Takikardia", "Bradikardia"]`.
   Used whenever we train/evaluate **across datasets** (Chapman ↔ PTB-XL) so that both
   datasets share one identical label space.
2. **Dataset-native scheme** — the labels exactly as published by each dataset
   (PTB-XL SCP codes / Chapman SNOMED-CT codes). Used for **per-dataset** training so
   no information is lost and no mapping is applied.

`LABEL_SCHEME` in `src/config/experiment_configs.py` selects which one an experiment
uses. The label column is then derived **data-driven** from the dataset manifest —
training scripts never hard-code a mapping table.

> Full background reference: `docs/dataset-label-map.md` (label catalogs of both datasets).

---

## 2. Shared 4-class scheme (`LABEL_SCHEME = "mapped"`)

Class order is fixed (this also defines the Softmax/Sigmoid output index order):

```
["Normal", "AF", "Takikardia", "Bradikardia"]
```

### 2.1 PTB-XL → 4-class

**Input**: `scp_codes` column (dictionary of SCP codes → confidence), and the
`diagnostic_superclass` column (`NORM`, `MI`, `CD`, `STTC`, `HYP`) used for the simple
fallback rule.

Rules (`PTBXL_TO_TARGET_MAPPING` in `src/config/config_labels.py`) applied to the SCP codes:

| PTB-XL SCP code | Target class |
| :-------------- | :----------- |
| `SR`, `SARRH`, `NORM` | `Normal` |
| `AFIB`, `AFLT` | `AF` |
| `STACH`, `SVTAC`, `PSVT` | `Takikardia` |
| `SBRAD` | `Bradikardia` |
| `PACE`, `SVARR`, `BIGU`, `TRIGU` | `Others` (excluded from 4-class training) |

Healthy patients are kept exclusively via `diagnostic_superclass == 'NORM'` and the
explicit `NORM` SCP code.

### 2.2 Chapman → 4-class

**Input**: `#Dx` line (1+ SNOMED-CT codes) per record.

Rules (`CHAPMAN_TO_TARGET_MAPPING` in `src/config/config_labels.py`); each row lists
short label, full English name, and SNOMED-CT code for the same class:

| Short / Name / SNOMED | Target class |
| :-------------------- | :----------- |
| `SR`, `Sinus Rhythm`, `426783006` | `Normal` |
| `AFIB`, `AF`, `Atrial Fibrillation`, `164889003`, `Atrial Flutter`, `164890007` | `AF` |
| `ST`, `Sinus Tachycardia`, `427084000`, `SVT`, `Supraventricular Tachycardia`, `426761007`, `AT`, `Atrial Tachycardia`, `713422000` | `Takikardia` |
| `SB`, `Sinus Bradycardia`, `426177001` | `Bradikardia` |
| `AVNRT`/`233896004`, `AVRT`/`233897008`, `SAAWR`/`195101003`, `SA`/`427393009` (& synonyms) | `Others` (excluded) |

> `Others` rows are excluded from **mapped-scheme** training but kept available for
> native-scheme training.

---

## 3. Dataset-native scheme (`LABEL_SCHEME = "native"`)

### 3.1 PTB-XL

- Primary: **SCP codes** from `scp_statements.csv` / the `scp_codes` column
  (e.g. `NORM`, `AFIB`, `SBRAD`, `STACH`, `MI`, `LBBB`, …).
- Fallback / summary: `diagnostic_superclass` (`NORM`, `MI`, `CD`, `STTC`, `HYP`) plus
  the `diagnostic_subclass` / `diagnostic_statement` columns.
- Label count = number of distinct codes kept in the manifest
  (`manifest_counts.json` written during preprocessing).

### 3.2 Chapman

- Primary: **SNOMED-CT codes** from the `#Dx` line and `ConditionNames_SNOMED-CT.csv`
  (e.g. `164889003`, `426177001`, `427084000`, `426783006`, and the non-mapped codes
  such as `284470004` etc., which are kept only in native mode).
- The human-readable name is available from the ConditionNames sidecar for reporting.

### 3.3 Class count inference

The number of classes is **inferred from the manifest**, not hard-coded:

* `mapped` scheme → always 4 (`TARGET_CLASSES`).
* `native` scheme → `max integer-encoded label + 1` from the dataframe produced by the
  dataset loader; the class-name list is the sorted set of unique native labels.

---

## 4. Cross-dataset coupling

| Mode | `TRAIN_DATASET` | `TEST_DATASET` | `LABEL_SCHEME` | Expected |
| :--- | :-------------- | :------------- | :------------- | :------- |
| Per-dataset PTB-XL | `PTBXL` | `PTBXL` | `native` | PTB-XL SCP classes |
| Per-dataset Chapman | `CHAPMAN` | `CHAPMAN` | `native` | Chapman SNOMED classes |
| Cross-dataset PTB→Chapman | `PTBXL` | `CHAPMAN` | `mapped` | shared 4-class |
| Cross-dataset Chapman→PTB | `CHAPMAN` | `PTBXL` | `mapped` | shared 4-class |

For the cross-dataset modes the model is trained on the train split of the source
dataset and evaluated (zero-shot) on the **test split** of the target dataset, using the
same 4-class output head. See `src/evaluation/cross_dataset_test.py`.

---

## 5. Preprocessing / label persistence

* `scripts/data_utils.py` (`load_*_data`) create the module-level dataframe that already
  contains the final `label` column using the scheme above — no per-script mapping.
* A `manifest_counts.json` is exported alongside per-dataset pipeline output so class
  counts are reproducible and visible without re-reading the raw data.
* The 4-class `mapped` filtering only touches train/val/test splits; the full native
  labels remain stored in the manifest for later native experiments.