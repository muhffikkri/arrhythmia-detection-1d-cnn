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
- Stored per record as `native_label` in `manifest_ptbxl.csv`.
- Native class count = number of distinct `native_label` SCP codes in the manifest
  (dropped later at load time only for classes too rare to stratify).

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

* Preprocessing (`proccess_ptbxl.py` / `proccess_chapman.py`, pipeline **v6.1+**) stores
  **every** ECG record in the 100/500 Hz folders. Each manifest row carries:
  - `native_label` → the original SCP (PTB-XL) / SNOMED-CT (Chapman) code, used directly by
    `LABEL_SCHEME="native"` — **no class mapping is applied**;
  - `target_class` → the derived shared 4-class value (blank for records that do not map),
    used only by the legacy / cross-dataset `mapped` scheme.
* No record is dropped at preprocessing time because it falls outside the 4-class mapping;
  the mapped filter is applied **only at load time** by `DatasetLoader.resolve_label`
  (`target_class.isin(TARGET_CLASSES)`).
* Native experiments therefore see the full per-dataset label space; the class list is derived
  data-driven from the manifest (`class_names = sorted(unique native labels)`).

---

## 6. Config-driven class selection

Which classes a native experiment actually uses is decided **only by config**
(`src/config/experiment_configs.py`). Training and EDA both read it, so the class
distribution shown by EDA is exactly the one training sees.

| Config | Default | Meaning |
| :--- | :--- | :--- |
| `LABEL_SCHEME` | `"native"` | `"native"` → per-dataset codes; `"mapped"` → shared 4-class |
| `USE_NATIVE_ALL_CLASSES` | `True` | **The native toggle**: while `True` no mapping/filtering is applied — every class present in the manifest is used (3 leads) |
| `NATIVE_CLASS_SELECTION` | `"all"` | Future hook (ignored while `USE_NATIVE_ALL_CLASSES=True`): `"all"` / `"allowlist"` / `"map"` |
| `NATIVE_CLASS_ALLOWLIST` | `[]` | Native codes kept when `NATIVE_CLASS_SELECTION="allowlist"` |
| `NATIVE_CLASS_MAPPING` | `{}` | Rewrites native code → target label when `NATIVE_CLASS_SELECTION="map"` |
| `MIN_CLASS_COUNT` | `8` | Classes with fewer samples are dropped at split time (cannot be stratified) |

* `DatasetLoader.resolve_label` applies the selection after resolving labels, then derives
  `class_names` data-driven from the surviving labels.
* `src/analysis/eda_dataset_profiles.py` calls the **same** `resolve_label`, so its figures
  and CSVs follow the config: with `native` + all classes it plots the **full** native class
  distribution (one figure per folder + per active folder, long-format
  `class_distribution_by_folder.csv`); with `mapped` it plots the shared 4-class distribution.