# =====================================================================
# FILE: eda_dataset_profiles.py
# PER-DATASET + COMBINED DATASET PROFILE
# =====================================================================
# Per-folder class distribution, PTB-XL / Chapman demographics (age, sex),
# and a missing-data audit across the 100/500 Hz raw/cleaned folders.
# All figures land in output/eda_dataset_profiles/.
# =====================================================================

import os
import sys
import ast
import warnings

# Add the project root directory to the python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Windows console default (cp1252) cannot encode emoji/symbols used in prints.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from src.config import config as cfg
from src.config.experiment_configs import Config
from src.training import dataset_loader as dl

warnings.filterwarnings("ignore")

# =====================================================================
# OUTPUT DIRECTORY
# =====================================================================

OUT_DIR = os.path.join(cfg.BASE_DIR, "output", "eda_dataset_profiles")
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

# Active 100/500 Hz folders to profile (raw + clean for each dataset/fs).
ACTIVE_FOLDERS = [
    "ptbxl_raw_100hz",
    "ptbxl_clean_100hz",
    "ptbxl_raw_500hz",
    "ptbxl_clean_500hz",
    "chapman_raw_100hz",
    "chapman_clean_100hz",
    "chapman_raw_500hz",
    "chapman_clean_500hz",
]

print("=" * 80)
print("PROFIL DATASET (PTB-XL vs CHAPMAN)")
print("=" * 80)

# =====================================================================
# LOAD MANIFESTS
# =====================================================================

ptb_manifest_path = os.path.join(cfg.RESAMPLE_BASE, "manifest_ptbxl.csv")
chap_manifest_path = os.path.join(cfg.RESAMPLE_BASE, "manifest_chapman.csv")

ptb_manifest = pd.read_csv(ptb_manifest_path)
chap_manifest = pd.read_csv(chap_manifest_path)

print(f"\nPTB-XL  manifest rows: {len(ptb_manifest)}")
print(f"Chapman manifest rows: {len(chap_manifest)}")

# =====================================================================
# CONFIG-DRIVEN LABEL RESOLUTION (EDA FOLLOWS THE TRAINING CONFIG)
# =====================================================================
# The class distribution below mirrors exactly what training will use:
#   LABEL_SCHEME="native" (+ Config.USE_NATIVE_ALL_CLASSES) -> the FULL
#   native class space (every class); "mapped" -> shared 4 classes.
# Future Config.NATIVE_CLASS_SELECTION allowlist/map filters are honoured
# automatically because EDA reuses the dataset loader's resolve_label().
LABEL_SCHEME = getattr(Config, "LABEL_SCHEME", "native")

def resolve_manifest_labels(manifest, dataset):
    resolved, names = dl.resolve_label(manifest.copy(), dataset, LABEL_SCHEME)
    print(
        f"[EDA] {dataset}: LABEL_SCHEME='{LABEL_SCHEME}' "
        f"-> {len(names)} class(es)"
    )
    return resolved, names

ptb_manifest, PTB_CLASSES = resolve_manifest_labels(ptb_manifest, "PTBXL")
chap_manifest, CHAP_CLASSES = resolve_manifest_labels(chap_manifest, "CHAPMAN")
print(f"[EDA] PTB-XL class names  : {PTB_CLASSES}")
print(f"[EDA] Chapman class names : {CHAP_CLASSES}")

# =====================================================================
# PTB-XL DEMOGRAPHICS (age / sex from the database CSV)
# =====================================================================

def load_ptb_demographics(manifest):
    demo = manifest.copy()
    if not os.path.exists(cfg.PTBXL_CSV):
        demo["Age"] = np.nan
        demo["Sex"] = "Unknown"
        return demo

    db = pd.read_csv(cfg.PTBXL_CSV, index_col=0)
    merge_cols = [c for c in ["age", "sex"] if c in db.columns]
    if not merge_cols:
        demo["Age"] = np.nan
        demo["Sex"] = "Unknown"
        return demo

    demo = demo.merge(db[merge_cols], left_on="ecg_id", right_index=True, how="left")
    demo["Sex"] = demo["sex"].map({0: "Male", 1: "Female"})
    demo = demo.drop(columns=["sex"], errors="ignore")
    demo["Age"] = pd.to_numeric(demo["age"], errors="coerce")
    demo = demo.drop(columns=["age"], errors="ignore")
    demo.loc[(demo["Age"] < 0) | (demo["Age"] > 120), "Age"] = np.nan
    demo["Sex"] = demo["Sex"].fillna("Unknown")
    return demo


def load_chapman_demographics(manifest):
    """Single-pass .hea walk -> Age/Sex map (matches eda_visualization)."""
    hea_map = {}
    for root, _, files in os.walk(cfg.CHAPMAN_RECS):
        for fname in files:
            if fname.endswith(".hea"):
                hea_map[os.path.splitext(fname)[0]] = os.path.join(root, fname)

    records = []
    for fname in tqdm(manifest["filename_npy"], desc="Parsing Chapman headers"):
        file_id = fname.replace("chap_", "").replace(".npy", "")
        hea_path = hea_map.get(file_id)
        age, sex = np.nan, "Unknown"
        if hea_path is not None:
            try:
                with open(hea_path, "r") as fh:
                    for line in fh:
                        if line.startswith("#Age:"):
                            try:
                                age = float(line.split(":", 1)[1].strip())
                            except Exception:
                                age = np.nan
                        elif line.startswith("#Sex:"):
                            sex = line.split(":", 1)[1].strip()
            except Exception:
                pass
        records.append({"filename_npy": fname, "Age": age, "Sex": sex})

    demo = manifest.merge(pd.DataFrame(records), on="filename_npy", how="left")
    demo["Age"] = pd.to_numeric(demo["Age"], errors="coerce")
    demo.loc[(demo["Age"] < 0) | (demo["Age"] > 120), "Age"] = np.nan
    demo["Sex"] = demo["Sex"].replace({"M": "Male", "F": "Female"})
    demo["Sex"] = demo["Sex"].apply(
        lambda x: x if x in ["Male", "Female"] else "Unknown"
    )
    return demo


print("\n--> Merging PTB-XL demographics...")
ptb_dem = load_ptb_demographics(ptb_manifest)
ptb_dem["Dataset"] = "PTB-XL"

print("--> Merging Chapman demographics...")
chap_dem = load_chapman_demographics(chap_manifest)
chap_dem["Dataset"] = "Chapman"

for df in (ptb_dem, chap_dem):
    if "age" in df.columns and "Age" not in df.columns:
        df["Age"] = pd.to_numeric(df["age"], errors="coerce")

# =====================================================================
# PER-FOLDER CLASS DISTRIBUTION + MISSING-DATA AUDIT
# =====================================================================

summary_rows = []
class_rows = []
for key in ACTIVE_FOLDERS:
    dataset = "PTB-XL" if key.startswith("ptbxl") else "Chapman"
    dem = ptb_dem if dataset == "PTB-XL" else chap_dem
    class_list = PTB_CLASSES if dataset == "PTB-XL" else CHAP_CLASSES
    fs = cfg.folder_fs(key)

    path_col = f"path_{key}"
    if path_col not in dem.columns:
        continue

    has_path = dem[path_col].notna().astype(bool)
    n_present = int(has_path.sum())
    n_missing = int((~has_path).sum())
    n_on_disk = 0
    for p in dem.loc[has_path, path_col]:
        resolved = cfg.resolve_path(p)
        if resolved is not None and os.path.exists(resolved):
            n_on_disk += 1

    cls_counts = dem.loc[has_path, "label"].value_counts()
    row = {
        "folder_key": key,
        "dataset": dataset,
        "sampling_rate_hz": fs,
        "kind": "Clean" if "clean" in key else "Raw",
        "n_classes": int(len(class_list)),
        "n_manifest_rows": int(len(dem)),
        "n_signal_rows": int(has_path.sum()),
        "n_missing_path": n_missing,
        "n_missing_on_disk": n_missing + int(has_path.sum()) - n_on_disk,
        "n_missing_age": int(dem.loc[has_path, "Age"].isna().sum()),
        "n_unknown_sex": int((dem.loc[has_path, "Sex"] == "Unknown").sum()),
    }
    summary_rows.append(row)

    # Full, config-driven class distribution (long format scales to any
    # number of native classes, unlike one CSV column per class).
    for cls in class_list:
        class_rows.append({
            "folder_key": key,
            "dataset": dataset,
            "sampling_rate_hz": fs,
            "kind": "Clean" if "clean" in key else "Raw",
            "class_name": cls,
            "n_samples": int(cls_counts.get(cls, 0)),
        })

summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUT_DIR, "profile_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\nSaved: {summary_path}")

class_df = pd.DataFrame(class_rows)
class_path = os.path.join(OUT_DIR, "class_distribution_by_folder.csv")
class_df.to_csv(class_path, index=False)
print(f"Saved: {class_path}")

# =====================================================================
# MISSING-DATA AUDIT FIGURE
# =====================================================================

fig, ax = plt.subplots(figsize=(13, 5))
plot_df = summary_df.melt(
    id_vars=["folder_key"],
    value_vars=["n_missing_path", "n_missing_on_disk", "n_missing_age", "n_unknown_sex"],
    var_name="Type",
    value_name="Count"
)
sns.barplot(data=plot_df, x="folder_key", y="Count", hue="Type", ax=ax)
ax.set_title("Missing-Data Audit per Folder", fontsize=14, fontweight="bold")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "missing_data_audit.png"), dpi=150)
plt.close()

# =====================================================================
# CLASS DISTRIBUTION — FULL, CONFIG-DRIVEN (ONE FIGURE PER FOLDER)
# =====================================================================
# Native ("all classes") can contain dozens of labels, so each folder gets
# its own horizontal bar chart instead of a cramped 2x4 grid. The class
# list is exactly the one training will use (Config.LABEL_SCHEME + the
# native selection switch).

def _plot_full_distribution(counts, class_list, title, out_path):
    counts = counts.reindex(class_list, fill_value=0).sort_values(ascending=False)
    height = max(4.0, 0.32 * len(counts))
    fig, ax = plt.subplots(figsize=(11, height))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, palette="viridis")
    for i, v in enumerate(counts.values):
        ax.text(v, i, f" {int(v)}", va="center", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Class")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


for key in ACTIVE_FOLDERS:
    dataset = "PTB-XL" if key.startswith("ptbxl") else "Chapman"
    dem = ptb_dem if dataset == "PTB-XL" else chap_dem
    class_list = PTB_CLASSES if dataset == "PTB-XL" else CHAP_CLASSES
    path_col = f"path_{key}"
    if path_col not in dem.columns:
        continue
    sub = dem[dem[path_col].notna()]
    _plot_full_distribution(
        sub["label"].value_counts(),
        class_list,
        f"Class Distribution — {key} ({cfg.folder_fs(key)} Hz, "
        f"scheme='{LABEL_SCHEME}', {len(class_list)} classes)",
        os.path.join(OUT_DIR, f"class_distribution_{key}.png"),
    )

# =====================================================================
# COMBINED FIGURES (ACTIVE FOLDERS ONLY)
# =====================================================================

active_ptb = getattr(cfg, "FOLDER_PTBXL", "ptbxl_clean_500hz")
active_chap = getattr(cfg, "FOLDER_CHAPMAN", "chapman_clean_500hz")

# Active folder per dataset — full class distribution of what training
# actually selects under the current config.
for key, dem, class_list in (
    (active_ptb, ptb_dem, PTB_CLASSES),
    (active_chap, chap_dem, CHAP_CLASSES),
):
    path_col = f"path_{key}"
    if path_col not in dem.columns:
        continue
    sub = dem[dem[path_col].notna()]
    _plot_full_distribution(
        sub["label"].value_counts(),
        class_list,
        f"Active Folder Full Class Distribution — {key} "
        f"(scheme='{LABEL_SCHEME}', {len(class_list)} classes)",
        os.path.join(OUT_DIR, f"class_distribution_active_{key}.png"),
    )


def active_frame(ptb_key, chap_key):
    parts = []
    for key in (ptb_key, chap_key):
        dataset = "PTB-XL" if key.startswith("ptbxl") else "Chapman"
        dem = ptb_dem if dataset == "PTB-XL" else chap_dem
        path_col = f"path_{key}"
        if path_col not in dem.columns:
            continue
        sub = dem[dem[path_col].notna()].copy()
        parts.append(sub[["Dataset", "label", "Age", "Sex"]])
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


comb = active_frame(active_ptb, active_chap)

if not comb.empty:
    # Class distribution — only when labels are comparable across datasets
    # (shared 4-class "mapped" scheme). Under "native" the class spaces
    # differ, so the per-dataset full distributions above are used instead.
    if LABEL_SCHEME == "mapped":
        plt.figure(figsize=(12, 6))
        ax = sns.countplot(
            data=comb.rename(columns={"label": "Target Class"}),
            x="Target Class", hue="Dataset", order=cfg.CLASS_NAMES
        )
        plt.title(f"Class Distribution — {active_ptb} vs {active_chap}",
                  fontsize=14, fontweight="bold")
        for p in ax.patches:
            h = p.get_height()
            if h > 0:
                ax.annotate(f"{int(h)}", (p.get_x() + p.get_width() / 2, h),
                            ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "class_distribution_combined.png"), dpi=150)
        plt.close()

    # Age distribution — PTB-XL vs Chapman
    plt.figure(figsize=(12, 6))
    sns.histplot(data=comb, x="Age", hue="Dataset", bins=30, kde=True, alpha=0.4)
    plt.title("Age Distribution — PTB-XL vs Chapman", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "age_distribution_combined.png"), dpi=150)
    plt.close()

    # Sex distribution — PTB-XL vs Chapman
    df_sex = comb[comb["Sex"].isin(["Male", "Female"])]
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df_sex, x="Dataset", hue="Sex")
    plt.title("Sex Distribution — PTB-XL vs Chapman", fontsize=14, fontweight="bold")
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{int(h)}", (p.get_x() + p.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sex_distribution_combined.png"), dpi=150)
    plt.close()

# =====================================================================
# FINISHED
# =====================================================================

print("\n" + "=" * 80)
print("DATASET PROFILES FINISHED")
print("=" * 80)
print(f"\nAll outputs saved to:\n{OUT_DIR}")