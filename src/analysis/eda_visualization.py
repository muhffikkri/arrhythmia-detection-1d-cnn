# =====================================================================
# FILE: eda_visualization.py
# FINAL VERSION — DATASET EDA & DATA QUALITY AUDIT
# =====================================================================

import os
import sys
import ast
import warnings

# Add the project root directory to the python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from src.config import config as cfg

warnings.filterwarnings("ignore")

# =====================================================================
# OUTPUT DIRECTORY
# =====================================================================

EDA_DIR = os.path.join(
    cfg.BASE_DIR,
    "output",
    "eda"
)

os.makedirs(EDA_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

print("=" * 80)
print("📊 MEMULAI EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# =====================================================================
# LOAD MANIFESTS
# =====================================================================

ptb_manifest_path = os.path.join(
    cfg.RESAMPLE_BASE,
    "manifest_ptbxl.csv"
)

chap_manifest_path = os.path.join(
    cfg.RESAMPLE_BASE,
    "manifest_chapman.csv"
)

ptb_manifest = pd.read_csv(ptb_manifest_path)
chap_manifest = pd.read_csv(chap_manifest_path)

print(f"\n✓ PTB-XL Samples : {len(ptb_manifest)}")
print(f"✓ Chapman Samples: {len(chap_manifest)}")

# =====================================================================
# LOAD PTBXL DATABASE
# =====================================================================

print("\n--> Loading PTB-XL metadata...")

ptb_db = pd.read_csv(
    cfg.PTBXL_CSV,
    index_col=0
)

if "scp_codes" in ptb_db.columns:
    ptb_db["scp_codes"] = ptb_db["scp_codes"].apply(
        ast.literal_eval
    )

# =====================================================================
# PTBXL DEMOGRAPHICS
# =====================================================================

merge_cols = []

for col in ["age", "sex"]:
    if col in ptb_db.columns:
        merge_cols.append(col)

ptb_demo = ptb_manifest.merge(
    ptb_db[merge_cols],
    left_on="ecg_id",
    right_index=True,
    how="left"
)

ptb_demo["Dataset"] = "PTB-XL"

if "sex" in ptb_demo.columns:

    ptb_demo["sex"] = ptb_demo["sex"].map({
        0: "Male",
        1: "Female"
    })

ptb_demo.rename(columns={
    "age": "Age",
    "sex": "Sex",
    "target_class": "Target Class"
}, inplace=True)

# =====================================================================
# CHAPMAN DEMOGRAPHICS
# =====================================================================

print("\n--> Extracting Chapman demographics...")

def find_hea_file(file_id):

    for root, _, files in os.walk(cfg.CHAPMAN_RECS):

        if f"{file_id}.hea" in files:
            return os.path.join(root, f"{file_id}.hea")

    return None


chap_demographics = []

for filename in tqdm(
    chap_manifest["filename_npy"],
    desc="Parsing Chapman Headers"
):

    file_id = filename.replace(
        "chap_",
        ""
    ).replace(
        ".npy",
        ""
    )

    hea_path = find_hea_file(file_id)

    age = np.nan
    sex = "Unknown"

    if hea_path is not None:

        try:

            with open(hea_path, "r") as f:

                for line in f:

                    if line.startswith("#Age:"):

                        try:
                            age = float(
                                line.split(":")[1].strip()
                            )
                        except:
                            pass

                    elif line.startswith("#Sex:"):

                        sex = line.split(":")[1].strip()

        except:
            pass

    chap_demographics.append({
        "filename_npy": filename,
        "Age": age,
        "Sex": sex
    })

chap_demo_df = pd.DataFrame(chap_demographics)

chap_demo = chap_manifest.merge(
    chap_demo_df,
    on="filename_npy",
    how="left"
)

chap_demo["Dataset"] = "Chapman"

chap_demo.rename(columns={
    "target_class": "Target Class"
}, inplace=True)

chap_demo["Sex"] = chap_demo["Sex"].replace({
    "M": "Male",
    "F": "Female"
})

chap_demo["Sex"] = chap_demo["Sex"].apply(
    lambda x:
    x if x in ["Male", "Female"]
    else "Unknown"
)

# =====================================================================
# MASTER DATAFRAME
# =====================================================================

df_master = pd.concat([

    ptb_demo[[
        "Dataset",
        "Target Class",
        "Age",
        "Sex"
    ]],

    chap_demo[[
        "Dataset",
        "Target Class",
        "Age",
        "Sex"
    ]]

], ignore_index=True)

# =====================================================================
# CLEAN AGE
# =====================================================================

df_master["Age"] = pd.to_numeric(
    df_master["Age"],
    errors="coerce"
)

df_master.loc[
    (df_master["Age"] < 0) |
    (df_master["Age"] > 120),
    "Age"
] = np.nan

# =====================================================================
# SAVE BASIC STATISTICS
# =====================================================================

stats_df = df_master.groupby(
    ["Dataset", "Target Class"]
).agg({
    "Age": ["count", "mean", "median", "std"]
})

stats_path = os.path.join(
    EDA_DIR,
    "dataset_statistics.csv"
)

stats_df.to_csv(stats_path)

print(f"\n✓ Statistics saved:")
print(stats_path)

# =====================================================================
# CLASS DISTRIBUTION
# =====================================================================

print("\n--> Rendering class distribution...")

plt.figure(figsize=(12, 6))

ax = sns.countplot(
    data=df_master,
    x="Target Class",
    hue="Dataset",
    order=cfg.CLASS_NAMES
)

plt.title(
    "Class Distribution",
    fontsize=14,
    fontweight="bold"
)

for p in ax.patches:

    height = p.get_height()

    if height > 0:

        ax.annotate(
            f"{int(height)}",
            (
                p.get_x() + p.get_width()/2,
                height
            ),
            ha="center",
            va="bottom",
            fontsize=8
        )

plt.tight_layout()

plt.savefig(
    os.path.join(
        EDA_DIR,
        "plot_1_class_distribution.png"
    ),
    dpi=300
)

plt.close()

# =====================================================================
# AGE DISTRIBUTION
# =====================================================================

print("--> Rendering age distribution...")

plt.figure(figsize=(12, 6))

sns.histplot(
    data=df_master,
    x="Age",
    hue="Dataset",
    bins=30,
    kde=True,
    alpha=0.4
)

plt.title(
    "Age Distribution",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        EDA_DIR,
        "plot_2_age_distribution.png"
    ),
    dpi=300
)

plt.close()

# =====================================================================
# SEX DISTRIBUTION
# =====================================================================

print("--> Rendering sex distribution...")

df_sex = df_master[
    df_master["Sex"].isin([
        "Male",
        "Female"
    ])
]

plt.figure(figsize=(8, 6))

ax = sns.countplot(
    data=df_sex,
    x="Dataset",
    hue="Sex"
)

plt.title(
    "Sex Distribution",
    fontsize=14,
    fontweight="bold"
)

for p in ax.patches:

    height = p.get_height()

    if height > 0:

        ax.annotate(
            f"{int(height)}",
            (
                p.get_x() + p.get_width()/2,
                height
            ),
            ha="center",
            va="bottom",
            fontsize=8
        )

plt.tight_layout()

plt.savefig(
    os.path.join(
        EDA_DIR,
        "plot_3_sex_distribution.png"
    ),
    dpi=300
)

plt.close()

# =====================================================================
# AGE VS CLASS
# =====================================================================

print("--> Rendering age vs class...")

plt.figure(figsize=(14, 6))

sns.boxplot(
    data=df_master,
    x="Target Class",
    y="Age",
    hue="Dataset",
    order=cfg.CLASS_NAMES
)

plt.title(
    "Age vs Arrhythmia Class",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        EDA_DIR,
        "plot_4_age_vs_class.png"
    ),
    dpi=300
)

plt.close()

# =====================================================================
# STRATIFIED FOLD DISTRIBUTION
# =====================================================================

if "strat_fold" in ptb_manifest.columns:

    print("--> Rendering stratified fold distribution...")

    plt.figure(figsize=(10, 5))

    fold_counts = (
        ptb_manifest["strat_fold"]
        .value_counts()
        .sort_index()
    )

    ax = sns.barplot(
        x=fold_counts.index,
        y=fold_counts.values
    )

    plt.title(
        "PTB-XL Stratified Fold Distribution",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("Fold")
    plt.ylabel("Samples")

    for i, v in enumerate(fold_counts.values):

        plt.text(
            i,
            v + 5,
            str(v),
            ha="center"
        )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EDA_DIR,
            "plot_5_strat_fold_distribution.png"
        ),
        dpi=300
    )

    plt.close()

# =====================================================================
# DUPLICATE CHECK
# =====================================================================

print("\n--> Running duplicate audit...")

dup_ptb = ptb_manifest[
    ptb_manifest["filename_npy"].duplicated()
]

dup_chap = chap_manifest[
    chap_manifest["filename_npy"].duplicated()
]

print(f"PTB duplicates : {len(dup_ptb)}")
print(f"Chap duplicates: {len(dup_chap)}")

# =====================================================================
# PATIENT LEAKAGE AUDIT
# =====================================================================

print("\n--> Running patient leakage audit...")

if "strat_fold" in ptb_manifest.columns:

    train_patients = set(
        ptb_manifest[
            ptb_manifest["strat_fold"].isin(range(1, 9))
        ]["patient_id"]
    )

    test_patients = set(
        ptb_manifest[
            ptb_manifest["strat_fold"] == 10
        ]["patient_id"]
    )

    overlap = train_patients.intersection(
        test_patients
    )

    print(f"Train Patients : {len(train_patients)}")
    print(f"Test Patients  : {len(test_patients)}")
    print(f"Overlap        : {len(overlap)}")

    if len(overlap) == 0:
        print("✓ No patient leakage detected")
    else:
        print("⚠ Leakage detected")

else:

    overlap = []

# =====================================================================
# ECG OVERLAP AUDIT
# =====================================================================

print("\n--> Running ECG overlap audit...")

if "strat_fold" in ptb_manifest.columns:

    ecg_overlap = ptb_manifest.groupby(
        "ecg_id"
    )["strat_fold"].nunique()

    ecg_overlap = ecg_overlap[
        ecg_overlap > 1
    ]

    print(f"ECG overlap count: {len(ecg_overlap)}")

else:

    ecg_overlap = []

# =====================================================================
# SIGNAL LENGTH DISTRIBUTION
# =====================================================================

print("\n--> Rendering signal length distribution...")

plt.figure(figsize=(10, 5))

length_col = None

if "original_len_500hz" in ptb_manifest.columns:
    length_col = "original_len_500hz"

elif "original_len_100hz" in ptb_manifest.columns:
    length_col = "original_len_100hz"

if length_col is not None:

    sns.histplot(
        ptb_manifest[length_col],
        bins=40,
        kde=True
    )

    plt.title(
        "Original Signal Length Distribution",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("Signal Length")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EDA_DIR,
            "plot_6_signal_length_distribution.png"
        ),
        dpi=300
    )

    plt.close()

# =====================================================================
# CLASS DISTRIBUTION TABLE
# =====================================================================

class_table = pd.crosstab(
    df_master["Dataset"],
    df_master["Target Class"]
)

class_table.to_csv(
    os.path.join(
        EDA_DIR,
        "class_distribution_table.csv"
    )
)

# =====================================================================
# SUMMARY REPORT
# =====================================================================

summary_report = {

    "PTBXL_Total_Samples":
        len(ptb_manifest),

    "Chapman_Total_Samples":
        len(chap_manifest),

    "Total_Combined_Samples":
        len(df_master),

    "Duplicate_PTBFilenames":
        len(dup_ptb),

    "Duplicate_ChapmanFilenames":
        len(dup_chap),

    "Patient_Leakage_Count":
        len(overlap),

    "ECG_Split_Overlap_Count":
        len(ecg_overlap)
}

summary_df = pd.DataFrame(
    [summary_report]
)

summary_df.to_csv(
    os.path.join(
        EDA_DIR,
        "eda_summary_report.csv"
    ),
    index=False
)

# =====================================================================
# FINISHED
# =====================================================================

print("\n" + "=" * 80)
print("✅ EDA FINISHED")
print("=" * 80)

print(f"\nAll outputs saved to:")
print(EDA_DIR)
