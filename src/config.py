# config.py

import os

# =========================================================

# BASE DIRECTORY

# =========================================================

BASE_DIR = r"C:\Users\DESKTOP\Desktop\arrhythmia-detection-1d-cnn"

DATASET_DIR = os.path.join(BASE_DIR, "dataset")

OUTPUT_DIR = os.path.join(
BASE_DIR,
"output",
# "experiment_check"
)

# =========================================================

# PTB-XL

# =========================================================

PTBXL_CSV = os.path.join(
DATASET_DIR,
"PTB-XL",
"ptbxl_database.csv"
)

PTBXL_SCP = os.path.join(
DATASET_DIR,
"PTB-XL",
"scp_statements.csv"
)

# =========================================================

# CHAPMAN

# =========================================================

CHAPMAN_CSV = os.path.join(
DATASET_DIR,
"Chapman",
"ConditionNames_SNOMED-CT.csv"
)

CHAPMAN_RECS = os.path.join(
DATASET_DIR,
"Chapman",
"WFDBRecords"
)

# =========================================================

# RESAMPLE DIRECTORY

# =========================================================

RESAMPLE_BASE = os.path.join(
DATASET_DIR,
"resample"
)

# =========================================================

# DATA FOLDERS

# =========================================================

SUB_FOLDERS = {

# =====================================================
# LEGACY / OPTIONAL
# =====================================================

"E1_100_native":
    os.path.join(RESAMPLE_BASE, "exp_100_native"),

"E2_100_to_250":
    os.path.join(RESAMPLE_BASE, "exp_100_to_250"),

"E3_500_to_250":
    os.path.join(RESAMPLE_BASE, "exp_500_to_250"),

"E4_500_native":
    os.path.join(RESAMPLE_BASE, "exp_500_native"),

# =====================================================
# CLEANED SIGNALS
# =====================================================

"E1_clean_100_native":
    os.path.join(RESAMPLE_BASE, "cleaned_100_native"),

"E2_clean_100_to_250":
    os.path.join(RESAMPLE_BASE, "cleaned_100_to_250"),

"E3_clean_500_to_250":
    os.path.join(RESAMPLE_BASE, "cleaned_500_to_250"),

"E4_clean_500_native":
    os.path.join(RESAMPLE_BASE, "cleaned_500_native"),

# =====================================================
# FINAL ACTIVE DATASETS
# =====================================================

# Final Chapman external validation dataset
"CHAPMAN_CLEAN_250HZ":
    os.path.join(RESAMPLE_BASE, "chapman_clean_500_to_250")

}

# =========================================================

# AUTO CREATE DIRECTORIES

# =========================================================

for folder_path in SUB_FOLDERS.values():
    os.makedirs(folder_path, exist_ok=True)

# =========================================================

# CLASS LABELS

# =========================================================

CLASS_NAMES = [
'Normal',
'AF',
'Takikardia',
'Bradikardia',
'Others'
]

# =========================================================

# ECG LEADS

# =========================================================

LEAD_NAMES = ['I', 'II', 'III']

LEAD_INDICES = [0, 1, 2]

# =========================================================

# TARGET SIGNAL LENGTHS

# =========================================================

TARGET_LEN = {
100: 1000,
250: 2500,
500: 5000
}

# =========================================================

# OUTPUT CLASS DIRECTORIES

# =========================================================

for cls in CLASS_NAMES:

    os.makedirs(
        os.path.join(OUTPUT_DIR, cls),
        exist_ok=True
    )
