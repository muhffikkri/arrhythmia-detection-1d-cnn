# config.py

import os

# =========================================================

# BASE DIRECTORY

# =========================================================

try:
    # 1. LOCAL ENVIRONMENT (Python Scripts)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")
    RESAMPLE_BASE = os.path.join(DATASET_DIR, "resample")
except NameError:
    # 2. NOTEBOOK ENVIRONMENT (__file__ is undefined)
    BASE_DIR = os.getcwd()
    
    if os.path.exists("/kaggle/input"):
        # Kaggle Environment
        # Auto-detect the dataset path in case the Kaggle slug differs
        _found_resample = None
        for root, dirs, files in os.walk("/kaggle/input"):
            if "manifest_ptbxl.csv" in files:
                _found_resample = root
                break
        
        if _found_resample:
            RESAMPLE_BASE = _found_resample
        else:
            RESAMPLE_BASE = "/kaggle/input/ekg-dataset"
            
        DATASET_DIR = os.path.dirname(RESAMPLE_BASE)
    else:
        # Local Jupyter Notebook Environment
        DATASET_DIR = os.path.join(BASE_DIR, "dataset")
        RESAMPLE_BASE = os.path.join(DATASET_DIR, "resample")

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
    "PTBXL",
    "ptbxl_database.csv"
)

PTBXL_SCP = os.path.join(
    DATASET_DIR,
    "PTBXL",
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


# =========================================================

# PORTABLE (RELATIVE) PATHS FOR MANIFESTS

# =========================================================
# Manifest columns must store paths RELATIVE to BASE_DIR so the
# dataset/resample folder can be moved or uploaded (e.g. to Kaggle)
# without rewriting the CSVs. Use to_relative_path() when writing and
# resolve_path() when reading.

def to_relative_path(path):
    """Absolute path -> portable path relative to BASE_DIR (POSIX sep)."""
    if path is None:
        return None
    try:
        rel = os.path.relpath(os.path.abspath(str(path)), BASE_DIR)
    except (TypeError, ValueError):
        return path
    return rel.replace(os.sep, "/")


def to_kaggle_relative_path(path):
    """Absolute path -> portable path relative to RESAMPLE_BASE (POSIX sep).
    Used for Kaggle manifests where the resample/ folder is uploaded as dataset root.
    Result: 'ptbxl_raw_500hz/filename.npy' instead of 'dataset/resample/ptbxl_raw_500hz/filename.npy'.
    """
    if path is None:
        return None
    try:
        rel = os.path.relpath(os.path.abspath(str(path)), RESAMPLE_BASE)
    except (TypeError, ValueError):
        return path
    rel = rel.replace(os.sep, "/")
    return f"/kaggle/input/datasets/muhffikkri/ecg-dataset/{rel}"


def resolve_path(path):
    """Manifest path -> absolute path (relative entries resolved vs BASE_DIR)."""
    if not isinstance(path, str) or path in ("", "nan", "None"):
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, *path.replace("\\", "/").split("/"))

# =========================================================

# DATA FOLDERS

# =========================================================

SUB_FOLDERS = {

# =====================================================
# ACTIVE SCHEME (100 Hz / 500 Hz) -- current research schedule
# =====================================================
# Cleaned (preprocessed) signal folders per dataset.
# Raw ("murni") folders hold the unprocessed wave-forms (length/tensor only).

# PTB-XL native 100 Hz literals -> 100 Hz
"ptbxl_raw_100hz":
    os.path.join(RESAMPLE_BASE, "ptbxl_raw_100hz"),

"ptbxl_clean_100hz":
    os.path.join(RESAMPLE_BASE, "ptbxl_clean_100hz"),

# PTB-XL native 500 Hz literals -> 500 Hz
"ptbxl_raw_500hz":
    os.path.join(RESAMPLE_BASE, "ptbxl_raw_500hz"),

"ptbxl_clean_500hz":
    os.path.join(RESAMPLE_BASE, "ptbxl_clean_500hz"),

# Chapman native 500 Hz literals -> 500 Hz
"chapman_raw_500hz":
    os.path.join(RESAMPLE_BASE, "chapman_raw_500hz"),

"chapman_clean_500hz":
    os.path.join(RESAMPLE_BASE, "chapman_clean_500hz"),

# Chapman down-sampled to 100 Hz
"chapman_raw_100hz":
    os.path.join(RESAMPLE_BASE, "chapman_raw_100hz"),

"chapman_clean_100hz":
    os.path.join(RESAMPLE_BASE, "chapman_clean_100hz"),

# =====================================================
# LEGACY / OPTIONAL (250 Hz scheme, deferred to the future)
# =====================================================

"E1_100_native":
    os.path.join(RESAMPLE_BASE, "exp_100_native"),

"E2_100_to_250":
    os.path.join(RESAMPLE_BASE, "exp_100_to_250"),

"E3_500_to_250":
    os.path.join(RESAMPLE_BASE, "exp_500_to_250"),

"E4_500_native":
    os.path.join(RESAMPLE_BASE, "exp_500_native"),

"E1_clean_100_native":
    os.path.join(RESAMPLE_BASE, "cleaned_100_native"),

"E2_clean_100_to_250":
    os.path.join(RESAMPLE_BASE, "cleaned_100_to_250"),

"E3_clean_500_to_250":
    os.path.join(RESAMPLE_BASE, "cleaned_500_to_250"),

"E4_clean_500_native":
    os.path.join(RESAMPLE_BASE, "cleaned_500_native"),

# Legacy Chapman 250 Hz external validation dataset (pre-rework)
"Chapman_clean_500_to_250":
    os.path.join(RESAMPLE_BASE, "chapman_clean_500_to_250")

}

# =========================================================
# FOLDER -> SAMPLING RATE (single source for input-shape resolution)
# =========================================================

FOLDER_FS = {
    # Active 100/500 Hz scheme
    "ptbxl_raw_100hz": 100,
    "ptbxl_clean_100hz": 100,
    "ptbxl_raw_500hz": 500,
    "ptbxl_clean_500hz": 500,
    "chapman_raw_100hz": 100,
    "chapman_clean_100hz": 100,
    "chapman_raw_500hz": 500,
    "chapman_clean_500hz": 500,
    # Legacy 250 Hz scheme
    "E1_100_native": 100,
    "E2_100_to_250": 250,
    "E3_500_to_250": 250,
    "E4_500_native": 500,
    "E1_clean_100_native": 100,
    "E2_clean_100_to_250": 250,
    "E3_clean_500_to_250": 250,
    "E4_clean_500_native": 500,
    "Chapman_clean_500_to_250": 250,
}


def folder_fs(folder_key):
    """Sampling rate of a SUB_FOLDERS key (defaults to 250 Hz)."""
    return FOLDER_FS.get(folder_key, 250)

# =========================================================

# AUTO CREATE DIRECTORIES

# =========================================================

for folder_path in SUB_FOLDERS.values():
    os.makedirs(folder_path, exist_ok=True)

# =========================================================

# CLASS LABELS

# =========================================================

# Canonical shared 4-class scheme. Single source of truth used by every
# consumer (training runners, evaluators, reporting scripts). Selecting
# LEGACY order compatibility is done in the loaders via LABEL_SCHEME.
TARGET_CLASSES = [
    'Normal',
    'AF',
    'Takikardia',
    'Bradikardia',
]

CLASS_NAMES = TARGET_CLASSES

NUM_CLASSES = len(TARGET_CLASSES)

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
