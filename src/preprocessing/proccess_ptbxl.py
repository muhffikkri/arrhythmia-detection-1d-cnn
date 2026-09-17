# =====================================================================
# FILE 3: proccess_ptbxl.py
# PTB-XL FULL PROCESSING + REPRODUCIBLE MANIFEST SYSTEM (100/500 Hz)
# =====================================================================
# Produces, per PTB-XL record:
#   - raw ("murni") tensors  : 100 Hz and 500 Hz
#   - cleaned tensors        : 100 Hz and 500 Hz
# Cleaning stages are driven by preprocessing.CLEANING_FLAGS
# (wavelet + median baseline + bandpass enabled, z-score disabled).
# The 250 Hz scheme is deferred to a future experiment (GENERATE_SCHEMES).
# =====================================================================

import os
import sys
import ast
import hashlib

# Add the project root directory to the python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import wfdb
import numpy as np
import pandas as pd

from tqdm import tqdm

from src.config import config as cfg
from src.config import config_labels as label_cfg
from src.preprocessing import preprocessing as dsp


# =====================================================================
# OUTPUT SCHEME (active = 100/500 Hz, 250 Hz deferred)
# =====================================================================

GENERATE_SCHEMES = {
    "100hz": True,
    "500hz": True,
    "250hz": False,  # deferred to a future experiment
}


# =====================================================================
# PTB-XL LABEL MAPPING
# =====================================================================

ptbxl_to_target_mapping = label_cfg.PTBXL_TO_TARGET_MAPPING


# =====================================================================
# PRIORITY CLASS RESOLUTION
# =====================================================================

def map_ptbxl_classes(diagnostic_dict):

    matched_labels = set()

    for key in diagnostic_dict.keys():

        if key in ptbxl_to_target_mapping:
            matched_labels.add(
                ptbxl_to_target_mapping[key]
            )

    # PRIORITY OVERRIDE

    if 'AF' in matched_labels:
        return 'AF'

    if 'Takikardia' in matched_labels:
        return 'Takikardia'

    if 'Bradikardia' in matched_labels:
        return 'Bradikardia'

    if 'Others' in matched_labels:
        return 'Others'

    if 'Normal' in matched_labels:
        return 'Normal'

    return None


# =====================================================================
# HASHING UTILITIES
# =====================================================================

def compute_md5(array):
    return hashlib.md5(
        array.tobytes()
    ).hexdigest()


def compute_sha1(array):
    return hashlib.sha1(
        array.tobytes()
    ).hexdigest()


# =====================================================================
# SAFE SAVE
# =====================================================================

def save_numpy(path, array):

    array = np.nan_to_num(
        array,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)

    np.save(path, array)


# =====================================================================
# LOAD PTB-XL METADATA
# =====================================================================

print("=" * 70)
print("LOADING PTB-XL DATABASE")
print("=" * 70)

df = pd.read_csv(
    cfg.PTBXL_CSV,
    index_col=0
)

df['scp_codes'] = df['scp_codes'].apply(
    lambda x: ast.literal_eval(x)
)

print(f"Total Records : {len(df)}")
print(f"Active schemes: "
      f"{'100Hz ' if GENERATE_SCHEMES['100hz'] else ''}"
      f"{'500Hz ' if GENERATE_SCHEMES['500hz'] else ''}"
      f"{'250Hz ' if GENERATE_SCHEMES['250hz'] else ''}"
      f"| cleaning = {dsp.cleaning_pipeline_description()}")


# =====================================================================
# MANIFEST STORAGE
# =====================================================================

manifest_records = []


# =====================================================================
# MAIN ITERATION
# =====================================================================

for ecg_id, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Processing PTB-XL"
):

    # ================================================================
    # TARGET CLASS
    # ================================================================

    target_cls = map_ptbxl_classes(
        row['scp_codes']
    )

    # NOTE: the mapped 4-class target is derived for the legacy /
    # cross-dataset scheme ONLY and does NOT filter the dataset. Every
    # PTB-XL record is stored so native all-class baselines
    # (LABEL_SCHEME="native") see the full label space; unmapped records
    # simply keep target_class=None while native_label still carries the
    # SCP code.

    # Native (highest-confidence SCP code) label: makes the manifest
    # self-contained so LABEL_SCHEME="native" does not need the raw
    # ptbxl_database.csv at training time (e.g. cleaned-only Kaggle runs).
    primary_scp = None

    if isinstance(row['scp_codes'], dict) and row['scp_codes']:

        primary_scp = str(
            max(row['scp_codes'], key=row['scp_codes'].get)
        )

    # ================================================================
    # CORE METADATA
    # ================================================================

    patient_id = int(row['patient_id'])

    strat_fold = int(row['strat_fold'])

    base_filename = f"ptb_{ecg_id:05d}.npy"

    # ================================================================
    # STORAGE PATHS
    # ================================================================

    paths = {}

    # ================================================================
    # ORIGINAL LENGTHS
    # ================================================================

    orig_len_100 = 0
    orig_len_500 = 0

    # ================================================================
    # 100Hz PROCESSING (raw + cleaned)
    # ================================================================

    path_100 = os.path.join(
        cfg.DATASET_DIR,
        "PTBXL",
        row['filename_lr']
    )

    if os.path.exists(path_100 + ".dat"):

        try:
            raw_signal_100, meta_100 = wfdb.rdsamp(path_100)

            lead_3_100 = raw_signal_100[:, cfg.LEAD_INDICES]

            orig_len_100 = int(
                lead_3_100.shape[0]
            )

            if GENERATE_SCHEMES["100hz"]:

                # Raw ("murni") 100Hz tensor
                raw_100 = dsp.ensure_length(
                    lead_3_100,
                    cfg.TARGET_LEN[100]
                )

                raw_100_path = os.path.join(
                    cfg.SUB_FOLDERS["ptbxl_raw_100hz"],
                    base_filename
                )

                save_numpy(raw_100_path, raw_100)

                paths["ptbxl_raw_100hz"] = raw_100_path

                # Cleaned 100Hz tensor
                clean_native_100 = dsp.advanced_cleaning_pipeline(
                    raw_signal=lead_3_100,
                    src_fs=100.0,
                    target_fs=100.0
                )

                clean_100 = dsp.ensure_length(
                    clean_native_100,
                    cfg.TARGET_LEN[100]
                )

                clean_100_path = os.path.join(
                    cfg.SUB_FOLDERS["ptbxl_clean_100hz"],
                    base_filename
                )

                save_numpy(clean_100_path, clean_100)

                paths["ptbxl_clean_100hz"] = clean_100_path

            if GENERATE_SCHEMES["250hz"]:

                # Deferred scheme: 100Hz -> 250Hz upsampled (legacy folders)
                raw_up = dsp.ensure_length(
                    dsp.apply_poly_resample(lead_3_100, 100.0, 250.0),
                    cfg.TARGET_LEN[250]
                )

                raw_up_path = os.path.join(
                    cfg.SUB_FOLDERS["E2_100_to_250"],
                    base_filename
                )

                save_numpy(raw_up_path, raw_up)

                clean_up = dsp.ensure_length(
                    dsp.advanced_cleaning_pipeline(
                        raw_signal=lead_3_100,
                        src_fs=100.0,
                        target_fs=250.0
                    ),
                    cfg.TARGET_LEN[250]
                )

                clean_up_path = os.path.join(
                    cfg.SUB_FOLDERS["E2_clean_100_to_250"],
                    base_filename
                )

                save_numpy(clean_up_path, clean_up)

        except Exception as e:

            print(f"[100Hz ERROR] {base_filename} -> {e}")

    # ================================================================
    # 500Hz PROCESSING (raw + cleaned)
    # ================================================================

    path_500 = os.path.join(
        cfg.DATASET_DIR,
        "PTBXL",
        row['filename_hr']
    )

    if os.path.exists(path_500 + ".dat"):

        try:

            raw_signal_500, meta_500 = wfdb.rdsamp(path_500)

            lead_3_500 = raw_signal_500[:, cfg.LEAD_INDICES]

            orig_len_500 = int(
                lead_3_500.shape[0]
            )

            if GENERATE_SCHEMES["500hz"]:

                # Raw ("murni") 500Hz tensor
                raw_500 = dsp.ensure_length(
                    lead_3_500,
                    cfg.TARGET_LEN[500]
                )

                raw_500_path = os.path.join(
                    cfg.SUB_FOLDERS["ptbxl_raw_500hz"],
                    base_filename
                )

                save_numpy(raw_500_path, raw_500)

                paths["ptbxl_raw_500hz"] = raw_500_path

                # Cleaned 500Hz tensor
                clean_native_500 = dsp.advanced_cleaning_pipeline(
                    raw_signal=lead_3_500,
                    src_fs=500.0,
                    target_fs=500.0
                )

                clean_500 = dsp.ensure_length(
                    clean_native_500,
                    cfg.TARGET_LEN[500]
                )

                clean_500_path = os.path.join(
                    cfg.SUB_FOLDERS["ptbxl_clean_500hz"],
                    base_filename
                )

                save_numpy(clean_500_path, clean_500)

                paths["ptbxl_clean_500hz"] = clean_500_path

            if GENERATE_SCHEMES["250hz"]:

                # Deferred scheme: 500Hz -> 250Hz downsampled (legacy folders)
                raw_dn = dsp.ensure_length(
                    dsp.apply_poly_resample(lead_3_500, 500.0, 250.0),
                    cfg.TARGET_LEN[250]
                )

                raw_dn_path = os.path.join(
                    cfg.SUB_FOLDERS["E3_500_to_250"],
                    base_filename
                )

                save_numpy(raw_dn_path, raw_dn)

                clean_dn = dsp.ensure_length(
                    dsp.advanced_cleaning_pipeline(
                        raw_signal=lead_3_500,
                        src_fs=500.0,
                        target_fs=250.0
                    ),
                    cfg.TARGET_LEN[250]
                )

                clean_dn_path = os.path.join(
                    cfg.SUB_FOLDERS["E3_clean_500_to_250"],
                    base_filename
                )

                save_numpy(clean_dn_path, clean_dn)

        except Exception as e:

            print(f"[500Hz ERROR] {base_filename} -> {e}")

    # ================================================================
    # HASH COMPUTATION (primary cleaned tensor: 500Hz when present)
    # ================================================================

    clean_md5 = None
    clean_sha1 = None

    primary_clean = (
        paths.get("ptbxl_clean_500hz")
        or paths.get("ptbxl_clean_100hz")
    )

    if primary_clean is not None:

        clean_array = np.load(primary_clean)

        clean_md5 = compute_md5(clean_array)
        clean_sha1 = compute_sha1(clean_array)

    # ================================================================
    # MANIFEST RECORD
    # ================================================================

    manifest_records.append({

        # ============================================================
        # IDENTITY
        # ============================================================

        "filename_npy": base_filename,

        "patient_id": patient_id,

        "ecg_id": int(ecg_id),

        "source_dataset": "PTBXL",

        # ============================================================
        # TARGET
        # ============================================================

        "target_class": target_cls,

        "native_label": primary_scp,

        # ============================================================
        # SPLIT
        # ============================================================

        "strat_fold": strat_fold,

        # ============================================================
        # SIGNAL
        # ============================================================

        "num_leads": len(cfg.LEAD_INDICES),

        "lead_names": ",".join(cfg.LEAD_NAMES),

        "original_len_100hz": orig_len_100,

        "original_len_500hz": orig_len_500,

        # ============================================================
        # PIPELINE
        # ============================================================

        "resample_method":
            "polyphase_fir",

        "cleaning_pipeline":
            dsp.cleaning_pipeline_description(),

        "pipeline_version":
            dsp.PIPELINE_VERSION,

        # ============================================================
        # HASH
        # ============================================================

        "clean_md5":
            clean_md5,

        "clean_sha1":
            clean_sha1,

        # ============================================================
        # PATHS (relative to BASE_DIR for portability, e.g. Kaggle)
        # ============================================================

        "path_ptbxl_raw_100hz":
            cfg.to_relative_path(paths.get("ptbxl_raw_100hz", None)),

        "path_ptbxl_clean_100hz":
            cfg.to_relative_path(paths.get("ptbxl_clean_100hz", None)),

        "path_ptbxl_raw_500hz":
            cfg.to_relative_path(paths.get("ptbxl_raw_500hz", None)),

        "path_ptbxl_clean_500hz":
            cfg.to_relative_path(paths.get("ptbxl_clean_500hz", None))
    })


# =====================================================================
# SAVE MANIFEST
# =====================================================================

manifest_df = pd.DataFrame(
    manifest_records
)

manifest_path = os.path.join(
    cfg.RESAMPLE_BASE,
    "manifest_ptbxl.csv"
)

manifest_df.to_csv(
    manifest_path,
    index=False
)

# =====================================================================
# SUMMARY
# =====================================================================

print("\n" + "=" * 70)
print("PTB-XL PROCESSING FINISHED")
print("=" * 70)

print(f"Total Processed : {len(manifest_df)}")

print(f"Manifest Saved  : {manifest_path}")

print("\nClass Distribution:")
print(
    manifest_df['target_class']
    .value_counts()
)

print("\nDone.")