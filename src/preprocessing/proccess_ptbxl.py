# =====================================================================
# FILE 3: process_ptbxl.py
# PTB-XL FULL PROCESSING + REPRODUCIBLE MANIFEST SYSTEM
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

    if target_cls is None:
        continue

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
    # 100Hz PROCESSING
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

            # ========================================================
            # E1 RAW 100Hz
            # ========================================================

            e1_raw = dsp.ensure_length(
                lead_3_100,
                cfg.TARGET_LEN[100]
            )

            e1_raw_path = os.path.join(
                cfg.SUB_FOLDERS["E1_100_native"],
                base_filename
            )

            save_numpy(
                e1_raw_path,
                e1_raw
            )

            paths["E1_raw"] = e1_raw_path

            # ========================================================
            # E1 CLEAN 100Hz
            # ========================================================

            clean_native_100 = dsp.advanced_cleaning_pipeline(
                raw_signal=lead_3_100,
                src_fs=100.0,
                target_fs=100.0
            )

            e1_clean = dsp.ensure_length(
                clean_native_100,
                cfg.TARGET_LEN[100]
            )

            e1_clean_path = os.path.join(
                cfg.SUB_FOLDERS["E1_clean_100_native"],
                base_filename
            )

            save_numpy(
                e1_clean_path,
                e1_clean
            )

            paths["E1_clean"] = e1_clean_path

            # ========================================================
            # E2 RAW 100 -> 250
            # ========================================================

            e2_raw_sig = dsp.apply_poly_resample(
                lead_3_100,
                100.0,
                250.0
            )

            e2_raw = dsp.ensure_length(
                e2_raw_sig,
                cfg.TARGET_LEN[250]
            )

            e2_raw_path = os.path.join(
                cfg.SUB_FOLDERS["E2_100_to_250"],
                base_filename
            )

            save_numpy(
                e2_raw_path,
                e2_raw
            )

            paths["E2_raw"] = e2_raw_path

            # ========================================================
            # E2 CLEAN 100 -> 250
            # ========================================================

            clean_upsampled_250 = dsp.advanced_cleaning_pipeline(
                raw_signal=lead_3_100,
                src_fs=100.0,
                target_fs=250.0
            )

            e2_clean = dsp.ensure_length(
                clean_upsampled_250,
                cfg.TARGET_LEN[250]
            )

            e2_clean_path = os.path.join(
                cfg.SUB_FOLDERS["E2_clean_100_to_250"],
                base_filename
            )

            save_numpy(
                e2_clean_path,
                e2_clean
            )

            paths["E2_clean"] = e2_clean_path

        except Exception as e:

            print(f"[100Hz ERROR] {base_filename} -> {e}")

    # ================================================================
    # 500Hz PROCESSING
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

            # ========================================================
            # E4 RAW 500Hz
            # ========================================================

            e4_raw = dsp.ensure_length(
                lead_3_500,
                cfg.TARGET_LEN[500]
            )

            e4_raw_path = os.path.join(
                cfg.SUB_FOLDERS["E4_500_native"],
                base_filename
            )

            save_numpy(
                e4_raw_path,
                e4_raw
            )

            paths["E4_raw"] = e4_raw_path

            # ========================================================
            # E4 CLEAN 500Hz
            # ========================================================

            clean_native_500 = dsp.advanced_cleaning_pipeline(
                raw_signal=lead_3_500,
                src_fs=500.0,
                target_fs=500.0
            )

            e4_clean = dsp.ensure_length(
                clean_native_500,
                cfg.TARGET_LEN[500]
            )

            e4_clean_path = os.path.join(
                cfg.SUB_FOLDERS["E4_clean_500_native"],
                base_filename
            )

            save_numpy(
                e4_clean_path,
                e4_clean
            )

            paths["E4_clean"] = e4_clean_path

            # ========================================================
            # E3 RAW 500 -> 250
            # ========================================================

            e3_raw_sig = dsp.apply_poly_resample(
                lead_3_500,
                500.0,
                250.0
            )

            e3_raw = dsp.ensure_length(
                e3_raw_sig,
                cfg.TARGET_LEN[250]
            )

            e3_raw_path = os.path.join(
                cfg.SUB_FOLDERS["E3_500_to_250"],
                base_filename
            )

            save_numpy(
                e3_raw_path,
                e3_raw
            )

            paths["E3_raw"] = e3_raw_path

            # ========================================================
            # E3 CLEAN 500 -> 250
            # ========================================================

            clean_downsampled_250 = dsp.advanced_cleaning_pipeline(
                raw_signal=lead_3_500,
                src_fs=500.0,
                target_fs=250.0
            )

            e3_clean = dsp.ensure_length(
                clean_downsampled_250,
                cfg.TARGET_LEN[250]
            )

            e3_clean_path = os.path.join(
                cfg.SUB_FOLDERS["E3_clean_500_to_250"],
                base_filename
            )

            save_numpy(
                e3_clean_path,
                e3_clean
            )

            paths["E3_clean"] = e3_clean_path

            # ========================================================
            # E5 REFERENCE COPY
            # ========================================================

            e5_reference_path = os.path.join(
                cfg.SUB_FOLDERS["E5_clean_500_to_250_reference"],
                base_filename
            )

            save_numpy(
                e5_reference_path,
                e3_clean
            )

            paths["E5_reference"] = e5_reference_path

        except Exception as e:

            print(f"[500Hz ERROR] {base_filename} -> {e}")

    # ================================================================
    # HASH COMPUTATION
    # ================================================================

    clean_md5 = None
    clean_sha1 = None

    if "E3_clean" in paths:

        clean_array = np.load(
            paths["E3_clean"]
        )

        clean_md5 = compute_md5(
            clean_array
        )

        clean_sha1 = compute_sha1(
            clean_array
        )

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

        "target_fs_100hz": 100,

        "target_fs_250hz": 250,

        "target_fs_500hz": 500,

        # ============================================================
        # PIPELINE
        # ============================================================

        "resample_method":
            "polyphase_fir",

        "cleaning_pipeline":
            "wavelet_db4"
            "+median_baseline"
            "+bandpass"
            "+zscore_clip",

        "pipeline_version":
            "v5.1_full_multiresolution",

        # ============================================================
        # HASH
        # ============================================================

        "clean_md5":
            clean_md5,

        "clean_sha1":
            clean_sha1,

        # ============================================================
        # PATHS
        # ============================================================

        "path_e1_raw":
            paths.get("E1_raw", None),

        "path_e1_clean":
            paths.get("E1_clean", None),

        "path_e2_raw":
            paths.get("E2_raw", None),

        "path_e2_clean":
            paths.get("E2_clean", None),

        "path_e3_raw":
            paths.get("E3_raw", None),

        "path_e3_clean":
            paths.get("E3_clean", None),

        "path_e4_raw":
            paths.get("E4_raw", None),

        "path_e4_clean":
            paths.get("E4_clean", None),

        "path_e5_reference":
            paths.get("E5_reference", None)
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