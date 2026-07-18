# =====================================================================
# FILE 4: process_chapman.py
# CHAPMAN ECG PROCESSING + CROSS DATASET VALIDATION MANIFEST
# =====================================================================

import os
import sys
import hashlib
import traceback

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
# CHAPMAN → TARGET LABEL MAPPING
# =====================================================================

chapman_to_target_mapping = label_cfg.CHAPMAN_TO_TARGET_MAPPING


# =====================================================================
# LABEL PRIORITY MAPPING
# =====================================================================

def map_chapman_classes(dx_string):

    codes = dx_string.split(',')

    matched_labels = set()

    for code in codes:

        code = code.strip()

        if code in chapman_to_target_mapping:
            matched_labels.add(
                chapman_to_target_mapping[code]
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
# HASH UTILS
# =====================================================================

def compute_array_hash(arr):

    arr_bytes = arr.tobytes()

    md5_hash = hashlib.md5(arr_bytes).hexdigest()
    sha1_hash = hashlib.sha1(arr_bytes).hexdigest()

    return md5_hash, sha1_hash


# =====================================================================
# MAIN PROCESS
# =====================================================================

print("=" * 70)
print("CHAPMAN ECG PROCESSING")
print("=" * 70)

print("--> Crawling Chapman WFDBRecords...")

hea_files = []

for root, _, files in os.walk(cfg.CHAPMAN_RECS):

    for file in files:

        if file.endswith(".hea"):

            hea_files.append(
                os.path.join(root, file)
            )

print(f"--> Total .hea ditemukan : {len(hea_files)}")


# =====================================================================
# STORAGE
# =====================================================================

manifest_records = []

failed_files = []


# =====================================================================
# PROCESS LOOP
# =====================================================================

for hea_path in tqdm(hea_files, desc="Processing Chapman"):

    try:

        # =========================================================
        # EXTRACT DX
        # =========================================================

        dx_str = ""

        with open(hea_path, "r") as f:

            for line in f:

                if line.startswith("#Dx:"):

                    dx_str = line.split("#Dx:")[1].strip()
                    break

        if dx_str == "":
            continue

        # =========================================================
        # LABEL MAPPING
        # =========================================================

        target_cls = map_chapman_classes(dx_str)

        if target_cls is None:
            continue

        # =========================================================
        # RECORD PATH
        # =========================================================

        record_path = hea_path.replace(".hea", "")

        file_id = os.path.basename(record_path)

        base_filename = f"chap_{file_id}.npy"

        # =========================================================
        # LOAD SIGNAL
        # =========================================================

        raw_signal, meta = wfdb.rdsamp(record_path)

        src_fs = float(meta["fs"])

        # =========================================================
        # LEAD SELECTION
        # =========================================================

        lead_3 = raw_signal[:, cfg.LEAD_INDICES]

        # =========================================================
        # CLEANING + RESAMPLING
        # OUTPUT ABSOLUTE = 250Hz
        # =========================================================

        cleaned_250 = dsp.advanced_cleaning_pipeline(
            raw_signal=lead_3,
            src_fs=src_fs,
            target_fs=250.0
        )

        # =========================================================
        # FIXED LENGTH
        # =========================================================

        final_250 = dsp.ensure_length(
            cleaned_250,
            cfg.TARGET_LEN[250]
        )

        final_250 = final_250.astype(np.float32)

        # =========================================================
        # SAVE SIGNAL
        # =========================================================

        save_path = os.path.join(
            cfg.SUB_FOLDERS["Chapman_clean_500_to_250"],
            base_filename
        )

        np.save(save_path, final_250)

        # =========================================================
        # HASHING
        # =========================================================

        md5_hash, sha1_hash = compute_array_hash(final_250)

        # =========================================================
        # MANIFEST RECORD
        # =========================================================

        manifest_records.append({

            # ---------------------------------------------
            # IDENTIFIER
            # ---------------------------------------------
            "filename_npy": base_filename,
            "patient_id": file_id,
            "ecg_id": file_id,

            # ---------------------------------------------
            # DATASET
            # ---------------------------------------------
            "source_dataset": "Chapman",

            # ---------------------------------------------
            # LABEL
            # ---------------------------------------------
            "target_class": target_cls,
            "diagnostic_string": dx_str,

            # ---------------------------------------------
            # SHAPE
            # ---------------------------------------------
            "original_length": int(lead_3.shape[0]),
            "final_length": int(final_250.shape[0]),
            "num_leads": int(final_250.shape[1]),

            # ---------------------------------------------
            # SAMPLING
            # ---------------------------------------------
            "source_sampling_rate": float(src_fs),
            "target_sampling_rate": 250.0,

            # ---------------------------------------------
            # PROCESSING
            # ---------------------------------------------
            "resample_method":
                "polyphase_fir_smart_router",

            "cleaning_pipeline":
                "wavelet_db4+median_baseline+"
                "bandpass+zscore_clip",

            "pipeline_version":
                "v5.0_rebuilt",

            # ---------------------------------------------
            # HASH AUDIT
            # ---------------------------------------------
            "md5": md5_hash,
            "sha1": sha1_hash,

            # ---------------------------------------------
            # STORAGE
            # ---------------------------------------------
            "save_path": save_path
        })

    except Exception as e:

        failed_files.append({
            "file": hea_path,
            "error": str(e)
        })

        continue


# =====================================================================
# SAVE MANIFEST
# =====================================================================

manifest_df = pd.DataFrame(manifest_records)

manifest_path = os.path.join(
    cfg.RESAMPLE_BASE,
    "manifest_chapman.csv"
)

manifest_df.to_csv(
    manifest_path,
    index=False
)


# =====================================================================
# SAVE FAILED LOG
# =====================================================================

failed_path = os.path.join(
    cfg.RESAMPLE_BASE,
    "chapman_failed_records.csv"
)

pd.DataFrame(failed_files).to_csv(
    failed_path,
    index=False
)


# =====================================================================
# SUMMARY
# =====================================================================

print("\n" + "=" * 70)
print("CHAPMAN PROCESSING FINISHED")
print("=" * 70)

print(f"Total berhasil diproses : {len(manifest_df)}")
print(f"Total gagal             : {len(failed_files)}")

print("\nDistribusi kelas:")

print(
    manifest_df["target_class"]
    .value_counts()
)

print(f"\nManifest disimpan di:")
print(manifest_path)

print(f"\nFailed log disimpan di:")
print(failed_path)

print("\n✓ Chapman digunakan hanya untuk cross-dataset validation.")
print("✓ Tidak digunakan pada training PTB-XL.")
print("✓ Metadata audit lengkap telah disimpan.")