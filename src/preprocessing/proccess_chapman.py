# =====================================================================
# FILE 4: proccess_chapman.py
# CHAPMAN ECG PROCESSING + CROSS DATASET VALIDATION MANIFEST (100/500 Hz)
# =====================================================================
# Produces, per Chapman record:
#   - raw ("murni") tensors  : native 500 Hz and down-sampled 100 Hz
#   - cleaned tensors        : native 500 Hz and down-sampled 100 Hz
# Cleaning stages are driven by preprocessing.CLEANING_FLAGS
# (wavelet + median baseline + bandpass enabled, z-score disabled).
# The 250 Hz scheme is deferred to a future experiment.
# =====================================================================

import os
import sys
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
    "100hz": True,   # native 500 Hz down-sampled to 100 Hz
    "500hz": True,   # native 500 Hz
    "250hz": False,  # deferred to a future experiment
}


# =====================================================================
# CHAPMAN -> TARGET LABEL MAPPING
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
print(f"Active schemes: "
      f"{'100Hz ' if GENERATE_SCHEMES['100hz'] else ''}"
      f"{'500Hz ' if GENERATE_SCHEMES['500hz'] else ''}"
      f"{'250Hz ' if GENERATE_SCHEMES['250hz'] else ''}"
      f"| cleaning = {dsp.cleaning_pipeline_description()}")


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

        # NOTE: the mapped 4-class target is derived for the legacy /
        # cross-dataset scheme ONLY and does NOT filter the dataset. Every
        # Chapman record is stored so native all-class baselines
        # (LABEL_SCHEME="native") see the full label space; unmapped records
        # simply keep target_class=None while native_label still carries the
        # SNOMED code.

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

        paths = {}

        # =========================================================
        # CLEANING + RESAMPLING (100 Hz & 500 Hz, raw + cleaned)
        # =========================================================

        if GENERATE_SCHEMES["500hz"]:

            # Raw ("murni") native 500Hz tensor
            raw_500 = dsp.ensure_length(
                lead_3,
                cfg.TARGET_LEN[500]
            )

            raw_500_path = os.path.join(
                cfg.SUB_FOLDERS["chapman_raw_500hz"],
                base_filename
            )

            np.save(raw_500_path, raw_500.astype(np.float32))

            paths["chapman_raw_500hz"] = raw_500_path

            # Cleaned native 500Hz tensor
            clean_500 = dsp.ensure_length(
                dsp.advanced_cleaning_pipeline(
                    raw_signal=lead_3,
                    src_fs=src_fs,
                    target_fs=500.0
                ),
                cfg.TARGET_LEN[500]
            )

            clean_500_path = os.path.join(
                cfg.SUB_FOLDERS["chapman_clean_500hz"],
                base_filename
            )

            np.save(clean_500_path, clean_500.astype(np.float32))

            paths["chapman_clean_500hz"] = clean_500_path

        if GENERATE_SCHEMES["100hz"]:

            # Raw ("murni") down-sampled 100Hz tensor
            raw_100 = dsp.ensure_length(
                dsp.apply_poly_resample(lead_3, src_fs, 100.0),
                cfg.TARGET_LEN[100]
            )

            raw_100_path = os.path.join(
                cfg.SUB_FOLDERS["chapman_raw_100hz"],
                base_filename
            )

            np.save(raw_100_path, raw_100.astype(np.float32))

            paths["chapman_raw_100hz"] = raw_100_path

            # Cleaned down-sampled 100Hz tensor
            clean_100 = dsp.ensure_length(
                dsp.advanced_cleaning_pipeline(
                    raw_signal=lead_3,
                    src_fs=src_fs,
                    target_fs=100.0
                ),
                cfg.TARGET_LEN[100]
            )

            clean_100_path = os.path.join(
                cfg.SUB_FOLDERS["chapman_clean_100hz"],
                base_filename
            )

            np.save(clean_100_path, clean_100.astype(np.float32))

            paths["chapman_clean_100hz"] = clean_100_path

        if GENERATE_SCHEMES["250hz"]:

            # Deferred scheme: unified 250Hz legacy folder
            clean_250 = dsp.ensure_length(
                dsp.advanced_cleaning_pipeline(
                    raw_signal=lead_3,
                    src_fs=src_fs,
                    target_fs=250.0
                ),
                cfg.TARGET_LEN[250]
            )

            save_path = os.path.join(
                cfg.SUB_FOLDERS["Chapman_clean_500_to_250"],
                base_filename
            )

            np.save(save_path, clean_250.astype(np.float32))

            paths["Chapman_clean_500_to_250"] = save_path

        # =========================================================
        # HASHING (primary cleaned tensor: native 500Hz)
        # =========================================================

        md5_hash = None
        sha1_hash = None

        primary_clean = paths.get("chapman_clean_500hz")

        if primary_clean is not None:

            md5_hash, sha1_hash = compute_array_hash(
                np.load(primary_clean)
            )

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

            # Native (primary SNOMED-CT code) label: makes the manifest
            # self-contained for cleaned-only Kaggle runs with
            # LABEL_SCHEME="native" (drops the trainer dependency on the
            # raw .hea files).
            "native_label": str(dx_str).split(",")[0].strip(),

            # ---------------------------------------------
            # SHAPE
            # ---------------------------------------------
            "original_length": int(lead_3.shape[0]),
            "num_leads": int(lead_3.shape[1]),

            # ---------------------------------------------
            # SAMPLING
            # ---------------------------------------------
            "source_sampling_rate": float(src_fs),
            "target_fs_100hz": 100,
            "target_fs_500hz": 500,

            # ---------------------------------------------
            # PROCESSING
            # ---------------------------------------------
            "resample_method":
                "polyphase_fir",

            "cleaning_pipeline":
                dsp.cleaning_pipeline_description(),

            "pipeline_version":
                dsp.PIPELINE_VERSION,

            # ---------------------------------------------
            # HASH AUDIT
            # ---------------------------------------------
            "md5": md5_hash,
            "sha1": sha1_hash,

            # ---------------------------------------------
            # STORAGE (relative to BASE_DIR for portability)
            # ---------------------------------------------
            "path_chapman_raw_500hz":
                cfg.to_relative_path(paths.get("chapman_raw_500hz", None)),

            "path_chapman_clean_500hz":
                cfg.to_relative_path(paths.get("chapman_clean_500hz", None)),

            "path_chapman_raw_100hz":
                cfg.to_relative_path(paths.get("chapman_raw_100hz", None)),

            "path_chapman_clean_100hz":
                cfg.to_relative_path(paths.get("chapman_clean_100hz", None))
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

print("\n✓ Chapman dipakai untuk mapping 4-kelas (cross-dataset) dan "
      "native (SNOMED-CT).")
print("✓ Output: raw & cleaned 500 Hz (native) dan 100 Hz (downsampled).")
print("✓ Metadata audit lengkap telah disimpan.")