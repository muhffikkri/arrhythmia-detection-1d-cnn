# =====================================================================
# FILE: eda_quantitative_audit.py
# FINAL VERSION — QUANTITATIVE EDA & PREPROCESSING AUDIT
# Compatible with current config.py structure
# =====================================================================

import os
import sys
import warnings

# Add the project root directory to the python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import welch
from scipy.stats import pearsonr
from fastdtw import fastdtw
from tqdm import tqdm

import neurokit2 as nk

from src.config import config as cfg

warnings.filterwarnings("ignore")

# =====================================================================
# OUTPUT DIRECTORY
# =====================================================================

EDA_DIR = os.path.join(
    cfg.BASE_DIR,
    "output",
    "eda_quantitative_audit"
)

os.makedirs(EDA_DIR, exist_ok=True)

# =====================================================================
# SETTINGS
# =====================================================================

FS = 250.0
SAMPLE_LIMIT_PER_CLASS = 50

# =====================================================================
# LOAD MANIFEST
# =====================================================================

print("=" * 80)
print("QUANTITATIVE EDA & PREPROCESSING AUDIT")
print("=" * 80)

manifest_path = os.path.join(
    cfg.RESAMPLE_BASE,
    "manifest_ptbxl.csv"
)

df_manifest = pd.read_csv(manifest_path)

# =====================================================================
# PIPELINE DEFINITIONS
# =====================================================================

PIPELINES = {

    "100Hz_to_250Hz": {
        "raw_folder": "E2_100_to_250",
        "clean_folder": "E2_clean_100_to_250"
    },

    "500Hz_to_250Hz": {
        "raw_folder": "E3_500_to_250",
        "clean_folder": "E3_clean_500_to_250"
    }
}

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def compute_pseudo_snr(clean_signal, removed_noise):

    signal_power = np.mean(clean_signal ** 2)
    noise_power = np.mean(removed_noise ** 2)

    if noise_power < 1e-12:
        return 100.0

    return 10 * np.log10(signal_power / noise_power)


def compute_frequency_metrics(signal, fs=250.0):

    nperseg = min(len(signal), int(fs * 2))

    freqs, psd = welch(
        signal,
        fs=fs,
        nperseg=nperseg
    )

    total_energy = np.sum(psd)

    if total_energy < 1e-12:

        return {
            "BW_Energy": 0.0,
            "HF_Energy": 0.0,
            "Powerline_Energy": 0.0,
            "Dominant_Freq": np.nan,
            "Spectral_Entropy": np.nan
        }

    # =========================================================
    # BASELINE WANDER (<0.7Hz)
    # =========================================================

    idx_bw = np.where(freqs <= 0.7)[0]

    bw_energy = (
        np.trapezoid(psd[idx_bw], freqs[idx_bw])
        if len(idx_bw) > 1 else 0.0
    )

    # =========================================================
    # POWERLINE (49-51Hz)
    # =========================================================

    idx_pl = np.where(
        (freqs >= 49) & (freqs <= 51)
    )[0]

    pl_energy = (
        np.trapezoid(psd[idx_pl], freqs[idx_pl])
        if len(idx_pl) > 1 else 0.0
    )

    # =========================================================
    # HIGH FREQUENCY (>40Hz)
    # =========================================================

    idx_hf = np.where(freqs >= 40)[0]

    hf_energy = (
        np.trapezoid(psd[idx_hf], freqs[idx_hf])
        if len(idx_hf) > 1 else 0.0
    )

    dominant_freq = freqs[np.argmax(psd)]

    psd_norm = psd / total_energy
    psd_norm = psd_norm[psd_norm > 0]

    spectral_entropy = -np.sum(
        psd_norm * np.log2(psd_norm)
    )

    return {
        "BW_Energy": bw_energy,
        "HF_Energy": hf_energy,
        "Powerline_Energy": pl_energy,
        "Dominant_Freq": dominant_freq,
        "Spectral_Entropy": spectral_entropy
    }


def compute_hrv_metrics(signal, fs=250.0):

    metrics = {
        "Heart_Rate_BPM": np.nan,
        "RR_Mean": np.nan,
        "RR_STD": np.nan,
        "RR_RMSSD": np.nan,
        "RPeak_Count": 0,
        "RPeak_Success": False
    }

    try:

        processed, info = nk.ecg_process(
            signal,
            sampling_rate=fs
        )

        rpeaks = info["ECG_R_Peaks"]

        metrics["RPeak_Count"] = len(rpeaks)

        if len(rpeaks) > 1:

            rr = np.diff(rpeaks) / fs

            metrics["RR_Mean"] = np.mean(rr)
            metrics["RR_STD"] = np.std(rr)

            metrics["RR_RMSSD"] = np.sqrt(
                np.mean(np.diff(rr) ** 2)
            )

        if "ECG_Rate" in processed:

            metrics["Heart_Rate_BPM"] = np.mean(
                processed["ECG_Rate"]
            )

        metrics["RPeak_Success"] = True

    except Exception:
        pass

    return metrics


def compute_morphology_metrics(raw_sig, clean_sig):

    metrics = {}

    # =========================================================
    # PEARSON CORRELATION
    # =========================================================

    try:

        corr, _ = pearsonr(
            raw_sig,
            clean_sig
        )

        metrics["Pearson_Correlation"] = corr

    except Exception:

        metrics["Pearson_Correlation"] = np.nan

    # =========================================================
    # DTW DISTANCE
    # =========================================================

    try:

        distance, _ = fastdtw(
            raw_sig,
            clean_sig,
            dist=lambda x, y: abs(x - y)
        )

        metrics["DTW_Distance"] = distance

    except Exception:

        metrics["DTW_Distance"] = np.nan

    # =========================================================
    # PSEUDO SNR
    # =========================================================

    removed_noise = raw_sig - clean_sig

    metrics["Pseudo_SNR_dB"] = compute_pseudo_snr(
        clean_sig,
        removed_noise
    )

    return metrics


def audit_single_signal(raw_signal, clean_signal):

    metrics = {}

    # =========================================================
    # BASIC STATISTICS
    # =========================================================

    metrics["Mean"] = np.mean(clean_signal)
    metrics["Std"] = np.std(clean_signal)

    # =========================================================
    # HRV METRICS
    # =========================================================

    metrics.update(
        compute_hrv_metrics(
            clean_signal,
            fs=FS
        )
    )

    # =========================================================
    # FREQUENCY DOMAIN
    # =========================================================

    raw_freq = compute_frequency_metrics(
        raw_signal,
        fs=FS
    )

    clean_freq = compute_frequency_metrics(
        clean_signal,
        fs=FS
    )

    metrics["BW_Energy_Reduction"] = (
        raw_freq["BW_Energy"] -
        clean_freq["BW_Energy"]
    )

    metrics["HF_Energy_Reduction"] = (
        raw_freq["HF_Energy"] -
        clean_freq["HF_Energy"]
    )

    metrics["Powerline_Reduction"] = (
        raw_freq["Powerline_Energy"] -
        clean_freq["Powerline_Energy"]
    )

    metrics["Dominant_Frequency"] = (
        clean_freq["Dominant_Freq"]
    )

    metrics["Spectral_Entropy"] = (
        clean_freq["Spectral_Entropy"]
    )

    # =========================================================
    # MORPHOLOGY PRESERVATION
    # =========================================================

    metrics.update(
        compute_morphology_metrics(
            raw_signal,
            clean_signal
        )
    )

    return metrics

# =====================================================================
# MAIN AUDIT LOOP
# =====================================================================

audit_results = []

for pipeline_name, folders in PIPELINES.items():

    print(f"\n[PIPELINE] {pipeline_name}")

    raw_dir = cfg.SUB_FOLDERS[
        folders["raw_folder"]
    ]

    clean_dir = cfg.SUB_FOLDERS[
        folders["clean_folder"]
    ]

    for cls_name in cfg.CLASS_NAMES:

        print(f"   ↳ Auditing Class: {cls_name}")

        class_df = df_manifest[
            df_manifest["target_class"] == cls_name
        ].head(SAMPLE_LIMIT_PER_CLASS)

        for _, row in tqdm(
            class_df.iterrows(),
            total=len(class_df),
            leave=False
        ):

            filename = row["filename_npy"]

            raw_path = os.path.join(
                raw_dir,
                filename
            )

            clean_path = os.path.join(
                clean_dir,
                filename
            )

            if (
                not os.path.exists(raw_path)
                or
                not os.path.exists(clean_path)
            ):
                continue

            try:

                raw_tensor = np.load(raw_path)
                clean_tensor = np.load(clean_path)

                # Lead II as reference
                raw_sig = raw_tensor[:, 1]
                clean_sig = clean_tensor[:, 1]

                metrics = audit_single_signal(
                    raw_sig,
                    clean_sig
                )

                metrics["Pipeline"] = pipeline_name
                metrics["Target_Class"] = cls_name
                metrics["Filename"] = filename

                audit_results.append(metrics)

            except Exception as e:

                print(
                    f"Error processing {filename}: {e}"
                )

# =====================================================================
# SAVE CSV
# =====================================================================

df_audit = pd.DataFrame(audit_results)

csv_path = os.path.join(
    EDA_DIR,
    "quantitative_audit_metrics.csv"
)

df_audit.to_csv(
    csv_path,
    index=False
)

print(f"\nAudit CSV saved: {csv_path}")

# =====================================================================
# VISUALIZATION SETTINGS
# =====================================================================

print("\n--> Rendering morphology preservation plots...")

sns.set_theme(
    style="whitegrid"
)

# =====================================================================
# 1. PEARSON CORRELATION
# =====================================================================

df_corr = df_audit.dropna(
    subset=["Pearson_Correlation"]
)

if len(df_corr) > 0:

    plt.figure(figsize=(12, 6))

    sns.boxplot(
        data=df_corr,
        x="Pipeline",
        y="Pearson_Correlation",
        hue="Target_Class"
    )

    plt.title(
        "Morphology Preservation — Pearson Correlation",
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EDA_DIR,
            "plot_1_pearson_correlation.png"
        ),
        dpi=300
    )

    plt.close()

# =====================================================================
# 2. DTW DISTANCE
# =====================================================================

df_dtw = df_audit.dropna(
    subset=["DTW_Distance"]
)

if len(df_dtw) > 0:

    plt.figure(figsize=(12, 6))

    sns.boxplot(
        data=df_dtw,
        x="Pipeline",
        y="DTW_Distance",
        hue="Target_Class"
    )

    plt.title(
        "Nonlinear Morphology Deformation — DTW Distance",
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EDA_DIR,
            "plot_2_dtw_distance.png"
        ),
        dpi=300
    )

    plt.close()

# =====================================================================
# 3. PSEUDO SNR
# =====================================================================

df_snr = df_audit.dropna(
    subset=["Pseudo_SNR_dB"]
)

if len(df_snr) > 0:

    plt.figure(figsize=(12, 6))

    sns.boxplot(
        data=df_snr,
        x="Pipeline",
        y="Pseudo_SNR_dB",
        hue="Target_Class"
    )

    plt.title(
        "Pseudo-SNR Improvement after Preprocessing",
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EDA_DIR,
            "plot_3_pseudo_snr.png"
        ),
        dpi=300
    )

    plt.close()

# =====================================================================
# 4. BASELINE WANDER REDUCTION
# =====================================================================

if len(df_audit) > 0:

    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=df_audit,
        x="Pipeline",
        y="BW_Energy_Reduction",
        hue="Target_Class"
    )

    plt.title(
        "Baseline Wander Reduction (<0.7Hz)",
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EDA_DIR,
            "plot_4_bw_reduction.png"
        ),
        dpi=300
    )

    plt.close()

# =====================================================================
# 5. RR RMSSD DISTRIBUTION
# =====================================================================

df_rr = df_audit.dropna(
    subset=["RR_RMSSD"]
)

if len(df_rr) > 0:

    plt.figure(figsize=(12, 6))

    sns.violinplot(
        data=df_rr,
        x="Target_Class",
        y="RR_RMSSD",
        hue="Pipeline"
    )

    plt.title(
        "RR Interval Preservation (RMSSD)",
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EDA_DIR,
            "plot_5_rr_rmssd.png"
        ),
        dpi=300
    )

    plt.close()

# =====================================================================
# SUMMARY REPORT
# =====================================================================

summary = df_audit.groupby(
    "Pipeline"
)[
    [
        "Pearson_Correlation",
        "DTW_Distance",
        "Pseudo_SNR_dB",
        "BW_Energy_Reduction"
    ]
].mean()

summary_path = os.path.join(
    EDA_DIR,
    "audit_summary.csv"
)

summary.to_csv(summary_path)

print(f"\nSummary saved: {summary_path}")

# =====================================================================
# FINAL MESSAGE
# =====================================================================

print("\n" + "=" * 80)
print("QUANTITATIVE EDA & PREPROCESSING AUDIT FINISHED")
print("=" * 80)