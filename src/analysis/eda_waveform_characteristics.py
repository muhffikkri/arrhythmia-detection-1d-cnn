# =====================================================================
# FILE: eda_waveform_characteristics.py
# WAVEFORM CHARACTERISTICS EDA (PER-DATASET, PER-CLASS) + CROSS-DATASET
# =====================================================================
# - Computes per-class aggregate waveform statistics (mapped 4-class scheme)
#   for both PTB-XL and Chapman at the unified 250 Hz representation:
#       lead-wise RMS / peak-to-peak / DC level, and Lead-II spectral
#       properties (dominant frequency, spectral entropy).
# - Cross-dataset morphological comparison per class:
#       Pearson correlation, mean absolute error and amplitude ratio of the
#       dataset mean waveforms, per class per lead.
# - Outputs:
#       output/eda_waveform_characteristics/
#           waveform_stats_per_class.csv
#           cross_dataset_morphology.csv
#           plot_mean_waveforms_per_class.png
#           plot_spectral_by_class.png
#           plot_cross_dataset_heatmap.png
# =====================================================================

import os
import sys
import argparse
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

from src.config import config as cfg

warnings.filterwarnings("ignore")

# =====================================================================
# OUTPUT DIRECTORY
# =====================================================================

OUT_DIR = os.path.join(cfg.BASE_DIR, "output", "eda_waveform_characteristics")
os.makedirs(OUT_DIR, exist_ok=True)

FS = 250.0

# =====================================================================
# ARGS
# =====================================================================

parser = argparse.ArgumentParser(
    description="Per-class waveform characteristics per dataset + cross-dataset comparison."
)
parser.add_argument(
    "--n",
    type=int,
    default=100,
    help="Max samples per class per dataset (default: 100).",
)
parser.add_argument(
    "--ptbxl-folder",
    default="E2_clean_100_to_250",
    help="cfg.SUB_FOLDERS key for PTB-XL signals (default: E2_clean_100_to_250).",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for stratified sampling (default: 42).",
)
args = parser.parse_args()

N_PER_CLASS = args.n
PTBXL_FOLDER = cfg.SUB_FOLDERS[args.ptbxl_folder]
CHAPMAN_FOLDER = cfg.SUB_FOLDERS["Chapman_clean_500_to_250"]

CLASS_NAMES = list(cfg.CLASS_NAMES)
LEAD_NAMES = list(cfg.LEAD_NAMES)
N_LEADS = len(LEAD_NAMES)

# =====================================================================
# MANIFEST LOADING (MAPPED 4-CLASS SCHEME)
# =====================================================================

ptb_manifest = pd.read_csv(os.path.join(cfg.RESAMPLE_BASE, "manifest_ptbxl.csv"))
chap_manifest = pd.read_csv(os.path.join(cfg.RESAMPLE_BASE, "manifest_chapman.csv"))

ptb_manifest = ptb_manifest[ptb_manifest["target_class"].isin(CLASS_NAMES)].copy()
chap_manifest = chap_manifest[chap_manifest["target_class"].isin(CLASS_NAMES)].copy()

DATASETS = [
    ("PTB-XL", ptb_manifest, PTBXL_FOLDER),
    ("Chapman", chap_manifest, CHAPMAN_FOLDER),
]

# =====================================================================
# HELPERS
# =====================================================================

def spectral_features(signal, fs=FS):
    """Dominant frequency + spectral entropy of a single univariate signal."""
    nperseg = min(len(signal), int(fs * 2))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    total = np.sum(psd)
    if total < 1e-12:
        return np.nan, np.nan

    dominant_freq = float(freqs[np.argmax(psd)])

    psd_norm = psd / total
    psd_norm = psd_norm[psd_norm > 0]

    spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm)))

    return dominant_freq, spectral_entropy


def per_lead_stats(signal):
    """RMS, peak-to-peak amplitude and DC level for every lead."""
    rms = np.sqrt(np.mean(signal ** 2, axis=0))
    p2p = np.ptp(signal, axis=0)
    dc = np.mean(signal, axis=0)
    return rms, p2p, dc


HALF_WINDOW = 125  # +/- 0.5 s around the R-peak at 250 Hz


def align_beats(signals, half=HALF_WINDOW):
    """Align every beat on the Lead-II R-peak (edge-padded)."""
    n, sig_len, n_leads = signals.shape
    aligned = np.empty((n, 2 * half + 1, n_leads), dtype=np.float32)
    ref = signals[:, :, 1]

    for k in range(n):
        pk = int(np.argmax(ref[k] ** 2))
        lo, hi = pk - half, pk + half + 1
        seg = signals[k, max(lo, 0):min(hi, sig_len)]
        seg = np.pad(
            seg,
            ((max(0, -lo), max(0, hi - sig_len)), (0, 0)),
            mode="constant",
        )
        aligned[k] = seg

    return aligned


def load_class_signals(manifest, folder, class_name, n, seed):
    """Stratified sample of `n` signals for one class."""
    sub = manifest[manifest["target_class"] == class_name]
    avail = len(sub)
    if avail == 0:
        return None, 0

    picked = sub.sample(n=min(n, avail), random_state=seed)

    signals = []
    for row in picked.itertuples(index=False):
        path = os.path.join(folder, row.filename_npy)
        if not os.path.exists(path):
            continue
        try:
            signals.append(np.load(path).astype(np.float32))
        except Exception:
            continue

    if not signals:
        return None, avail

    return np.stack(signals), avail


# =====================================================================
# COLLECT WAVEFORM STATISTICS
# =====================================================================

print("=" * 78)
print("WAVEFORM CHARACTERISTICS EDA  (250Hz, QRS-aligned, mapped 4-class scheme)")
print("=" * 78)

stat_rows = []
envelopes = {}  # (class, dataset) -> (names, mean[n_lead, len], std[n_lead, len])

for class_name in CLASS_NAMES:

    for dataset_name, manifest, folder in DATASETS:

        signals, avail = load_class_signals(
            manifest, folder, class_name, N_PER_CLASS, args.seed
        )

        if signals is None:
            print(f"  {dataset_name:<9} {class_name:<12}: skipping (0 samples available)")
            continue

        n_samples, sig_len, n_leads = signals.shape

        # Align every beat on the Lead-II R-peak so the average produces a
        # meaningful morphological envelope (unaligned averaging washes out QRS).
        aligned = align_beats(signals, half=HALF_WINDOW)

        mean_wave = aligned.mean(axis=0)      # (win_len, n_leads)
        std_wave = aligned.std(axis=0)        # (win_len, n_leads)

        envelopes[(class_name, dataset_name)] = (LEAD_NAMES, mean_wave, std_wave)

        rms_all, p2p_all, dc_all = per_lead_stats(signals)
        dom_freq, spec_entropy = spectral_features(
            signals[:, :, 1].mean(axis=0)
        )

        row = {"Class": class_name, "Dataset": dataset_name,
               "Samples_Analyzed": n_samples, "Samples_Available": avail}

        for i, lead in enumerate(LEAD_NAMES):
            row[f"RMS_{lead}"] = float(np.mean(rms_all[:, i]))
            row[f"PeakToPeak_{lead}"] = float(np.mean(p2p_all[:, i]))
            row[f"DC_Level_{lead}"] = float(np.mean(dc_all[:, i]))

        row["LeadII_Dominant_Freq_Hz"] = dom_freq
        row["LeadII_Spectral_Entropy"] = spec_entropy

        stat_rows.append(row)
        print(f"  {dataset_name:<9} {class_name:<12}: "
              f"n={n_samples}/{avail} | LeadII DF={dom_freq:.2f} Hz "
              f"| SEnt={spec_entropy:.2f} | RMS[II]={float(np.mean(rms_all[:, 1])):.3f}")

stat_df = pd.DataFrame(stat_rows)
stat_csv = os.path.join(OUT_DIR, "waveform_stats_per_class.csv")
stat_df.to_csv(stat_csv, index=False)
print(f"\nSaved stats CSV: {stat_csv}")

# =====================================================================
# CROSS-DATASET MORPHOLOGY COMPARISON
# =====================================================================

print("\n" + "=" * 78)
print("CROSS-DATASET MORPHOLOGY COMPARISON (QRS-aligned mean waveforms)")
print("=" * 78)

cross_rows = []

for class_name in CLASS_NAMES:

    ptb = envelopes.get((class_name, "PTB-XL"))
    chap = envelopes.get((class_name, "Chapman"))

    if ptb is None or chap is None:
        print(f"  {class_name:<12}: incomplete pair, skipped")
        continue

    _, mean_ptb, std_ptb = ptb
    _, mean_chap, std_chap = chap

    for i, lead in enumerate(LEAD_NAMES):

        a = mean_ptb[:, i]
        b = mean_chap[:, i]

        corr, _ = pearsonr(a, b)

        mae = float(np.mean(np.abs(a - b)))

        amp_ptb = float(np.max(a) - np.min(a))
        amp_chap = float(np.max(b) - np.min(b))

        cross_rows.append({
            "Class": class_name,
            "Lead": lead,
            "Pearson_Correlation": float(corr),
            "Mean_Abs_Error": mae,
            "Amplitude_PTBXL": amp_ptb,
            "Amplitude_Chapman": amp_chap,
            "Amplitude_Ratio_PTBXL_over_Chapman": (
                amp_ptb / amp_chap if amp_chap > 1e-12 else np.nan
            ),
        })

        print(f"  {class_name:<12} lead {lead}: Pearson={corr:.3f} | "
              f"MAE={mae:.3f} | amp ratio={amp_ptb / amp_chap:.2f}")

cross_df = pd.DataFrame(cross_rows)
cross_csv = os.path.join(OUT_DIR, "cross_dataset_morphology.csv")
cross_df.to_csv(cross_csv, index=False)
print(f"\nSaved cross-dataset CSV: {cross_csv}")

# =====================================================================
# PLOT 1: MEAN WAVEFORMS PER CLASS (LEAD II, +/- 1 STD)
# =====================================================================

print("\nRendering mean waveform plots (Lead II)...")

fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True, sharey=True)
axes = axes.reshape(-1)
time = np.arange(-HALF_WINDOW, HALF_WINDOW + 1) / FS  # centered on R-peak

for idx, class_name in enumerate(CLASS_NAMES):

    ax = axes[idx]

    for dataset_name, color in [("PTB-XL", "#1A365D"), ("Chapman", "#C53030")]:

        env = envelopes.get((class_name, dataset_name))

        if env is None:
            continue

        _, mean_wave, std_wave = env

        ax.plot(time, mean_wave[:, 1], color=color, linewidth=1.4,
                label=f"{dataset_name} (mean)")
        ax.fill_between(
            time,
            mean_wave[:, 1] - std_wave[:, 1],
            mean_wave[:, 1] + std_wave[:, 1],
            color=color, alpha=0.18, linewidth=0
        )

    ax.set_title(f"{class_name}", fontweight="bold")
    ax.set_ylabel("Amplitude (Lead II)", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8)

fig.suptitle(
    "Mean +/- 1 Std Waveform per Class (QRS-aligned, 250Hz) - PTB-XL vs Chapman",
    fontweight="bold", y=0.98
)
fig.tight_layout(rect=[0, 0, 1, 0.96])

plot1 = os.path.join(OUT_DIR, "plot_mean_waveforms_per_class.png")
fig.savefig(plot1, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {plot1}")

# =====================================================================
# PLOT 2: SPECTRAL / AMPLITUDE SUMMARY BARS
# =====================================================================

print("Rendering spectral + amplitude summary...")

stat_plot = stat_df.copy()
pivot = stat_plot.pivot(index="Class", columns="Dataset",
                        values="LeadII_Dominant_Freq_Hz")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

pivot.plot(kind="bar", ax=axes[0], color=["#1A365D", "#C53030"])
axes[0].set_title("Lead II Dominant Frequency by Class", fontweight="bold")
axes[0].set_ylabel("Hz")
axes[0].set_ylim(0, max(pivot.max()) * 1.2 if len(pivot) else 1)
axes[0].grid(True, axis="y", linestyle=":", alpha=0.6)

pivot_ent = stat_plot.pivot(index="Class", columns="Dataset",
                            values="LeadII_Spectral_Entropy")
pivot_ent.plot(kind="bar", ax=axes[1], color=["#1A365D", "#C53030"])
axes[1].set_title("Lead II Spectral Entropy by Class", fontweight="bold")
axes[1].set_ylabel("bits")
axes[1].set_ylim(0, max(pivot_ent.max()) * 1.2 if len(pivot_ent) else 1)
axes[1].grid(True, axis="y", linestyle=":", alpha=0.6)

fig.tight_layout()

plot2 = os.path.join(OUT_DIR, "plot_spectral_by_class.png")
fig.savefig(plot2, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {plot2}")

# =====================================================================
# PLOT 3: CROSS-DATASET CORRELATION HEATMAP
# =====================================================================

if len(cross_df):

    print("Rendering cross-dataset correlation heatmap...")

    cm = cross_df.pivot(index="Lead", columns="Class",
                        values="Pearson_Correlation")
    cm = cm[CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(
        cm, annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=-1.0, vmax=1.0, linewidths=0.5,
        linecolor="white", cbar_kws={"label": "Pearson r of mean waveforms"},
        ax=ax
    )
    ax.set_title(
        "Cross-Dataset Morphology Match (PTB-XL mean vs Chapman mean)",
        fontweight="bold"
    )
    ax.set_ylabel("Lead")
    ax.set_xlabel("Class")

    fig.tight_layout()

    plot3 = os.path.join(OUT_DIR, "plot_cross_dataset_heatmap.png")
    fig.savefig(plot3, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {plot3}")

# =====================================================================
# DONE
# =====================================================================

print("\n" + "=" * 78)
print("DONE - outputs in:", OUT_DIR)
print("=" * 78)