# =====================================================================
# FILE 5: visualize_morphology.py — FINAL FIXED VERSION
# 9-PANEL MORPHOLOGY TRACKER
# =====================================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as cfg

# =====================================================================
# LOAD MANIFEST
# =====================================================================

manifest_path = os.path.join(
    cfg.RESAMPLE_BASE,
    "manifest_ptbxl.csv"
)

df_manifest = pd.read_csv(manifest_path)

# =====================================================================
# EXPERIMENT VISUAL MAP
# =====================================================================

"""
Format tuple:

(
    raw_native_folder,
    raw_fs,

    cleaned_native_folder,
    cleaned_fs,

    final_resampled_folder,
    final_fs
)
"""

exp_visual_map = {

    # =========================================================
    # 100Hz Native
    # =========================================================

    "100Hz Native → 250Hz Final": (

        "E1_100_native",
        100,

        "E1_clean_100_native",
        100,

        "E2_clean_100_to_250",
        250
    ),

    # =========================================================
    # 500Hz Native
    # =========================================================

    "500Hz Native → 250Hz Final": (

        "E4_500_native",
        500,

        "E4_clean_500_native",
        500,

        "E3_clean_500_to_250",
        250
    )
}

# =====================================================================
# OUTPUT
# =====================================================================

print("\n--> Merender Plot Morfologi 3 Tahap")
print("    (Raw Native → Clean Native → Final Resampled)...")

# =====================================================================
# MAIN LOOP
# =====================================================================

for cls in cfg.CLASS_NAMES:

    samples = df_manifest[
        df_manifest["target_class"] == cls
    ]

    if len(samples) == 0:

        print(f"   ⚠️ Melewati {cls}: Tidak ada sampel.")
        continue

    filename = samples.iloc[0]["filename_npy"]

    print(f"   ↳ Merender Kelas: [{cls}]")

    # =========================================================
    # EXPERIMENT LOOP
    # =========================================================

    for exp_label, (
        raw_key,
        raw_fs,

        clean_key,
        clean_fs,

        final_key,
        final_fs

    ) in exp_visual_map.items():

        # =====================================================
        # PATHS
        # =====================================================

        raw_path = os.path.join(
            cfg.SUB_FOLDERS[raw_key],
            filename
        )

        clean_path = os.path.join(
            cfg.SUB_FOLDERS[clean_key],
            filename
        )

        final_path = os.path.join(
            cfg.SUB_FOLDERS[final_key],
            filename
        )

        # =====================================================
        # EXISTENCE CHECK
        # =====================================================

        if not os.path.exists(raw_path):

            print(f"      ⚠ Missing RAW: {raw_path}")
            continue

        if not os.path.exists(clean_path):

            print(f"      ⚠ Missing CLEAN: {clean_path}")
            continue

        if not os.path.exists(final_path):

            print(f"      ⚠ Missing FINAL: {final_path}")
            continue

        # =====================================================
        # LOAD SIGNALS
        # =====================================================

        try:

            raw_sig = np.load(raw_path)
            clean_sig = np.load(clean_path)
            final_sig = np.load(final_path)

        except Exception as e:

            print(f"      ⚠ Error loading {filename}: {e}")
            continue

        # =====================================================
        # TIME AXIS
        # =====================================================

        time_raw = np.arange(len(raw_sig)) / raw_fs

        time_clean = np.arange(len(clean_sig)) / clean_fs

        time_final = np.arange(len(final_sig)) / final_fs

        # =====================================================
        # FIGURE
        # =====================================================

        fig, axes = plt.subplots(
            3,
            3,
            figsize=(18, 8),
            sharex='col',
            sharey='row'
        )

        # =====================================================
        # LEAD LOOP
        # =====================================================

        for lead_idx in range(3):

            # =================================================
            # RAW NATIVE
            # =================================================

            axes[lead_idx, 0].plot(
                time_raw,
                raw_sig[:, lead_idx],
                color='#757575',
                linewidth=1.2,
                alpha=0.9
            )

            axes[lead_idx, 0].grid(
                linestyle=':',
                alpha=0.5
            )

            axes[lead_idx, 0].set_ylabel(
                cfg.LEAD_NAMES[lead_idx],
                fontweight='bold'
            )

            if lead_idx == 0:

                axes[lead_idx, 0].set_title(
                    f"1. RAW NATIVE ({raw_fs}Hz)",
                    fontweight='bold',
                    color='#424242'
                )

            if lead_idx == 2:

                axes[lead_idx, 0].set_xlabel(
                    "Time (Seconds)"
                )

            # =================================================
            # CLEAN NATIVE
            # =================================================

            axes[lead_idx, 1].plot(
                time_clean,
                clean_sig[:, lead_idx],
                color='#8E24AA',
                linewidth=1.2,
                alpha=0.9
            )

            axes[lead_idx, 1].grid(
                linestyle=':',
                alpha=0.5
            )

            if lead_idx == 0:

                axes[lead_idx, 1].set_title(
                    (
                        f"2. CLEANED NATIVE ({clean_fs}Hz)\n"
                        "Wavelet + Median Filter"
                    ),
                    fontweight='bold',
                    color='#6A1B9A'
                )

            if lead_idx == 2:

                axes[lead_idx, 1].set_xlabel(
                    "Time (Seconds)"
                )

            # =================================================
            # FINAL RESAMPLED
            # =================================================

            axes[lead_idx, 2].plot(
                time_final,
                final_sig[:, lead_idx],
                color='#1976D2',
                linewidth=1.2,
                alpha=0.9
            )

            axes[lead_idx, 2].grid(
                linestyle=':',
                alpha=0.5
            )

            if lead_idx == 0:

                axes[lead_idx, 2].set_title(
                    (
                        f"3. FINAL RESAMPLED ({final_fs}Hz)\n"
                        "Ready for CNN"
                    ),
                    fontweight='bold',
                    color='#0D47A1'
                )

            if lead_idx == 2:

                axes[lead_idx, 2].set_xlabel(
                    "Time (Seconds)"
                )

        # =====================================================
        # TITLE
        # =====================================================

        plt.suptitle(
            (
                f"Signal Transformation Pipeline — "
                f"{cls} ({exp_label})"
            ),
            fontsize=15,
            fontweight='bold',
            y=1.02
        )

        plt.tight_layout(
            rect=[0, 0, 1, 0.95]
        )

        # =====================================================
        # SAVE
        # =====================================================

        safe_label = (
            exp_label
            .lower()
            .replace(" ", "_")
            .replace("→", "_to_")
        )

        save_path = os.path.join(
            cfg.OUTPUT_DIR,
            cls,
            f"plot_morphology_{cls}_{safe_label}.png"
        )

        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches='tight'
        )

        plt.close()

        print(f"      ✓ Saved: {save_path}")

# =====================================================================
# FINISHED
# =====================================================================

print("\n✓ Visualisasi 9-Panel Transformasi Sinyal Berhasil.")