# =====================================================================
# FILE 7: visualize_cross_dataset.py — DOMAIN SHIFT MORPHOLOGY MATCH
# =====================================================================
import os
import sys

# Add the project root directory to the python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import config as cfg

CROSS_DIR = os.path.join(cfg.BASE_DIR, "output", "cross_dataset_comparison")
os.makedirs(CROSS_DIR, exist_ok=True)

df_ptb = pd.read_csv(os.path.join(cfg.RESAMPLE_BASE, "manifest_ptbxl.csv"))
df_chap = pd.read_csv(os.path.join(cfg.RESAMPLE_BASE, "manifest_chapman.csv"))

print("--> Merender Plot Komparasi Lintas Dataset (PTB-XL vs Chapman)...")

for cls in cfg.CLASS_NAMES:
    ptb_samples = df_ptb[df_ptb['target_class'] == cls]
    chap_samples = df_chap[df_chap['target_class'] == cls]
    
    if len(ptb_samples) == 0 or len(chap_samples) == 0:
        print(f"   ⚠️ Melewati {cls}: Sampel tidak lengkap di kedua dataset.")
        continue
        
    ptb_filename = ptb_samples.iloc[0]['filename_npy']
    chap_filename = chap_samples.iloc[0]['filename_npy']
    
    print(f"   ↳ Menyandingkan Kelas: [{cls}]")
    
    # PTB-XL memprioritaskan referensi baseline 250Hz (500 downsampled) 
    # atau fallback ke 100Hz (upsampled) sesuai pipeline literatur
    ptb_path = os.path.join(
        cfg.SUB_FOLDERS["E3_clean_500_to_250"],
        ptb_filename
    )
        
    chap_path = os.path.join(cfg.SUB_FOLDERS["CHAPMAN_CLEAN_250HZ"], chap_filename)
    
    ptb_sig = np.load(ptb_path)
    chap_sig = np.load(chap_path)
    
    fs = 250.0
    time_ptb = np.arange(len(ptb_sig)) / fs
    time_chap = np.arange(len(chap_sig)) / fs
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex='col')
    
    for lead_idx in range(3):
        # PTB-XL Plot (Kolom Kiri)
        axes[lead_idx, 0].plot(time_ptb, ptb_sig[:, lead_idx], color='#1A365D', alpha=0.9, linewidth=1.2)
        axes[lead_idx, 0].set_ylabel(cfg.LEAD_NAMES[lead_idx], fontweight='bold')
        axes[lead_idx, 0].grid(linestyle=':', alpha=0.6)
        if lead_idx == 0: axes[lead_idx, 0].set_title(f"PTB-XL (Internal Domain)\nCleaned & Resampled to 250Hz", fontweight='bold', color='#1A365D')
        if lead_idx == 2: axes[lead_idx, 0].set_xlabel("Time (Seconds)")
            
        # Chapman Plot (Kolom Kanan)
        axes[lead_idx, 1].plot(time_chap, chap_sig[:, lead_idx], color='#C53030', alpha=0.9, linewidth=1.2)
        axes[lead_idx, 1].grid(linestyle=':', alpha=0.6)
        if lead_idx == 0: axes[lead_idx, 1].set_title(f"CHAPMAN (External Domain)\nUnified 250Hz ECG Representation", fontweight='bold', color='#C53030')
        if lead_idx == 2: axes[lead_idx, 1].set_xlabel("Time (Seconds)")
    
    plt.suptitle(f"Cross-Dataset Morphology Match (Literature Pipeline) — Kelas: {cls}", fontsize=15, fontweight='bold', y=0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plt.savefig(os.path.join(CROSS_DIR, f"cross_domain_{cls.replace(' ', '_')}.png"), dpi=200, bbox_inches='tight')
    plt.close()

print("✓ Komparasi Lintas Dataset Tuntas Diekspor.")