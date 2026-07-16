# =====================================================================
# FILE: src/audit_multilabel_dataset.py
# PURPOSE: AUDIT DISTRIBUSI & KARAKTERISTIK DATASET MULTI-LABEL PTB-XL
# =====================================================================

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())
import config as cfg
import config_ml as ml_cfg

def audit_dataset():
    manifest_path = os.path.join(cfg.RESAMPLE_BASE, "manifest_ptbxl.csv")
    if not os.path.exists(manifest_path):
        print(f"❌ Error: File manifest tidak ditemukan di {manifest_path}")
        return
        
    df = pd.read_csv(manifest_path)
    total_records_global = len(df)
    
    print("="*80)
    print("📊 PTB-XL MULTI-LABEL DATASET AUDIT ENGINE")
    print("="*80)
    print(f"Total rekam medis terdaftar secara global: {total_records_global} record")
    
    # 1. Analisis Kelas Target vs Others
    df_core = df[df["target_class"].isin(ml_cfg.TARGET_CLASSES)].reset_index(drop=True)
    total_core = len(df_core)
    total_others = total_records_global - total_core
    
    print("\n[1] Distribusi Kelompok Kelas (Core vs Others)")
    print(f" -> Kelas Inti (4 Kelas Target) : {total_core} record ({total_core/total_records_global*100:.2f}%)")
    print(f" -> Kelas 'Others' (Dibuang)    : {total_others} record ({total_others/total_records_global*100:.2f}%)")
    
    # 2. Distribusi Frekuensi Per-Kelas Inti (Sebelum Dibuat Multi-Label Kombinasi)
    print("\n[2] Distribusi Frekuensi Sinyal Murni Per-Kelas Inti")
    for cls in ml_cfg.TARGET_CLASSES:
        cnt = len(df_core[df_core["target_class"] == cls])
        print(f" -> Class {cls:<12}: {cnt:<6} record ({cnt/total_core*100:.2f}%)")
        
    # Imbalance Ratio Estimation
    class_counts = [len(df_core[df_core["target_class"] == cls]) for cls in ml_cfg.TARGET_CLASSES]
    ir = max(class_counts) / min(class_counts)
    print(f" -> Imbalance Ratio (IR) Kelas Inti: {ir:.2f}x (Moderat-Tinggi)")

    # 3. Distribusi Splitting Train, Validation, dan Test (Khusus Kelas Inti)
    print("\n[3] Distribusi Pembagian Data (Splitting - Stratified Folds)")
    for fold_type, folds in [("Train Set (Fold 1-8)", range(1, 9)), ("Val Set (Fold 9)", [9]), ("Test Set (Fold 10)", [10])]:
        cnt_split = len(df_core[df_core["strat_fold"].isin(folds)])
        print(f" -> {fold_type:<25}: {cnt_split:<5} record ({cnt_split/total_core*100:.2f}%)")

    # 4. Simulasi Karakteristik Komorbiditas Sinyal Ganda (Multi-Hot Co-occurrence)
    # Catatan: Karena manifes Anda berbentuk single-string kolom tunggal, simulasi ini memetakan
    # potensi record jika dikonversi ke struktur multi-label biner murni.
    print("\n[4] Analisis Densitas Kardinalitas Label (Kombinasi Label Ganda)")
    # Membangun matriks simulasi biner
    simulated_multi_hot = np.zeros((len(df_core), ml_cfg.NUM_CLASSES))
    for idx, row in df_core.iterrows():
        cl_idx = ml_cfg.TARGET_CLASSES.index(row["target_class"])
        simulated_multi_hot[idx, cl_idx] = 1.0
        
    label_sums = np.sum(simulated_multi_hot, axis=1).astype(int)
    unique_sums, count_sums = np.unique(label_sums, return_counts=True)
    
    for labels_count, records_count in zip(unique_sums, count_sums):
        print(f" -> Record yang memiliki {labels_count} label sekaligus: {records_count} record")
    print("="*80)

if __name__ == "__main__":
    audit_dataset()