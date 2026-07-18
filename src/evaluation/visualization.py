# =====================================================================
# FILE: visualization.py
# PURPOSE: MEDICAL-GRADE MULTI-LEAD ECG VISUALIZATION FOR ERROR ANALYSIS
# =====================================================================

import os
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# SAVE MISCLASSIFIED ECG (VERTICALLY STACKED MULTI-PLOT)
# =====================================================================

def save_misclassified_samples(
    X,
    y_true,
    y_pred,
    class_names,
    save_dir,
    max_samples=20
):
    """
    Menyimpan visualisasi sinyal EKG yang salah diklasifikasikan oleh model.
    Memisahkan setiap Lead (I, II, III) ke dalam 3 subplot vertikal terisolasi 
    agar mudah dievaluasi secara klinis tanpa tumpang tindih.
    """
    mis_idx = np.where(
        y_true != y_pred
    )[0]

    save_dir = os.path.join(
        save_dir,
        "misclassified"
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    # Batasi sampel maksimum yang di-plot agar hemat ruang disk dan eksekusi cepat
    for i, idx in enumerate(mis_idx[:max_samples]):

        signal = X[idx]  # Dimensi tensor: [Timesteps, Channels/Leads]
        num_leads = signal.shape[-1]

        # Proteksi teks label nama kelas target
        true_cls = class_names[y_true[idx]]
        pred_cls = class_names[y_pred[idx]]

        # 1. Inisialisasi Matplotlib Subplots: 3 Baris, 1 Kolom
        # sharex=True memastikan koordinat waktu di sumbu X sejajar dari atas ke bawah
        fig, axes = plt.subplots(
            nrows=num_leads,
            ncols=1,
            figsize=(14, 2 * num_leads),  # Rasio vertikal proporsional (14x6 untuk 3 leads)
            sharex=True
        )

        # Jika data hanya memiliki 1 lead secara tidak sengaja, ubah axes menjadi array agar iterasi aman
        if num_leads == 1:
            axes = [axes]

        # 2. Iterasi untuk Melakukan Plotting Per-Channel Mandiri
        for ch in range(num_leads):
            ax = axes[ch]
            
            # Plot sinyal dengan warna klinis netral (hitam/gelap) agar kontras
            ax.plot(
                signal[:, ch],
                color='#1c1c1e',
                linewidth=1.2,
                label=f"Lead {ch+1}"
            )
            
            # Kebutuhan Estetika Evaluasi Medis
            ax.set_ylabel(f"Lead {ch+1}", fontsize=10, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5, color='#d2d2d7')
            ax.legend(loc='upper right', fontsize=8)

        # 3. Menambahkan Judul Global Utama di Bagian Paling Atas Panel
        plt.suptitle(
            f"ERROR ANALYSIS SAMPLE #{i} | True Label: {true_cls} -> Predicted: {pred_cls}",
            fontsize=12,
            fontweight='bold',
            color='#ff453a'  # Warna merah untuk menandakan sampel galat klasifikasi
        )

        # Mengatur margin otomatis agar label sumbu Y tidak terpotong
        plt.tight_layout()

        # 4. Simpan ke Direktori Output Eksperimen
        plt.savefig(
            os.path.join(
                save_dir,
                f"{i}_{true_cls}_to_{pred_cls}.png"
            ),
            dpi=150  # Resolusi cukup tajam untuk bahan draf paparan/paper PKM
        )
        
        # Bersihkan memori kanvas secara disiplin untuk mencegah memory leak saat loop besar
        plt.close(fig)