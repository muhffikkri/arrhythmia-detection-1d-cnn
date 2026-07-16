# src/config_multilabel/config_ml.py
import os
import config as global_cfg  # Import config global lama jika diperlukan

# =====================================================================
# MULTI-LABEL EXPERIMENT CONFIGURATION
# =====================================================================
# Menetapkan 4 Kelas Target Utama (Indeks 0 hingga 3)
TARGET_CLASSES = ["Normal", "AF", "Takikardia", "Bradikardia"]
NUM_CLASSES = len(TARGET_CLASSES)

# Jalur Output Eksperimen Baru agar Terpisah
MULTILABEL_OUTPUT_DIR = os.path.join(
    global_cfg.OUTPUT_DIR, 
    "multilabel_experiments"
)

# Threshold Default Sebelum Optimalisasi (Tuning)
DEFAULT_THRESHOLDS = [0.5, 0.5, 0.5, 0.5]