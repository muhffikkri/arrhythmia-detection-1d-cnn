# src/config/config_labels.py
import os
from src.config import config as global_cfg

# =====================================================================
# DATASET LABEL MAPPINGS
# =====================================================================

PTBXL_TO_TARGET_MAPPING = {
    'SR': 'Normal',
    'SARRH': 'Normal',
    'NORM': 'Normal',

    'AFIB': 'AF',
    'AFLT': 'AF',

    'STACH': 'Takikardia',
    'SVTAC': 'Takikardia',
    'PSVT': 'Takikardia',

    'SBRAD': 'Bradikardia',

    'PACE': 'Others',
    'SVARR': 'Others',
    'BIGU': 'Others',
    'TRIGU': 'Others'
}

CHAPMAN_TO_TARGET_MAPPING = {
    # BRADIKARDIA
    'SB': 'Bradikardia',
    'Sinus Bradycardia': 'Bradikardia',
    '426177001': 'Bradikardia',

    # NORMAL
    'SR': 'Normal',
    'Sinus Rhythm': 'Normal',
    '426783006': 'Normal',

    # ATRIAL FIBRILLATION
    'AFIB': 'AF',
    'Atrial Fibrillation': 'AF',
    '164889003': 'AF',
    'AF': 'AF',
    'Atrial Flutter': 'AF',
    '164890007': 'AF',

    # TAKIKARDIA
    'ST': 'Takikardia',
    'Sinus Tachycardia': 'Takikardia',
    '427084000': 'Takikardia',
    'SVT': 'Takikardia',
    'Supraventricular Tachycardia': 'Takikardia',
    '426761007': 'Takikardia',
    'AT': 'Takikardia',
    'Atrial Tachycardia': 'Takikardia',
    '713422000': 'Takikardia',

    # OTHERS
    'AVNRT': 'Others',
    'Atrioventricular Node Reentrant Tachycardia': 'Others',
    '233896004': 'Others',
    'AVRT': 'Others',
    'Atrioventricular Reentrant Tachycardia': 'Others',
    '233897008': 'Others',
    'SAAWR': 'Others',
    'Sinus Atrium to Atrial Wandering Rhythm': 'Others',
    '195101003': 'Others',
    'SA': 'Others',
    'Sinus Irregularity': 'Others',
    '427393009': 'Others'
}

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