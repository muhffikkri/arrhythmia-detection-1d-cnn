# =====================================================================
# FILE: experiment_configs.py
# =====================================================================

class Config:

    # =========================================================
    # INPUT
    # =========================================================

    INPUT_SHAPE = (2500, 3)

    # Class count for the shared 4-class scheme. When LABEL_SCHEME is
    # "native", the number of classes is derived from the manifest by the
    # dataset loader and overrides this value.
    CLASSES = 4

    # =========================================================
    # LABEL SCHEME (DATA-DRIVEN)
    # =========================================================
    # - "mapped": the shared 4-class scheme (config.TARGET_CLASSES) used
    #   for cross-dataset training/evaluation (Chapman <-> PTB-XL).
    # - "native": labels use the dataset's original names (PTB-XL SCP codes,
    #   Chapman SNOMED-CT codes) read straight from the manifest; no mapping.
    LABEL_SCHEME = "mapped"

    # =========================================================
    # TRAIN / TEST DATASET SELECTION (CROSS-DATASET MATRIX)
    # =========================================================
    # Options: "PTBXL" | "CHAPMAN"
    # - TRAIN_DATASET == TEST_DATASET -> per-dataset training/test.
    # - TRAIN_DATASET="PTBXL", TEST_DATASET="CHAPMAN" -> zero-shot external
    #   validation (and the reverse direction likewise).
    TRAIN_DATASET = "PTBXL"
    TEST_DATASET = "PTBXL"

    # =========================================================
    # TRAINING
    # =========================================================

    BATCH_SIZE = 64
    EPOCHS = 35
    LEARNING_RATE = 3e-4
    KFOLD_SPLITS = 5
    LABEL_SMOOTHING = 0.0
    USE_MIXUP = False
    MIXUP_ALPHA = 0.1
    USE_SEPARABLE_CONV = False
    USE_AUGMENTATION = False
    STOCHASTIC_DEPTH_RATE = 0.0
    COSINE_DECAY_STEPS = 3000
    COSINE_ALPHA = 1e-2
    OPTIMIZER = "Adam"
    USE_COSINE_DECAY = True

    # =========================================================
    # FOCAL LOSS
    # =========================================================

    FOCAL_CONFIGS = [
        {
            "alpha": 1.0,
            "gamma": 2.0
        }
    ]

    # =========================================================
    # ACTIVE DATASETS FOR EXPERIMENTS
    # =========================================================
    # Options for raw vs cleaned datasets (matching cfg.SUB_FOLDERS):
    # - Raw Datasets:     "E1_100_native", "E2_100_to_250", "E3_500_to_250", "E4_500_native"
    # - Cleaned Datasets: "E1_clean_100_native", "E2_clean_100_to_250", "E3_clean_500_to_250", "E4_clean_500_native"
    ACTIVE_DATASETS = [
        "E2_clean_100_to_250",
        "E3_clean_500_to_250",
        # "E2_100_to_250",      # Raw 100Hz upsampled to 250Hz
        # "E3_500_to_250",      # Raw 500Hz downsampled to 250Hz
    ]

    # =========================================================
    # FILTER SPACES
    # =========================================================

    FILTER_SPACES = {
        # "Small": [32, 64, 128, 128, 256],
        "Medium": [64, 128, 256, 256, 512]
    }

    # =========================================================
    # KERNEL SPACES
    # =========================================================

    KERNEL_SPACES = {
        "Balanced": [15, 11, 7, 5, 3],
        # "Balanced_v2": [11, 9, 7, 5, 3],
        # "Local_Focused": [7, 5, 5, 3, 3],
        # "Large_Receptive": [21, 15, 11, 7, 5],
        # "Huge_Receptive": [31, 21, 15, 9, 5]
        # "Hybrid_Morphology": [25, 15, 11, 5, 3]
    }

    # =========================================================
    # DILATION
    # =========================================================

    DILATION_SPACES = {
        "Progressive_Dilation": [1, 2, 4, 8, 16],
        # "Compact_Dilation": [1, 1, 2, 2, 4]
    }

    # =========================================================
    # TEMPORAL MODELS
    # =========================================================

    TEMPORAL_MODELS = [
        "Pure_CNN",
        # "CNN_BiLSTM",
        # "CNN_Attention",
        # "LSTM_Only",
    ]

    # =========================================================
    # EXPERIMENT TRACKER
    # =========================================================

    EXPERIMENT_GROUP = "balanced_baseline"

    MASTER_TRACKER_CSV = (
        "master_experiment_tracker.csv"
    )

    SAVE_MISCLASSIFIED = True

    SAVE_CONFUSION_MATRIX = True

    SAVE_HISTORY = True

    SAVE_MODEL_SUMMARY = True

    UNDERSAMPLE_RATIO = 10

    # =========================================================
    # OVERSAMPLING
    # =========================================================
    # Method of oversampling:
    # - None: No oversampling.
    # - "smote": Use SMOTE oversampling.
    # - "smote_tomek": Use SMOTE-Tomek oversampling.
    OVERSAMPLE_METHOD = "smote_tomek"

    # Strategy for oversampling.
    # Can be 'auto', float, or dictionary.
    # We use class name strings to avoid index order differences.
    # Set minority classes to 3000 or 5000:
    OVERSAMPLE_STRATEGY = {
        "AF": 3000,
        "Bradikardia": 3000,
        "Takikardia": 3000
    }
