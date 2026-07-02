# =====================================================================
# FILE: experiment_configs.py
# =====================================================================

class Config:

    # =========================================================
    # INPUT
    # =========================================================

    INPUT_SHAPE = (2500, 3)

    CLASSES = 5

    # =========================================================
    # TRAINING
    # =========================================================

    BATCH_SIZE = 64
    EPOCHS = 35
    LEARNING_RATE = 3e-4
    KFOLD_SPLITS = 5
    LABEL_SMOOTHING = 0.0
    USE_MIXUP = True
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
        "Hybrid_Morphology": [25, 15, 11, 5, 3]
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
        "CNN_BiLSTM",
        "CNN_Attention"
    ]

    # =========================================================
    # EXPERIMENT TRACKER
    # =========================================================

    EXPERIMENT_GROUP = "baseline_experiments"

    MASTER_TRACKER_CSV = (
        "master_experiment_tracker.csv"
    )

    SAVE_MISCLASSIFIED = True

    SAVE_CONFUSION_MATRIX = True

    SAVE_HISTORY = True

    SAVE_MODEL_SUMMARY = True
