# =====================================================================
# FILE: run_experiment.py
# UNIFIED ECG RESEARCH PIPELINE RUNNER
# =====================================================================
# Single entry point replacing the legacy softmax/sigmoid runners.
# Behaviour is driven entirely by src/config/experiment_configs.py:
#   - SCHEME        : "softmax" (multiclass) | "sigmoid" (multilabel)
#   - LABEL_SCHEME  : "mapped" (shared 4-class) | "native" (dataset labels)
#   - TRAIN_DATASET / TEST_DATASET : "PTBXL" | "CHAPMAN" (cross-dataset matrix)
# Export: run_experiment(ptbxl=None, chapman=None, backend="tf", stage="train")
# =====================================================================

import os
import gc
import sys
import random
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    recall_score,
    classification_report,
    multilabel_confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    CSVLogger
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
import tensorflow.keras.backend as K

# Add project root directory to python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# =====================================================================
# LOCAL IMPORTS
# =====================================================================

from src.config import config as cfg
from src.config.experiment_configs import Config
from src.models.model_factory import build_dynamic_cnn
from src.models.loss_factory import get_loss_function
from src.evaluation.evaluator import calculate_ml_metrics
from src.training.data_utils import create_tf_dataset
from src.training.dataset_loader import DatasetLoader

from src.training.tracker import (
    create_experiment_dir,
    save_experiment_config,
    save_training_history,
    save_confusion_matrix,
    save_classification_report,
    save_model_summary,
    update_master_tracker
)

from src.evaluation.visualization import save_misclassified_samples

warnings.filterwarnings("ignore")

# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def set_global_seeds(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_global_seeds(42)

# =====================================================================
# OUTPUT ROOT
# =====================================================================

ROOT_EXP_DIR = os.path.join(cfg.OUTPUT_DIR, "research_experiments")
os.makedirs(ROOT_EXP_DIR, exist_ok=True)

# =====================================================================
# LOSS FUNCTION
# =====================================================================

SCHEME = getattr(Config, "SCHEME", "softmax").lower()
LABEL_SCHEME = getattr(Config, "LABEL_SCHEME", "mapped").lower()
TRAIN_DATASET = getattr(Config, "TRAIN_DATASET", "PTBXL").upper()
TEST_DATASET = getattr(Config, "TEST_DATASET", "PTBXL").upper()
IS_CROSS_DATASET = TRAIN_DATASET != TEST_DATASET

if IS_CROSS_DATASET and LABEL_SCHEME != "mapped":
    raise ValueError(
        "Cross-dataset experiments require LABEL_SCHEME='mapped' because the "
        "native label spaces (SCP codes vs SNOMED-CT codes) do not overlap."
    )

CHAPMAN_FOLDER_KEY = "Chapman_clean_500_to_250"


def get_loss():
    """Focal loss for multiclass; binary crossentropy for multilabel."""
    if SCHEME == "sigmoid":
        print("[Loss] Using BinaryCrossentropy")
        return tf.keras.losses.BinaryCrossentropy(from_logits=False)
    focal_cfg = Config.FOCAL_CONFIGS[0]
    return get_loss_function(
        strategy="focal",
        gamma=focal_cfg["gamma"],
        alpha=focal_cfg["alpha"],
        label_smoothing=Config.LABEL_SMOOTHING
    )


# =====================================================================
# CALLBACKS
# =====================================================================

def get_callbacks(exp_dir, model_path):
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        CSVLogger(
            os.path.join(exp_dir, "training_log.csv"),
            append=False
        )
    ]


# =====================================================================
# CLASS WEIGHT
# =====================================================================

def build_class_weights(y_train):
    y_labels = np.argmax(y_train, axis=1)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_labels),
        y=y_labels
    )
    return {int(c): float(w) for c, w in zip(np.unique(y_labels), weights)}


# =====================================================================
# DATA LOADING (via DatasetLoader)
# =====================================================================

def load_training_data(folder_key=None):
    """Train/val/test splits from the configured TRAIN_DATASET + LABEL_SCHEME."""
    loader = DatasetLoader(
        dataset=TRAIN_DATASET,
        label_scheme=LABEL_SCHEME,
        folder_key=folder_key
    )
    loader.print_summary()
    return loader

def load_eval_data(loader):
    """
    Build the evaluation set under the *training* class ordering.
    Per-dataset -> the loader's own test split.
    Cross-dataset -> the target dataset's test split (mapped scheme only).
    """
    if not IS_CROSS_DATASET:
        return loader.build_eval_split("test")

    eval_loader = DatasetLoader(
        dataset=TEST_DATASET,
        label_scheme="mapped"
    )
    return eval_loader.build_eval_split("test")


# =====================================================================
# THRESHOLD TUNING (SIGMOID)
# =====================================================================

def perform_threshold_tuning(model, X_val, y_val, class_names):
    print("\n[Optimization] Optimizing Per-Class Decision Thresholds via Validation Set...")
    y_val_prob = model.predict(X_val, batch_size=Config.BATCH_SIZE, verbose=0)

    best_thresholds = []
    threshold_range = np.linspace(0.1, 0.9, 81)

    for cl_idx, class_name in enumerate(class_names):
        best_f1, best_th = 0.0, 0.5
        for th in threshold_range:
            y_pred_bin = (y_val_prob[:, cl_idx] >= th).astype(int)
            y_true_bin = y_val[:, cl_idx].astype(int)
            score = f1_score(y_true_bin, y_pred_bin, zero_division=0)
            if score > best_f1:
                best_f1, best_th = score, th
        print(f" -> Best Threshold for {class_name}: {best_th:.2f} (Val F1: {best_f1:.4f})")
        best_thresholds.append(best_th)
    return best_thresholds


# =====================================================================
# EVALUATION — SIGMOID
# =====================================================================

def evaluate_sigmoid(model, X_test, y_test, thresholds, class_names):
    y_prob = model.predict(X_test, batch_size=Config.BATCH_SIZE, verbose=0)

    y_bin = np.zeros_like(y_prob)
    for i in range(len(class_names)):
        y_bin[:, i] = (y_prob[:, i] >= thresholds[i]).astype(int)

    metrics = {
        "Scheme": "sigmoid",
        "Label_Scheme": LABEL_SCHEME,
        "Train_Dataset": TRAIN_DATASET,
        "Test_Dataset": TEST_DATASET,
        "Subset_Accuracy": accuracy_score(y_test, y_bin),
        "Macro_Recall": recall_score(y_test, y_bin, average="macro", zero_division=0),
        "Macro_F1": f1_score(y_test, y_bin, average="macro", zero_division=0),
        "Thresholds_Assigned": str(thresholds)
    }
    for i, cls in enumerate(class_names):
        metrics[f"F1_{cls}"] = float(f1_score(y_test[:, i], y_bin[:, i], zero_division=0))
        try:
            metrics[f"AUROC_{cls}"] = float(roc_auc_score(y_test[:, i], y_prob[:, i]))
        except ValueError:
            metrics[f"AUROC_{cls}"] = float("nan")

    return metrics, y_bin, y_prob


def save_multilabel_confusion_matrix(y_true, y_bin, class_names, save_dir):
    mcm = multilabel_confusion_matrix(y_true, y_bin)
    n = len(class_names)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.atleast_1d(np.array(axes).reshape(-1))
    for i, class_name in enumerate(class_names):
        ax = axes[i]
        cm = mcm[i]
        ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
        for g_i in range(cm.shape[0]):
            for g_j in range(cm.shape[1]):
                ax.text(x=g_j, y=g_i, s=cm[g_i, g_j], va="center", ha="center",
                        fontsize=12, fontweight="bold")
        ax.set_title(f"CM Biner: {class_name}", fontweight="bold")
        ax.set_xticklabels(["", "Neg", "Pos"])
        ax.set_yticklabels(["", "Neg", "Pos"])
    fig.suptitle("Multilabel Confusion Matrix", fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


def save_multilabel_misclassified(X, y_true, y_pred_prob, thresholds, class_names, save_dir):
    y_bin = np.zeros_like(y_pred_prob)
    for i in range(len(class_names)):
        y_bin[:, i] = (y_pred_prob[:, i] >= thresholds[i]).astype(int)

    mis_idx = np.where(np.any(y_true != y_bin, axis=1))[0]
    out_dir = os.path.join(save_dir, "misclassified")
    os.makedirs(out_dir, exist_ok=True)

    for count, idx in enumerate(mis_idx[:20]):
        signal = X[idx]
        num_leads = signal.shape[-1]
        fig, axes = plt.subplots(nrows=num_leads, ncols=1, figsize=(15, 2 * num_leads), sharex=True)
        if num_leads == 1:
            axes = [axes]
        for ch in range(num_leads):
            axes[ch].plot(signal[:, ch], color="#1c1c1e", linewidth=1.1)
            axes[ch].set_ylabel(f"Lead {ch+1}", fontsize=9, fontweight="bold")
            axes[ch].grid(True, linestyle=":", alpha=0.6)
        title_lines = []
        for c_i, c_name in enumerate(class_names):
            status_true = "POS" if y_true[idx, c_i] == 1 else "NEG"
            status_pred = "POS" if y_bin[idx, c_i] == 1 else "NEG"
            title_lines.append(f"{c_name}(True:{status_true}|Pred:{status_pred}|Conf:{y_pred_prob[idx, c_i]:.3f})")
        plt.suptitle(
            f"ERROR SAMPLE #{count} | ANALYSIS MATRIX:\n" + " | ".join(title_lines),
            fontsize=10, fontweight="bold", color="#ff453a", y=0.98
        )
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"error_{count}_multilabel.png"), dpi=150)
        plt.close()


# =====================================================================
# BALANCING (MAPPED SCHEME ONLY)
# =====================================================================

def balance_training_data(X_train, y_train):
    """Undersample / oversample the train split when LABEL_SCHEME == 'mapped'."""
    if LABEL_SCHEME != "mapped":
        print("[Balancing] Skipped for native label scheme.")
        return X_train, y_train

    ratio = getattr(Config, "UNDERSAMPLE_RATIO", None)
    if ratio is not None:
        from src.training.data_utils import undersample_training_data
        X_train, y_train = undersample_training_data(
            X_train=X_train,
            y_train=y_train,
            ratio=ratio,
            class_names=cfg.CLASS_NAMES,
            random_state=42
        )

    method = getattr(Config, "OVERSAMPLE_METHOD", None)
    if method is not None:
        from src.training.data_utils import apply_smote_tomek
        X_train, y_train = apply_smote_tomek(
            X_train=X_train,
            y_train=y_train,
            method=method,
            sampling_strategy=getattr(Config, "OVERSAMPLE_STRATEGY", "auto"),
            class_names=cfg.CLASS_NAMES
        )
        new_labels = np.argmax(y_train, axis=1)
        dist = {cfg.CLASS_NAMES[cls]: count for cls, count in zip(*np.unique(new_labels, return_counts=True))}
        print(f" -> Oversampled training distribution: {dist}")
    return X_train, y_train


# =====================================================================
# RUN EXPERIMENT
# =====================================================================

def run_experiment(
    loader,
    X_eval,
    y_eval,
    experiment_name,
    filters,
    kernels,
    dilations,
    temporal_mode
):
    print("\n" + "=" * 80)
    print(f"RUNNING ({SCHEME} / {LABEL_SCHEME}): {experiment_name}")
    print("=" * 80)

    exp_dir = create_experiment_dir(
        ROOT_EXP_DIR,
        f"{SCHEME}_{LABEL_SCHEME}",
        experiment_name
    )

    config_dict = {
        "experiment_name": experiment_name,
        "scheme": SCHEME,
        "label_scheme": LABEL_SCHEME,
        "train_dataset": TRAIN_DATASET,
        "test_dataset": TEST_DATASET,
        "class_names": loader.class_names,
        "filters": filters,
        "kernels": kernels,
        "dilations": dilations,
        "temporal_mode": temporal_mode,
        "batch_size": Config.BATCH_SIZE,
        "epochs": Config.EPOCHS,
        "learning_rate": Config.LEARNING_RATE,
        "loss": "Focal Loss" if SCHEME == "softmax" else "BinaryCrossentropy",
        "label_smoothing": Config.LABEL_SMOOTHING,
        "use_augmentation": Config.USE_AUGMENTATION,
        "mixup_alpha": Config.MIXUP_ALPHA,
        "optimizer": Config.OPTIMIZER,
        "undersample_ratio": getattr(Config, "UNDERSAMPLE_RATIO", None),
        "oversample_method": getattr(Config, "OVERSAMPLE_METHOD", None),
        "oversample_strategy": str(getattr(Config, "OVERSAMPLE_STRATEGY", None))
    }
    save_experiment_config(config_dict, exp_dir)

    # ---- Model ----
    model = build_dynamic_cnn(
        filters=filters,
        kernels=kernels,
        dilations=dilations,
        temporal_mode=temporal_mode,
        use_separable=Config.USE_SEPARABLE_CONV,
        stochastic_depth_rate=Config.STOCHASTIC_DEPTH_RATE,
        scheme="multilabel" if SCHEME == "sigmoid" else "multiclass"
    )

    lr_schedule = CosineDecay(
        initial_learning_rate=Config.LEARNING_RATE,
        decay_steps=Config.COSINE_DECAY_STEPS,
        alpha=Config.COSINE_ALPHA
    )
    if SCHEME == "sigmoid":
        model.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=["binary_accuracy"]
        )
    else:
        model.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            loss=get_loss(),
            metrics=["accuracy"]
        )

    # Warm-up / trace graph & param count
    model(tf.random.normal((1, Config.INPUT_SHAPE[0], Config.INPUT_SHAPE[1])))
    total_params = model.count_params()

    # ---- Data ----
    splits = loader.build_splits()
    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val = splits["X_val"]
    y_val = splits["y_val"]

    X_train_res, y_train_res = balance_training_data(X_train, y_train)

    class_weight = None
    if SCHEME == "softmax":
        class_weight = build_class_weights(y_train_res)

    train_ds = create_tf_dataset(
        X_train_res,
        y_train_res,
        batch_size=Config.BATCH_SIZE,
        is_training=True,
        use_augmentation=Config.USE_AUGMENTATION,
        use_mixup=Config.USE_MIXUP,
        mixup_alpha=Config.MIXUP_ALPHA
    )
    val_ds = create_tf_dataset(
        X_val,
        y_val,
        batch_size=Config.BATCH_SIZE,
        is_training=False
    )

    model_path = os.path.join(exp_dir, "best_model.keras")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=Config.EPOCHS,
        callbacks=get_callbacks(exp_dir, model_path),
        class_weight=class_weight,
        verbose=1
    )

    save_training_history(history.history, exp_dir)
    save_model_summary(model, exp_dir)
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    # ---- Evaluation ----
    if SCHEME == "sigmoid":
        thresholds = perform_threshold_tuning(model, X_val, y_val, loader.class_names)
        metrics, y_pred_bin, y_prob = evaluate_sigmoid(
            model, np.array(X_eval, dtype=np.float32), np.array(y_eval, dtype=np.float32),
            thresholds, loader.class_names
        )
        metrics["Total_Params"] = total_params
        metrics["Model_Size_MB"] = model_size_mb
        metrics["Undersample_Ratio"] = str(getattr(Config, "UNDERSAMPLE_RATIO", None))
        metrics["Oversample_Method"] = str(getattr(Config, "OVERSAMPLE_METHOD", None))
        metrics["Oversample_Strategy"] = str(getattr(Config, "OVERSAMPLE_STRATEGY", None))

        pd.DataFrame([metrics]).to_csv(os.path.join(exp_dir, "metrics.csv"), index=False)

        rep_str = classification_report(y_eval, y_pred_bin, target_names=loader.class_names, zero_division=0)
        with open(os.path.join(exp_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(rep_str)

        save_multilabel_confusion_matrix(y_eval, y_pred_bin, loader.class_names, exp_dir)
        save_multilabel_misclassified(X_eval, y_eval, y_prob, thresholds, loader.class_names, exp_dir)
    else:
        metrics = calculate_ml_metrics(
            model=model,
            X_test=np.array(X_eval, dtype=np.float32),
            y_test=np.array(y_eval, dtype=np.float32),
            exp_name=experiment_name
        )
        metrics["Scheme"] = SCHEME
        metrics["Label_Scheme"] = LABEL_SCHEME
        metrics["Train_Dataset"] = TRAIN_DATASET
        metrics["Test_Dataset"] = TEST_DATASET
        metrics["Total_Params"] = total_params
        metrics["Model_Size_MB"] = model_size_mb
        metrics["Undersample_Ratio"] = str(getattr(Config, "UNDERSAMPLE_RATIO", None))
        metrics["Oversample_Method"] = str(getattr(Config, "OVERSAMPLE_METHOD", None))
        metrics["Oversample_Strategy"] = str(getattr(Config, "OVERSAMPLE_STRATEGY", None))

        pd.DataFrame([metrics]).to_csv(os.path.join(exp_dir, "metrics.csv"), index=False)

        y_prob = model.predict(np.array(X_eval, dtype=np.float32), batch_size=Config.BATCH_SIZE, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        y_true = np.argmax(np.array(y_eval, dtype=np.float32), axis=1)

        save_confusion_matrix(
            y_true=y_true, y_pred=y_pred, class_names=loader.class_names,
            save_dir=exp_dir, normalize=False
        )
        save_classification_report(
            y_true=y_true, y_pred=y_pred, class_names=loader.class_names, save_dir=exp_dir
        )
        save_misclassified_samples(
            X=X_eval, y_true=y_true, y_pred=y_pred, class_names=loader.class_names,
            save_dir=exp_dir, max_samples=20
        )

    # ---- Master tracker ----
    master_csv_path = os.path.join(ROOT_EXP_DIR, Config.MASTER_TRACKER_CSV)
    update_master_tracker(metrics, master_csv_path)

    print("\nExperiment Finished")
    del model
    K.clear_session()
    gc.collect()
    tf.compat.v1.reset_default_graph()


# =====================================================================
# MAIN LOOP GRID SEARCH
# =====================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(f"UNIFIED EXPERIMENT RUNNER | SCHEME={SCHEME} | LABEL={LABEL_SCHEME} "
          f"| TRAIN={TRAIN_DATASET} | TEST={TEST_DATASET}")
    print("=" * 80)

    # Train-loaders across the active dataset grid (PTB-XL only).
    # For Chapman the manifest/signal folder is fixed.
    folder_keys = (
        Config.ACTIVE_DATASETS
        if TRAIN_DATASET == "PTBXL"
        else [CHAPMAN_FOLDER_KEY]
    )

    loaders = {}
    for folder_key in folder_keys:
        print(f"\n[Data] Resolving splits for {TRAIN_DATASET} @ {folder_key}")
        loaders[folder_key] = load_training_data(folder_key)

    for filter_name, filters in Config.FILTER_SPACES.items():
        for kernel_name, kernels in Config.KERNEL_SPACES.items():
            for dilation_name, dilations in Config.DILATION_SPACES.items():
                for temporal_mode in Config.TEMPORAL_MODELS:
                    for folder_key, loader in loaders.items():
                        exp_name = (
                            f"{SCHEME}_{LABEL_SCHEME}_{folder_key}"
                            f"__{filter_name}__{kernel_name}__{dilation_name}"
                            f"__{temporal_mode}"
                        )
                        if not IS_CROSS_DATASET and getattr(Config, "UNDERSAMPLE_RATIO", None) is not None:
                            exp_name += f"__us_{Config.UNDERSAMPLE_RATIO}"
                        if IS_CROSS_DATASET:
                            exp_name += f"__test_{TEST_DATASET}"

                        # Evaluation set: same-loader test split, or target dataset.
                        X_eval, y_eval = load_eval_data(loader)

                        run_experiment(
                            loader=loader,
                            X_eval=X_eval,
                            y_eval=y_eval,
                            experiment_name=exp_name,
                            filters=filters,
                            kernels=kernels,
                            dilations=dilations,
                            temporal_mode=temporal_mode
                        )

    print("\nALL EXPERIMENTS FINISHED")