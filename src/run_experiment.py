# =====================================================================
# FILE: run_experiment.py
# ECG RESEARCH PIPELINE — TRACKING + LOGGING VERSION
# =====================================================================

import os
import gc
import sys
import json
import random
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm

from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    CSVLogger
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay

import tensorflow.keras.backend as K

# =====================================================================
# LOCAL IMPORTS
# =====================================================================

sys.path.append(os.getcwd())

import config as cfg
from experiment_configs import Config
from model_factory import build_dynamic_cnn
from loss_factory import get_loss_function
from evaluator import calculate_ml_metrics
from data_utils import create_tf_dataset

from tracker import (
    create_experiment_dir,
    save_experiment_config,
    save_training_history,
    save_confusion_matrix,
    save_classification_report,
    save_model_summary,
    update_master_tracker
)

from visualization import save_misclassified_samples

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

ROOT_EXP_DIR = os.path.join(
    cfg.OUTPUT_DIR,
    "research_experiments"
)
os.makedirs(ROOT_EXP_DIR, exist_ok=True)

# =====================================================================
# LOSS FUNCTION
# =====================================================================

FOCAL_CFG = Config.FOCAL_CONFIGS[0]

LOSS_FN = get_loss_function(
    strategy="focal",
    gamma=FOCAL_CFG["gamma"],
    alpha=FOCAL_CFG["alpha"],
    label_smoothing=Config.LABEL_SMOOTHING
)

# =====================================================================
# CALLBACKS
# =====================================================================

def get_callbacks(exp_dir, model_path):
    csv_log_path = os.path.join(
        exp_dir,
        "training_log.csv"
    )

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
            csv_log_path,
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

    return {
        i: float(w)
        for i, w in enumerate(weights)
    }

# =====================================================================
# LOAD PTB DATASET
# =====================================================================

def load_ptb_dataset(folder_key):
    manifest_path = os.path.join(
        cfg.RESAMPLE_BASE,
        "manifest_ptbxl.csv"
    )
    df_manifest = pd.read_csv(manifest_path)

    X = {"train": [], "val": [], "test": []}
    y = {"train": [], "val": [], "test": []}

    base_folder = cfg.SUB_FOLDERS[folder_key]
    lb = LabelBinarizer()
    lb.fit(cfg.CLASS_NAMES)

    print(f"\nLoading PTBXL from: {folder_key}")

    for _, row in tqdm(df_manifest.iterrows(), total=len(df_manifest)):
        file_path = os.path.join(
            base_folder,
            row["filename_npy"]
        )

        if not os.path.exists(file_path):
            continue

        try:
            signal = np.load(file_path).astype(np.float32)
            label = row["target_class"]
            fold = int(row["strat_fold"])

            if fold in range(1, 9):
                split = "train"
            elif fold == 9:
                split = "val"
            elif fold == 10:
                split = "test"
            else:
                continue

            X[split].append(signal)
            y[split].append(label)
        except:
            continue

    for split in ["train", "val", "test"]:
        X[split] = np.array(X[split], dtype=np.float32)
        y[split] = lb.transform(y[split]).astype(np.float32)

    return {
        "X_train": X["train"],
        "X_val": X["val"],
        "X_test": X["test"],
        "y_train": y["train"],
        "y_val": y["val"],
        "y_test": y["test"]
    }

# =====================================================================
# LOAD CHAPMAN
# =====================================================================

def load_chapman():
    manifest_path = os.path.join(
        cfg.RESAMPLE_BASE,
        "manifest_chapman.csv"
    )
    df = pd.read_csv(manifest_path)

    X = []
    y = []

    lb = LabelBinarizer()
    lb.fit(cfg.CLASS_NAMES)
    base_folder = cfg.SUB_FOLDERS["CHAPMAN_CLEAN_250HZ"]

    print("\nLoading Chapman Dataset")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        file_path = os.path.join(
            base_folder,
            row["filename_npy"]
        )

        if not os.path.exists(file_path):
            continue

        try:
            signal = np.load(file_path).astype(np.float32)
            X.append(signal)
            y.append(row["target_class"])
        except:
            continue

    X = np.array(X, dtype=np.float32)
    y = lb.transform(y).astype(np.float32)

    return X, y

# =====================================================================
# TRAIN MODEL
# =====================================================================

def train_model(
    train_ds,
    val_ds,
    filters,
    kernels,
    dilations,
    temporal_mode,
    experiment_name,
    exp_dir,
    class_weight=None
):
    print("\nBuilding Model...")

    model = build_dynamic_cnn(
        filters=filters,
        kernels=kernels,
        dilations=dilations,
        temporal_mode=temporal_mode,
        use_separable=Config.USE_SEPARABLE_CONV,
        stochastic_depth_rate=Config.STOCHASTIC_DEPTH_RATE
    )

    lr_schedule = CosineDecay(
        initial_learning_rate=Config.LEARNING_RATE,
        decay_steps=Config.COSINE_DECAY_STEPS,
        alpha=Config.COSINE_ALPHA
    )

    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss=LOSS_FN, metrics=["accuracy"])

    # Warm-up / Trace call graph
    model(tf.random.normal((1, Config.INPUT_SHAPE[0], Config.INPUT_SHAPE[1])))
    total_params = model.count_params()

    model_path = os.path.join(exp_dir, "best_model.keras")
    callbacks = get_callbacks(exp_dir, model_path)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=Config.EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    save_training_history(history.history, exp_dir)
    save_model_summary(model, exp_dir)

    return model, history, total_params, file_size_mb

# =====================================================================
# RUN EXPERIMENT
# =====================================================================

def run_experiment(
    PTB_DATA,
    X_chapman,
    Y_chapman,
    experiment_name,
    filters,
    kernels,
    dilations,
    temporal_mode
):
    print("\n" + "="*80)
    print(f"RUNNING : {experiment_name}")
    print("="*80)

    exp_dir = create_experiment_dir(
        ROOT_EXP_DIR,
        Config.EXPERIMENT_GROUP,
        experiment_name
    )

    config_dict = {
        "experiment_name": experiment_name,
        "filters": filters,
        "kernels": kernels,
        "dilations": dilations,
        "temporal_mode": temporal_mode,
        "batch_size": Config.BATCH_SIZE,
        "epochs": Config.EPOCHS,
        "learning_rate": Config.LEARNING_RATE,
        "loss": "Focal Loss",
        "label_smoothing": Config.LABEL_SMOOTHING,
        "use_augmentation": Config.USE_AUGMENTATION,
        "mixup_alpha": Config.MIXUP_ALPHA,
        "optimizer": Config.OPTIMIZER
    }

    save_experiment_config(config_dict, exp_dir)
    class_weights = build_class_weights(PTB_DATA["y_train"])

    train_ds = create_tf_dataset(
        PTB_DATA["X_train"],
        PTB_DATA["y_train"],
        batch_size=Config.BATCH_SIZE,
        is_training=True,
        use_augmentation=Config.USE_AUGMENTATION,
        use_mixup=Config.USE_MIXUP,
        mixup_alpha=Config.MIXUP_ALPHA
    )

    val_ds = create_tf_dataset(
        PTB_DATA["X_val"],
        PTB_DATA["y_val"],
        batch_size=Config.BATCH_SIZE,
        is_training=False
    )

    model, history, total_params, model_size = train_model(
        train_ds=train_ds,
        val_ds=val_ds,
        filters=filters,
        kernels=kernels,
        dilations=dilations,
        temporal_mode=temporal_mode,
        experiment_name=experiment_name,
        exp_dir=exp_dir,
        class_weight=class_weights
    )

    # =========================================================
    # EVALUATION
    # =========================================================

    print("\n[Evaluation] Locking Test Set Dimension...")
    
    # Kunci data uji ke dalam array lokal tunggal untuk mencegah pembacaan ulang graph
    X_test_eval = np.array(PTB_DATA["X_test"], dtype=np.float32)
    y_test_eval = np.array(PTB_DATA["y_test"], dtype=np.float32)

    # 1. Hitung seluruh metrik utama via Evaluator
    metrics = calculate_ml_metrics(
        model=model,
        X_test=X_test_eval,
        y_test=y_test_eval,
        exp_name=experiment_name
    )

    metrics["Total_Params"] = total_params
    metrics["Model_Size_MB"] = model_size

    # # 2. Ambil metrik hardware untuk pembuktian edge-computing PKM
    # hw_metrics = evaluate_hardware_efficiency(model, X_test_eval, experiment_name)
    # metrics["MACs_Millions"] = hw_metrics["MACs_Millions"]
    # metrics["Inference_Time_ms"] = hw_metrics["Inference_Time_ms"]

    # # 3. Jalankan Stress Test untuk menguji ketahanan model terhadap noise
    # stress_metrics = evaluate_stress_test(model, X_test_eval, y_test_eval)
    # metrics["Bal_Acc_AWGN_10dB"] = stress_metrics["Bal_Acc_AWGN_10dB"]
    # metrics["Bal_Acc_Baseline_Wander"] = stress_metrics["Bal_Acc_Baseline_Wander"]

    # Simpan ke CSV Eksperimen
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(exp_dir, "metrics.csv"), index=False)

    # 4. Ambil nilai argmax murni dari prediksi yang sama untuk Confusion Matrix
    y_pred_prob = model.predict(X_test_eval, batch_size=Config.BATCH_SIZE, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test_eval, axis=1)

    # Cetak Confusion Matrix dengan jumlah yang pasti sinkron
    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=cfg.CLASS_NAMES,
        save_dir=exp_dir,
        normalize=False
    )

    #  =========================================================
    # CLASSIFICATION REPORT
    #  =========================================================

    save_classification_report(
        y_true=y_true,
        y_pred=y_pred,
        class_names=cfg.CLASS_NAMES,
        save_dir=exp_dir
    )

    # =========================================================
    # MISCLASSIFIED SAMPLES
    # =========================================================
    save_misclassified_samples(
        X=PTB_DATA["X_test"],
        y_true=y_true,
        y_pred=y_pred,
        class_names=cfg.CLASS_NAMES,
        save_dir=exp_dir,
        max_samples=20
    )

    # =========================================================
    # MASTER TRACKER
    # =========================================================

    master_csv_path = os.path.join(
        ROOT_EXP_DIR,
        Config.MASTER_TRACKER_CSV
    )

    update_master_tracker(metrics, master_csv_path)

    print("\nExperiment Finished")
    del model
    K.clear_session()
    gc.collect()
    tf.compat.v1.reset_default_graph()

# =====================================================================
# MAIN LOOP GRID SEARCH EXPLORATION
# =====================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("LOADING DATASETS")
    print("="*80)

    X_chapman, Y_chapman = load_chapman()

    DATASETS = {
        "EXP_100HZ_TO_250HZ": load_ptb_dataset("E2_clean_100_to_250"),
        "EXP_500HZ_TO_250HZ": load_ptb_dataset("E3_clean_500_to_250")
    }

    FILTER_CONFIGS = Config.FILTER_SPACES
    KERNEL_CONFIGS = Config.KERNEL_SPACES
    DILATION_CONFIGS = Config.DILATION_SPACES
    TEMPORALS = Config.TEMPORAL_MODELS

    for dataset_name, dataset_data in DATASETS.items():
        for filter_name, filters in FILTER_CONFIGS.items():
            for kernel_name, kernels in KERNEL_CONFIGS.items():
                for dilation_name, dilations in DILATION_CONFIGS.items():
                    for temporal_mode in TEMPORALS:
                        exp_name = (
                            f"{dataset_name}"
                            f"__{filter_name}"
                            f"__{kernel_name}"
                            f"__{dilation_name}"
                            f"__{temporal_mode}"
                        )

                        run_experiment(
                            PTB_DATA=dataset_data,
                            X_chapman=X_chapman,
                            Y_chapman=Y_chapman,
                            experiment_name=exp_name,
                            filters=filters,
                            kernels=kernels,
                            dilations=dilations,
                            temporal_mode=temporal_mode
                        )

    print("\nALL EXPERIMENTS FINISHED")