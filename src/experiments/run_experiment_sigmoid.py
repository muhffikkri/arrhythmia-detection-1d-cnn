# =====================================================================
# FILE: src/run_experiment_sigmoid.py
# PURPOSE: MULTI-LABEL ECG RESEARCH PIPELINE WITH FULL TRACKING LOGS
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
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, multilabel_confusion_matrix, classification_report

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
import tensorflow.keras.backend as K

# Add project root directory to python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# LOCAL IMPORTS
from src.config import config as cfg
from src.config import config_labels as ml_cfg
from src.config.experiment_configs import Config
from src.models.model_factory import build_dynamic_cnn
from src.training.data_utils import create_tf_dataset
from src.training.tracker import create_experiment_dir, save_experiment_config, save_training_history

warnings.filterwarnings("ignore")

# =====================================================================
# REPRODUCIBILITY SEEDS
# =====================================================================
def set_global_seeds(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_global_seeds(42)

ROOT_EXP_DIR = ml_cfg.MULTILABEL_OUTPUT_DIR
os.makedirs(ROOT_EXP_DIR, exist_ok=True)

# =====================================================================
# DATA FILTERING & MULTI-HOT MAPPING
# =====================================================================
def load_curated_multilabel_dataset():
    manifest_path = os.path.join(cfg.RESAMPLE_BASE, "manifest_ptbxl.csv")
    df_manifest = pd.read_csv(manifest_path)
    
    df_filtered = df_manifest[df_manifest["target_class"].isin(ml_cfg.TARGET_CLASSES)].reset_index(drop=True)
    
    X_splits = {"train": [], "val": [], "test": []}
    y_splits = {"train": [], "val": [], "test": []}
    
    print(f"\n[Data Setup] Curating Multi-Label Dataset for: {ml_cfg.TARGET_CLASSES}")
    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
        file_path = row["path_e2_clean"]
        if not os.path.exists(file_path):
            filename = row["filename_npy"]
            file_path = os.path.join(cfg.SUB_FOLDERS["E2_clean_100_to_250"], filename)
            
        if not os.path.exists(file_path):
            continue
            
        try:
            signal = np.load(file_path).astype(np.float32)
            multi_hot = np.zeros(ml_cfg.NUM_CLASSES, dtype=np.float32)
            class_idx = ml_cfg.TARGET_CLASSES.index(row["target_class"])
            multi_hot[class_idx] = 1.0
            
            fold = int(row["strat_fold"])
            if fold in range(1, 9): split = "train"
            elif fold == 9: split = "val"
            elif fold == 10: split = "test"
            else: continue
                
            X_splits[split].append(signal)
            y_splits[split].append(multi_hot)
        except:
            continue

    return {
        "X_train": np.array(X_splits["train"], dtype=np.float32),
        "X_val": np.array(X_splits["val"], dtype=np.float32),
        "X_test": np.array(X_splits["test"], dtype=np.float32),
        "y_train": np.array(y_splits["train"], dtype=np.float32),
        "y_val": np.array(y_splits["val"], dtype=np.float32),
        "y_test": np.array(y_splits["test"], dtype=np.float32)
    }

# =====================================================================
# POST-TRAINING PER-CLASS THRESHOLD TUNING
# =====================================================================
def perform_threshold_tuning(model, X_val, y_val):
    print("\n[Optimization] Optimizing Per-Class Decision Thresholds via Validation Set...")
    y_val_pred_prob = model.predict(X_val, batch_size=Config.BATCH_SIZE, verbose=0)
    
    best_thresholds = []
    threshold_range = np.linspace(0.1, 0.9, 81)
    
    for cl_idx, class_name in enumerate(ml_cfg.TARGET_CLASSES):
        best_f1 = 0.0
        best_th = 0.5
        for th in threshold_range:
            y_pred_bin = (y_val_pred_prob[:, cl_idx] >= th).astype(int)
            y_true_bin = y_val[:, cl_idx].astype(int)
            score = f1_score(y_true_bin, y_pred_bin, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_th = th
        print(f" -> Best Threshold for {class_name}: {best_th:.2f} (Val F1: {best_f1:.4f})")
        best_thresholds.append(best_th)
    return best_thresholds

# =====================================================================
# MEDICAL GRADE MULTI-LEAD MULTI-LABEL MISCLASSIFIED VISUALIZATION
# =====================================================================
def save_multilabel_misclassified_with_confidence(X, y_true, y_pred_prob, thresholds, save_dir, max_samples=20):
    """
    Menyimpan visualisasi sampel salah klasifikasi multi-label.
    Memisahkan setiap lead secara vertikal dan mencetak nilai keyakinan seluruh neuron Sigmoid.
    """
    print("\n[Visualization] Generating Medical-Grade Multi-Lead Plot for Errors...")
    save_path_dir = os.path.join(save_dir, "misclassified")
    os.makedirs(save_path_dir, exist_ok=True)
    
    # Hitung matriks biner keputusan berdasarkan threshold adaptif
    y_pred_bin = np.zeros_like(y_pred_prob)
    for idx in range(ml_cfg.NUM_CLASSES):
        y_pred_bin[:, idx] = (y_pred_prob[:, idx] >= thresholds[idx]).astype(int)
        
    # Temukan indeks di mana keputusan biner tidak sama persis dengan target asli
    misclassified_indices = np.where(np.any(y_true != y_pred_bin, axis=1))[0]
    
    for count, idx in enumerate(misclassified_indices[:max_samples]):
        signal = X[idx]
        num_leads = signal.shape[-1]
        
        fig, axes = plt.subplots(nrows=num_leads, ncols=1, figsize=(15, 2 * num_leads), sharex=True)
        if num_leads == 1: axes = [axes]
            
        for ch in range(num_leads):
            ax = axes[ch]
            ax.plot(signal[:, ch], color='#1c1c1e', linewidth=1.1)
            ax.set_ylabel(f"Lead {ch+1}", fontsize=9, fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.6)
            
        # Membangun string deskripsi tingkat keyakinan (Confidence Score) masing-masing kelas
        title_lines = []
        for c_i, c_name in enumerate(ml_cfg.TARGET_CLASSES):
            status_true = "POS" if y_true[idx, c_i] == 1 else "NEG"
            status_pred = "POS" if y_pred_bin[idx, c_i] == 1 else "NEG"
            title_lines.append(f"{c_name}(True:{status_true}|Pred:{status_pred}|Conf:{y_pred_prob[idx, c_i]:.3f})")
            
        plt.suptitle(
            f"ERROR SAMPLE #{count} | ANALYSIS MATRIX:\n" + " | ".join(title_lines[:2]) + "\n" + " | ".join(title_lines[2:]),
            fontsize=10, fontweight='bold', color='#ff453a', y=0.98
        )
        plt.tight_layout()
        plt.savefig(os.path.join(save_path_dir, f"error_{count}_multilabel.png"), dpi=150)
        plt.close(fig)

# =====================================================================
# PIPELINE EXECUTION ENGINE
# =====================================================================
def run_multilabel_experiment(PTB_DATA, experiment_name, filters, kernels, dilations, temporal_mode):
    print("\n" + "="*80)
    print(f"RUNNING MULTI-LABEL EXPERIMENT: {experiment_name}")
    print("="*80)

    exp_dir = create_experiment_dir(ROOT_EXP_DIR, "multilabel_branch", experiment_name)
    
    # 1. Save Config JSON
    config_dict = {
        "experiment_name": experiment_name, "filters": filters, "kernels": kernels,
        "dilations": dilations, "temporal_mode": temporal_mode, "batch_size": Config.BATCH_SIZE,
        "epochs": Config.EPOCHS, "learning_rate": Config.LEARNING_RATE, "loss": "BinaryCrossentropy"
    }
    save_experiment_config(config_dict, exp_dir)
    
    # 2. Compile Model
    model = build_dynamic_cnn(
        filters=filters, kernels=kernels, dilations=dilations,
        temporal_mode=temporal_mode, use_separable=Config.USE_SEPARABLE_CONV,
        stochastic_depth_rate=Config.STOCHASTIC_DEPTH_RATE,
        scheme="multilabel"
    )
    
    lr_schedule = CosineDecay(initial_learning_rate=Config.LEARNING_RATE, decay_steps=Config.COSINE_DECAY_STEPS, alpha=Config.COSINE_ALPHA)
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['binary_accuracy'])
    
    # Warmup
    model(tf.random.normal((1, Config.INPUT_SHAPE[0], Config.INPUT_SHAPE[1])))
    total_params = model.count_params()
    
    # 3. Training Callbacks & Fit
    train_ds = create_tf_dataset(PTB_DATA["X_train"], PTB_DATA["y_train"], batch_size=Config.BATCH_SIZE, is_training=True, use_augmentation=Config.USE_AUGMENTATION, use_mixup=Config.USE_MIXUP, mixup_alpha=Config.MIXUP_ALPHA)
    val_ds = create_tf_dataset(PTB_DATA["X_val"], PTB_DATA["y_val"], batch_size=Config.BATCH_SIZE, is_training=False)
    
    model_path = os.path.join(exp_dir, "best_model.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=model_path, monitor="val_loss", save_best_only=True, verbose=1),
        CSVLogger(os.path.join(exp_dir, "training_log.csv"), append=False)
    ]
    
    history = model.fit(train_ds, validation_data=val_ds, epochs=Config.EPOCHS, callbacks=callbacks, verbose=1)
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    # Save History Curves
    save_training_history(history.history, exp_dir)
    
    # 4. Post-Training Tuning & Test Set Isolation
    optimized_thresholds = perform_threshold_tuning(model, PTB_DATA["X_val"], PTB_DATA["y_val"])
    
    X_test_iso = np.array(PTB_DATA["X_test"], dtype=np.float32)
    y_test_iso = np.array(PTB_DATA["y_test"], dtype=np.float32)
    
    y_test_pred_prob = model.predict(X_test_iso, batch_size=Config.BATCH_SIZE, verbose=0)
    
    # Convert To Binary Matrix untuk Report & Confusion Matrix
    y_test_pred_bin = np.zeros_like(y_test_pred_prob)
    for idx in range(ml_cfg.NUM_CLASSES):
        y_test_pred_bin[:, idx] = (y_test_pred_prob[:, idx] >= optimized_thresholds[idx]).astype(int)

    # 5. SAVE MULTI-LABEL CLASSIFICATION REPORT (UTF-8 SAFE)
    print("\n[Metrics] Writing Classification Report...")
    rep_str = classification_report(y_test_iso, y_test_pred_bin, target_names=ml_cfg.TARGET_CLASSES, zero_division=0)
    with open(os.path.join(exp_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(rep_str)

    # 6. SAVE MULTILABEL CONFUSION MATRIX (PER-CLASS BINARY)
    print("[Metrics] Generating Multilabel Confusion Matrix...")
    mcm = multilabel_confusion_matrix(y_test_iso, y_test_pred_bin)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for idx, class_name in enumerate(ml_cfg.TARGET_CLASSES):
        ax = axes[idx//2, idx%2]
        cm = mcm[idx]
        ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
        for g_i in range(cm.shape[0]):
            for g_j in range(cm.shape[1]):
                ax.text(x=g_j, y=g_i, s=cm[g_i, g_j], va='center', ha='center', fontsize=12, fontweight='bold')
        ax.set_title(f"CM Biner: {class_name}", fontweight='bold')
        ax.set_xticklabels(['', 'Neg', 'Pos'])
        ax.set_yticklabels(['', 'Neg', 'Pos'])
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # 7. SAVE VISUALIZATION
    save_multilabel_misclassified_with_confidence(X_test_iso, y_test_iso, y_test_pred_prob, optimized_thresholds, exp_dir)

    # 8. EXPORT METRICS CSV
    final_metrics = {"Experiment": experiment_name, "Total_Params": total_params, "Model_Size_MB": model_size_mb, "Thresholds_Assigned": str(optimized_thresholds)}
    for i, cls in enumerate(ml_cfg.TARGET_CLASSES):
        final_metrics[f"F1_{cls}"] = float(f1_score(y_test_iso[:, i], y_test_pred_bin[:, i], zero_division=0))
        final_metrics[f"AUC_{cls}"] = float(roc_auc_score(y_test_iso[:, i], y_test_pred_prob[:, i]))
        
    pd.DataFrame([final_metrics]).to_csv(os.path.join(exp_dir, "metrics.csv"), index=False)
    
    del model; K.clear_session(); gc.collect()
    print(f"[Finished] All outputs cleanly dumped.")

# =====================================================================
# MAIN LOOP GRID SEARCH
# =====================================================================
if __name__ == "__main__":
    PTB_MULTILABEL_DATA = load_curated_multilabel_dataset()
    
    FILTER_CONFIGS = Config.FILTER_SPACES
    KERNEL_CONFIGS = Config.KERNEL_SPACES
    DILATION_CONFIGS = Config.DILATION_SPACES
    TEMPORALS = Config.TEMPORAL_MODELS

    for filter_name, filters in FILTER_CONFIGS.items():
        for kernel_name, kernels in KERNEL_CONFIGS.items():
            for dilation_name, dilations in DILATION_CONFIGS.items():
                for temporal_mode in TEMPORALS:
                    exp_name = f"MULTILABEL_E2__{filter_name}__{kernel_name}__{dilation_name}__{temporal_mode}"
                    run_multilabel_experiment(
                        PTB_DATA=PTB_MULTILABEL_DATA, experiment_name=exp_name,
                        filters=filters, kernels=kernels, dilations=dilations,
                        temporal_mode=temporal_mode
                    )