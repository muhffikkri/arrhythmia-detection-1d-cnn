# =====================================================================
# FILE: src/run_experiment_multilabel.py
# PURPOSE: MULTI-LABEL ECG RESEARCH PIPELINE WITH THRESHOLD TUNING
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
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
import tensorflow.keras.backend as K

# LOCAL IMPORTS
sys.path.append(os.getcwd())
import config as cfg
import config_ml as ml_cfg
from experiment_configs import Config
from model_factory import build_dynamic_cnn
from data_utils import create_tf_dataset
from tracker import create_experiment_dir, save_experiment_config

# IMPORT MODUL VISUALISASI 3 PLOT VERTIKAL ASLI
from visualization import save_misclassified_samples

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

# ROOT OUTPUT
ROOT_EXP_DIR = ml_cfg.MULTILABEL_OUTPUT_DIR
os.makedirs(ROOT_EXP_DIR, exist_ok=True)

# =====================================================================
# DATA FILTERING & MULTI-HOT MAPPING
# =====================================================================
def load_curated_multilabel_dataset():
    """
    Memuat data PTB-XL berdasarkan manifest_ptbxl.csv.
    Menggunakan data e2_clean yang sudah siap pakai (100Hz -> 250Hz Clean).
    """
    manifest_path = os.path.join(cfg.RESAMPLE_BASE, "manifest_ptbxl.csv")
    df_manifest = pd.read_csv(manifest_path)
    
    # Filter: Ambil baris yang memiliki target_class di dalam target kita
    df_filtered = df_manifest[df_manifest["target_class"].isin(ml_cfg.TARGET_CLASSES)].reset_index(drop=True)
    
    X_splits = {"train": [], "val": [], "test": []}
    y_splits = {"train": [], "val": [], "test": []}
    
    print(f"\n[Data Setup] Curating Multi-Label Dataset for: {ml_cfg.TARGET_CLASSES}")
    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
        file_path = row["path_e2_clean"]
        
        if not os.path.exists(file_path):
            continue
            
        try:
            signal = np.load(file_path).astype(np.float32)
            
            # Pembentukan Vektor Multi-Hot [0, 0, 0, 0]
            multi_hot = np.zeros(ml_cfg.NUM_CLASSES, dtype=np.float32)
            class_idx = ml_cfg.TARGET_CLASSES.index(row["target_class"])
            multi_hot[class_idx] = 1.0
            
            # Stratifikasi Berbasis Fold Asli
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
    """
    Mencari ambang batas (threshold) terbaik untuk masing-masing kelas
    pada dataset validasi untuk memaksimalkan nilai F1-Score Macro.
    """
    print("\n[Optimization] Optimizing Per-Class Decision Thresholds via Validation Set...")
    y_val_pred_prob = model.predict(X_val, batch_size=Config.BATCH_SIZE, verbose=0)
    
    best_thresholds = []
    threshold_range = np.linspace(0.1, 0.9, 81)  # Scan 0.10 s.d 0.90 (step 0.01)
    
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
# EVALUATE WITH TUNED THRESHOLDS & INTERCEPT "OTHERS"
# =====================================================================
def evaluate_multi_label_metrics(y_true, y_pred_prob, thresholds):
    """
    Menghitung metrik akhir di test set. 
    Mengimplementasikan pendeteksian otomatis kelas 'Others'.
    """
    results = {}
    total_samples = y_true.shape[0]
    
    # 1. Evaluasi Standar untuk 4 Kelas Inti
    for i, cls in enumerate(ml_cfg.TARGET_CLASSES):
        th = thresholds[i]
        bin_pred = (y_pred_prob[:, i] >= th).astype(int)
        bin_true = y_true[:, i].astype(int)
        
        results[f"F1_{cls}"] = float(f1_score(bin_true, bin_pred, zero_division=0))
        results[f"Precision_{cls}"] = float(precision_score(bin_true, bin_pred, zero_division=0))
        results[f"Recall_{cls}"] = float(recall_score(bin_true, bin_pred, zero_division=0))
        try:
            results[f"AUC_{cls}"] = float(roc_auc_score(y_true[:, i], y_pred_prob[:, i]))
        except:
            results[f"AUC_{cls}"] = np.nan

    # 2. INTERSEPSI KELAS OTHERS AUTOMATIC DETECTOR
    others_pred_mask = np.all([y_pred_prob[:, idx] < thresholds[idx] for idx in range(ml_cfg.NUM_CLASSES)], axis=0)
    
    others_detected_count = np.sum(others_pred_mask)
    results["Others_Detected_Count"] = int(others_detected_count)
    results["Others_Detection_Rate"] = float(others_detected_count / total_samples)
            
    return results

# =====================================================================
# PIPELINE EXECUTION ENGINE
# =====================================================================
def run_multilabel_experiment(PTB_DATA, experiment_name, filters, kernels, dilations, temporal_mode):
    print("\n" + "="*80)
    print(f"RUNNING MULTI-LABEL EXPERIMENT: {experiment_name}")
    print("="*80)

    exp_dir = create_experiment_dir(ROOT_EXP_DIR, "multilabel_branch", experiment_name)
    
    # 1. Model Compilation Block
    model = build_dynamic_cnn(
        filters=filters, kernels=kernels, dilations=dilations,
        temporal_mode=temporal_mode, use_separable=Config.USE_SEPARABLE_CONV,
        stochastic_depth_rate=Config.STOCHASTIC_DEPTH_RATE
    )
    
    lr_schedule = CosineDecay(
        initial_learning_rate=Config.LEARNING_RATE,
        decay_steps=Config.COSINE_DECAY_STEPS,
        alpha=Config.COSINE_ALPHA
    )
    optimizer = Adam(learning_rate=lr_schedule)
    
    # Menggunakan Binary Crossentropy untuk Multi-Label
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
    
    # Trace/Warmup tensor
    model(tf.random.normal((1, Config.INPUT_SHAPE[0], Config.INPUT_SHAPE[1])))
    total_params = model.count_params()
    
    # 2. Pipeline Dataset Generation
    train_ds = create_tf_dataset(
        PTB_DATA["X_train"], PTB_DATA["y_train"], batch_size=Config.BATCH_SIZE,
        is_training=True, use_augmentation=Config.USE_AUGMENTATION,
        use_mixup=Config.USE_MIXUP, mixup_alpha=Config.MIXUP_ALPHA
    )
    val_ds = create_tf_dataset(PTB_DATA["X_val"], PTB_DATA["y_val"], batch_size=Config.BATCH_SIZE, is_training=False)
    
    model_path = os.path.join(exp_dir, "best_model.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=model_path, monitor="val_loss", save_best_only=True, verbose=1),
        CSVLogger(os.path.join(exp_dir, "training_log.csv"), append=False)
    ]
    
    # 3. Model Fitting
    model.fit(train_ds, validation_data=val_ds, epochs=Config.EPOCHS, callbacks=callbacks, verbose=1)
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    # 4. Post-Training Threshold Tuning
    optimized_thresholds = perform_threshold_tuning(model, PTB_DATA["X_val"], PTB_DATA["y_val"])
    
    # 5. Final Inference & Isolated Evaluation
    X_test_isolated = np.array(PTB_DATA["X_test"], dtype=np.float32)
    y_test_isolated = np.array(PTB_DATA["y_test"], dtype=np.float32)
    
    y_test_pred_prob = model.predict(X_test_isolated, batch_size=Config.BATCH_SIZE, verbose=0)
    final_metrics = evaluate_multi_label_metrics(y_test_isolated, y_test_pred_prob, optimized_thresholds)
    
    # -----------------------------------------------------------------
    # INTEGRASI: VISUALISASI MISCLASSIFIED UNTUK MULTI-LABEL BRANCH
    # -----------------------------------------------------------------
    # Mengonversi probabilitas kontinu menjadi biner berdasarkan threshold optimal masing-masing kelas
    y_pred_bin_matrix = np.zeros_like(y_test_pred_prob, dtype=np.int32)
    for idx in range(ml_cfg.NUM_CLASSES):
        y_pred_bin_matrix[:, idx] = (y_test_pred_prob[:, idx] >= optimized_thresholds[idx]).astype(np.int32)
        
    # Indeks galat ditentukan jika representasi biner tidak cocok dengan label asli
    # Karena ini multi-label, kita ratakan (flatten) perbandingan barisnya ke argmax tiruan untuk repositori plot
    y_true_simulated_labels = np.argmax(y_test_isolated, axis=1)
    y_pred_simulated_labels = np.argmax(y_pred_bin_matrix, axis=1)

    save_misclassified_samples(
        X=X_test_isolated,
        y_true=y_true_simulated_labels,
        y_pred=y_pred_simulated_labels,
        class_names=ml_cfg.TARGET_CLASSES,
        save_dir=exp_dir,
        max_samples=20
    )
    # -----------------------------------------------------------------
    
    # Append Metadata Tracker
    final_metrics["Experiment"] = experiment_name
    final_metrics["Total_Params"] = total_params
    final_metrics["Model_Size_MB"] = model_size_mb
    final_metrics["Thresholds_Assigned"] = str(optimized_thresholds)
    
    # Save Metrics File
    pd.DataFrame([final_metrics]).to_csv(os.path.join(exp_dir, "metrics_multilabel.csv"), index=False)
    
    # Clean Memory Streams
    del model; K.clear_session(); gc.collect()
    print(f"[Finished] Results successfully dumped to: {exp_dir}")

# =====================================================================
# MAIN LOOP GRID SEARCH
# =====================================================================
if __name__ == "__main__":
    import gc
    # Muat Dataset Berbasis Manifes Jalur Absolut Clean 250Hz
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