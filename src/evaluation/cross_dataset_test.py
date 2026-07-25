# =====================================================================
# FILE: src/evaluation/cross_dataset_test.py
# PURPOSE: Cross-dataset validation of all models on Chapman Dataset
# =====================================================================

import os
import gc
import sys
import re
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    multilabel_confusion_matrix
)

# Add project root directory to python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Local imports
from src.config import config as cfg
from src.config import config_labels as ml_cfg
from src.models.model_factory import StochasticDepth

warnings.filterwarnings("ignore")

# =====================================================================
# PATHS
# =====================================================================
MODELS_DIR = os.path.join(project_root, "models")
OUTPUT_DIR = os.path.join(project_root, "output", "cross_dataset_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# LOAD CHAPMAN DATASET
# =====================================================================
def load_chapman_dataset():
    manifest_path = os.path.join(cfg.RESAMPLE_BASE, "manifest_chapman.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Chapman manifest not found at {manifest_path}. "
            "Please run 'src/preprocessing/proccess_chapman.py' first."
        )

    df = pd.read_csv(manifest_path)
    
    # Filter to only contain the 4 classes of interest
    df = df[df["target_class"].isin(cfg.CLASS_NAMES)].reset_index(drop=True)
    
    X = []
    y = []
    
    base_folder = cfg.SUB_FOLDERS["CHAPMAN_CLEAN_250HZ"]
    print(f"--> Loading {len(df)} Chapman signals from {base_folder}...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading Chapman"):
        file_path = os.path.join(base_folder, row["filename_npy"])
        if not os.path.exists(file_path):
            continue
        try:
            signal = np.load(file_path).astype(np.float32)
            X.append(signal)
            y.append(row["target_class"])
        except Exception as e:
            continue
            
    X = np.array(X, dtype=np.float32)
    print(f"✓ Loaded Chapman dataset successfully. Shape: {X.shape}, labels count: {len(y)}")
    return X, y

# =====================================================================
# THRESHOLDS LOADER FOR SIGMOID MODELS
# =====================================================================
def load_sigmoid_thresholds(model_dir):
    thresholds = [0.5, 0.5, 0.5, 0.5]
    metrics_path = os.path.join(model_dir, "metrics.csv")
    if os.path.exists(metrics_path):
        try:
            df_m = pd.read_csv(metrics_path)
            if "Thresholds_Assigned" in df_m.columns:
                th_val = df_m["Thresholds_Assigned"].iloc[0]
                floats = [float(x) for x in re.findall(r"0\.\d+", str(th_val))]
                if len(floats) == 4:
                    thresholds = floats
                    print(f"    -> Loaded assigned thresholds from metrics.csv: {thresholds}")
        except Exception as e:
            print(f"    -> Warning: Failed to parse thresholds from metrics.csv: {e}. Using default [0.5, 0.5, 0.5, 0.5].")
    else:
        print("    -> No metrics.csv found. Using default thresholds [0.5, 0.5, 0.5, 0.5].")
    return thresholds

# =====================================================================
# DISCOVER MODELS
# =====================================================================
def discover_models(models_dir):
    dir_to_model = {}
    for root, _, files in os.walk(models_dir):
        for file in files:
            if file == "best_model.keras":
                if root not in dir_to_model:
                    dir_to_model[root] = os.path.join(root, file)
            elif file == "best_model_patched.keras":
                # Prioritize patched models if they exist
                dir_to_model[root] = os.path.join(root, file)
    return sorted(list(dir_to_model.values()))

# =====================================================================
# MAIN RUNNER
# =====================================================================
def main():
    print("=" * 80)
    print("STARTING CROSS DATASET TEST ON CHAPMAN DATASET")
    print("=" * 80)

    # 1. Load Chapman Dataset
    try:
        X_chapman, y_chapman = load_chapman_dataset()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # 2. Discover Models
    model_paths = discover_models(MODELS_DIR)
    print(f"--> Discovered {len(model_paths)} models for evaluation:")
    for mp in model_paths:
        print(f"  - {os.path.relpath(mp, MODELS_DIR)}")

    if not model_paths:
        print("No models found in the models directory.")
        sys.exit(0)

    summary_records = []

    # 3. Evaluate each model
    for idx, model_path in enumerate(model_paths):
        rel_path = os.path.relpath(os.path.dirname(model_path), MODELS_DIR)
        model_name = rel_path.replace(os.sep, "__").replace(" ", "_")
        model_output_dir = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        print("\n" + "-" * 80)
        print(f"Evaluating Model [{idx + 1}/{len(model_paths)}]: {rel_path}")
        print("-" * 80)

        # Load Keras Model
        try:
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={'StochasticDepth': StochasticDepth},
                compile=False
            )
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            continue

        # Determine Model Head Type (Softmax vs Sigmoid)
        last_layer = model.layers[-1]
        activation_name = None
        if hasattr(last_layer, 'activation') and last_layer.activation is not None:
            activation_name = last_layer.activation.__name__

        is_sigmoid = False
        if activation_name == 'sigmoid':
            is_sigmoid = True
        elif activation_name == 'softmax':
            is_sigmoid = False
        else:
            # Fallback to checking the folder path
            if "Sigmoid" in model_path or "multilabel" in model_path:
                is_sigmoid = True

        print(f"Detected Model Type: {'Sigmoid (Multilabel)' if is_sigmoid else 'Softmax (Multiclass)'} (activation={activation_name})")

        # Run Predictions
        print("Running inference...")
        try:
            y_pred_prob = model.predict(X_chapman, batch_size=64, verbose=1)
        except Exception as e:
            print(f"Error running inference for {model_name}: {e}")
            # Clean up
            del model
            tf.keras.backend.clear_session()
            gc.collect()
            continue

        # Evaluate and save based on model type
        if not is_sigmoid:
            # ==========================================
            # SOFTMAX MULTICLASS
            # ==========================================
            lb = LabelBinarizer()
            lb.fit(cfg.CLASS_NAMES)
            y_true_onehot = lb.transform(y_chapman)
            y_true_labels = np.argmax(y_true_onehot, axis=1)
            y_pred_labels = np.argmax(y_pred_prob, axis=1)

            # Metrics
            acc = accuracy_score(y_true_labels, y_pred_labels)
            rec = recall_score(y_true_labels, y_pred_labels, average='macro', zero_division=0)
            f1 = f1_score(y_true_labels, y_pred_labels, average='macro', zero_division=0)

            print(f"Results - Accuracy: {acc:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f}")

            # Classification Report
            rep_str = classification_report(
                y_true_labels,
                y_pred_labels,
                target_names=cfg.CLASS_NAMES,
                digits=4,
                zero_division=0
            )

            # Confusion Matrix Plot
            cm = confusion_matrix(y_true_labels, y_pred_labels)
            fig, ax = plt.subplots(figsize=(8, 8))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cfg.CLASS_NAMES)
            disp.plot(ax=ax, xticks_rotation=45, colorbar=False, cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix (Softmax)\n{rel_path}", fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(model_output_dir, "confusion_matrix.png"), dpi=150)
            plt.close()

            summary_records.append({
                "Model_Dir": rel_path,
                "Model_Name": model_name,
                "Model_Type": "Softmax",
                "Accuracy": acc,
                "Recall": rec,
                "F1_Score": f1,
                "Thresholds": "N/A"
            })

        else:
            # ==========================================
            # SIGMOID MULTILABEL
            # ==========================================
            # Binarize targets in Sigmoid target ordering
            y_true_onehot = np.zeros((len(y_chapman), len(ml_cfg.TARGET_CLASSES)), dtype=np.float32)
            for i, name in enumerate(y_chapman):
                class_idx = ml_cfg.TARGET_CLASSES.index(name)
                y_true_onehot[i, class_idx] = 1.0

            thresholds = load_sigmoid_thresholds(os.path.dirname(model_path))
            
            y_pred_bin = np.zeros_like(y_pred_prob)
            for i in range(4):
                y_pred_bin[:, i] = (y_pred_prob[:, i] >= thresholds[i]).astype(int)

            # Metrics
            # Subset accuracy (exact match)
            acc = accuracy_score(y_true_onehot, y_pred_bin)
            rec = recall_score(y_true_onehot, y_pred_bin, average='macro', zero_division=0)
            f1 = f1_score(y_true_onehot, y_pred_bin, average='macro', zero_division=0)

            # Mean binary accuracy (optional helper metric)
            mean_bin_acc = np.mean([accuracy_score(y_true_onehot[:, i], y_pred_bin[:, i]) for i in range(4)])
            print(f"Results - Subset Acc: {acc:.4f} (Mean Bin Acc: {mean_bin_acc:.4f}) | Recall: {rec:.4f} | F1-Score: {f1:.4f}")

            # Classification Report
            rep_str = classification_report(
                y_true_onehot,
                y_pred_bin,
                target_names=ml_cfg.TARGET_CLASSES,
                digits=4,
                zero_division=0
            )

            # Confusion Matrix Plot (2x2 binary grids)
            mcm = multilabel_confusion_matrix(y_true_onehot, y_pred_bin)
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            for i, class_name in enumerate(ml_cfg.TARGET_CLASSES):
                ax = axes[i // 2, i % 2]
                cm_bin = mcm[i]
                ax.matshow(cm_bin, cmap=plt.cm.Blues, alpha=0.3)
                for g_i in range(cm_bin.shape[0]):
                    for g_j in range(cm_bin.shape[1]):
                        ax.text(x=g_j, y=g_i, s=cm_bin[g_i, g_j], va='center', ha='center', fontsize=12, fontweight='bold')
                ax.set_title(f"CM Biner: {class_name}", fontweight='bold')
                ax.set_xticklabels(['', 'Neg', 'Pos'])
                ax.set_yticklabels(['', 'Neg', 'Pos'])
            plt.suptitle(f"Multilabel Confusion Matrix\n{rel_path}", fontweight='bold', y=0.98)
            plt.tight_layout()
            plt.savefig(os.path.join(model_output_dir, "confusion_matrix.png"), dpi=150)
            plt.close()

            summary_records.append({
                "Model_Dir": rel_path,
                "Model_Name": model_name,
                "Model_Type": f"Sigmoid (Subset Acc: {acc:.4f}, Mean Bin Acc: {mean_bin_acc:.4f})",
                "Accuracy": acc,
                "Recall": rec,
                "F1_Score": f1,
                "Thresholds": str(thresholds)
            })

        # Save Classification Report txt file
        rep_path = os.path.join(model_output_dir, "classification_report.txt")
        with open(rep_path, "w", encoding="utf-8") as f:
            f.write(rep_str)
        print(f"✓ Saved metrics and confusion matrix to {model_output_dir}")

        # Free memory
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    # 4. Save comparative summary CSV
    summary_df = pd.DataFrame(summary_records)
    summary_path = os.path.join(OUTPUT_DIR, "cross_dataset_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print("\n" + "=" * 80)
    print("ALL EVALUATIONS FINISHED")
    print("=" * 80)
    print(f"Summary table saved to: {summary_path}")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
