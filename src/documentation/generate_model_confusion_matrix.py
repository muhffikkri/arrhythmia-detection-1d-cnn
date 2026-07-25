# =====================================================================
# FILE: src/documentation/generate_model_confusion_matrix.py
# PURPOSE: REPORT DOCUMENTATION TOOL — HIGH-RES CONFUSION MATRIX & METRICS
# =====================================================================

import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report

# Add project root directory to python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import tensorflow as tf
from src.config import config as cfg
from src.config import config_labels as ml_cfg
from src.models.model_factory import StochasticDepth

def find_available_models():
    """Finds available .keras and .h5 model files in models/ and output/ directories."""
    default_dir = os.path.join(cfg.BASE_DIR, "models", "Pure CNN Softmax Head", "softmax_raw_500to250_cnn")
    if os.path.exists(default_dir):
        default_files = sorted(glob.glob(os.path.join(default_dir, "*.keras"))) + sorted(glob.glob(os.path.join(default_dir, "*.h5")))
        if default_files:
            return default_files

    search_dirs = [
        os.path.join(cfg.BASE_DIR, "models"),
        os.path.join(cfg.OUTPUT_DIR, "research_experiments"),
        os.path.join(cfg.OUTPUT_DIR, "multilabel_experiments"),
        cfg.OUTPUT_DIR
    ]
    
    found_models = []
    for d in search_dirs:
        if os.path.exists(d):
            found_models.extend(glob.glob(os.path.join(d, "**", "*.keras"), recursive=True))
            found_models.extend(glob.glob(os.path.join(d, "**", "*.h5"), recursive=True))
            
    return sorted(list(set(found_models)))

def load_split_dataset(folder_key="E2_clean_100_to_250", target_split="test", scheme="softmax"):
    """Loads specified split (train/val/test) for PTB-XL dataset."""
    manifest_path = os.path.join(cfg.RESAMPLE_BASE, "manifest_ptbxl.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest path not found: {manifest_path}")

    df_manifest = pd.read_csv(manifest_path)
    
    if scheme == "sigmoid":
        target_classes = ml_cfg.TARGET_CLASSES
    else:
        target_classes = cfg.CLASS_NAMES

    df_filtered = df_manifest[df_manifest["target_class"].isin(target_classes)].reset_index(drop=True)
    base_folder = cfg.SUB_FOLDERS.get(folder_key, cfg.SUB_FOLDERS["E2_clean_100_to_250"])

    X = []
    y = []

    split_fold_map = {
        "train": range(1, 9),
        "val": [9],
        "test": [10]
    }
    allowed_folds = split_fold_map.get(target_split.lower(), [10])

    print(f"\n[Dataset Loader] Loading {target_split.upper()} set from '{folder_key}'...")
    for _, row in df_filtered.iterrows():
        fold = int(row["strat_fold"])
        if fold not in allowed_folds:
            continue

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
    
    if scheme == "sigmoid":
        y_encoded = np.zeros((len(y), len(target_classes)), dtype=np.float32)
        for i, cls in enumerate(y):
            idx = target_classes.index(cls)
            y_encoded[i, idx] = 1.0
    else:
        lb = LabelBinarizer()
        lb.fit(target_classes)
        y_encoded = lb.transform(y).astype(np.float32)
        target_classes = list(lb.classes_)

    return X, y, y_encoded, target_classes

def evaluate_and_plot_confusion_matrix(model_path, folder_key="E2_clean_100_to_250", target_split="test", scheme=None):
    print("=" * 80)
    print("EVALUATING MODEL & GENERATING CONFUSION MATRIX REPORT")
    print("=" * 80)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Output directory
    out_dir = os.path.join(cfg.OUTPUT_DIR, "documentation_reports")
    os.makedirs(out_dir, exist_ok=True)

    model_basename = os.path.splitext(os.path.basename(model_path))[0]
    parent_dir = os.path.basename(os.path.dirname(model_path))
    model_identifier = f"{parent_dir}__{model_basename}".replace(" ", "_")

    print(f"[OK] Model File   : {model_path}")
    print(f"[OK] Output Label : {model_identifier}")

    # Load Model
    print("\n--> Loading Keras Model...")
    custom_objects = {"StochasticDepth": StochasticDepth}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

    # Auto-detect folder_key if model_path specifies resolution
    if folder_key == "E2_clean_100_to_250" and ("500to250" in model_path or "500_to_250" in model_path):
        folder_key = "E3_clean_500_to_250"

    # Detect Scheme based on model output shape or argument
    if scheme is None:
        num_output_units = model.output_shape[-1]
        if num_output_units == ml_cfg.NUM_CLASSES:
            scheme = "softmax"
        else:
            scheme = "softmax"

    # Load Evaluation Data
    X_eval, y_true_labels, y_true_binary, class_names = load_split_dataset(
        folder_key=folder_key,
        target_split=target_split,
        scheme=scheme
    )

    print(f"[OK] Evaluated Samples (N): {len(X_eval)}")

    # Model Inference
    y_pred_prob = model.predict(X_eval, batch_size=64, verbose=1)

    if scheme == "sigmoid":
        # Multi-label binary decisions
        y_pred_bin = (y_pred_prob >= 0.5).astype(int)
        y_true_indices = np.argmax(y_true_binary, axis=1)
        y_pred_indices = np.argmax(y_pred_prob, axis=1)
    else:
        # Multiclass Softmax decisions
        y_true_indices = np.argmax(y_true_binary, axis=1)
        y_pred_indices = np.argmax(y_pred_prob, axis=1)

    # Compute Confusion Matrix
    cm_counts = confusion_matrix(y_true_indices, y_pred_indices, labels=list(range(len(class_names))))
    
    # Compute Normalized Row Matrix (Recall)
    cm_norm = cm_counts.astype('float') / cm_counts.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    # Export Classification Report
    clr_str = classification_report(y_true_indices, y_pred_indices, target_names=class_names, digits=4)
    clr_dict = classification_report(y_true_indices, y_pred_indices, target_names=class_names, output_dict=True)
    
    print("\n--- Classification Report ---")
    print(clr_str)

    txt_report_path = os.path.join(out_dir, f"classification_report_{model_identifier}.txt")
    csv_report_path = os.path.join(out_dir, f"classification_report_{model_identifier}.csv")

    with open(txt_report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Dataset Key: {folder_key} | Split: {target_split.upper()}\n\n")
        f.write(clr_str)

    pd.DataFrame(clr_dict).transpose().to_csv(csv_report_path)
    print(f"[SAVED] Report TXT: {txt_report_path}")
    print(f"[SAVED] Report CSV: {csv_report_path}")

    # =====================================================================
    # PUBLICATION-GRADE DUAL-PANEL CONFUSION MATRIX (300 DPI)
    # =====================================================================
    sns.set_theme(style="white", font_scale=1.1)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), dpi=300)

    # Panel 1: Raw Sample Counts
    sns.heatmap(
        cm_counts,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,
        ax=axes[0],
        annot_kws={"size": 12, "weight": "bold"}
    )
    axes[0].set_title("A. Confusion Matrix (Raw Counts)", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xlabel("Predicted Label", fontsize=11, fontweight='bold', labelpad=8)
    axes[0].set_ylabel("True Label", fontsize=11, fontweight='bold', labelpad=8)

    # Panel 2: Normalized Recall Percentages
    sns.heatmap(
        cm_norm * 100,
        annot=True,
        fmt='.1f',
        cmap='Greens',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,
        ax=axes[1],
        annot_kws={"size": 12, "weight": "bold"}
    )
    # Add % suffix visually to panel 2 annotations
    for text in axes[1].texts:
        text.set_text(f"{text.get_text()}%")

    axes[1].set_title("B. Normalized Confusion Matrix (Recall %)", fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xlabel("Predicted Label", fontsize=11, fontweight='bold', labelpad=8)
    axes[1].set_ylabel("True Label", fontsize=11, fontweight='bold', labelpad=8)

    plt.suptitle(f"Model Performance Evaluation: {model_basename}", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    fig_path = os.path.join(out_dir, f"confusion_matrix_{model_identifier}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[SAVED] Dual Confusion Matrix Figure: {fig_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate publication-grade confusion matrix for selected model.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model (.keras or .h5)")
    parser.add_argument("--folder_key", type=str, default="E2_clean_100_to_250", help="Dataset subfolder key")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--scheme", type=str, default="softmax", choices=["softmax", "sigmoid"], help="Classification scheme")

    args = parser.parse_args()

    model_file = args.model_path
    if model_file is None or not os.path.exists(model_file):
        available = find_available_models()
        if available:
            model_file = available[0]
            print(f"[Auto Selection] Model not explicitly specified, using found checkpoint:\n -> {model_file}")
        else:
            raise FileNotFoundError("No trained model files found. Please specify --model_path.")

    evaluate_and_plot_confusion_matrix(
        model_path=model_file,
        folder_key=args.folder_key,
        target_split=args.split,
        scheme=args.scheme
    )
