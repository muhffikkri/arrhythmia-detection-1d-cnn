# =====================================================================
# FILE: src/evaluation/cross_dataset_test.py
# PURPOSE: Registry-driven, bidirectional cross-dataset evaluation.
# =====================================================================
# Evaluates every ACTIVE model in src/config/model_registry.py against the
# mapped test split of a target dataset.
#
# Supported directions (CLI):
#   PTBXL_to_CHAPMAN  - models evaluated on Chapman test split (zero-shot)
#   CHAPMAN_to_PTBXL  - models evaluated on PTB-XL test split (zero-shot)
#   --per-dataset     - additionally PTBXL_to_PTBXL and CHAPMAN_to_CHAPMAN
#
# Class ordering is resolved per model (data-driven):
#   1. experiment_config.json["class_names"]   (new unified runner)
#   2. class_names.json                         (annotated legacy artifacts)
#   3. cfg.CLASS_NAMES fallback with a warning

import os
import gc
import sys
import re
import json
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

from src.config import config as cfg
from src.config.model_registry import MODEL_REGISTRY, EVALUATE_ONLY_IDS
from src.models.model_factory import StochasticDepth
from src.training.dataset_loader import DatasetLoader

warnings.filterwarnings("ignore")

# =====================================================================
# PATHS
# =====================================================================
MODELS_DIR = os.path.join(project_root, "models")
OUTPUT_DIR = os.path.join(project_root, "output", "cross_dataset_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# REGISTRY HELPERS
# =====================================================================

def active_registry_models():
    entries = [e for e in MODEL_REGISTRY if e.get("active", True)]
    if EVALUATE_ONLY_IDS:
        entries = [e for e in entries if e["id"] in EVALUATE_ONLY_IDS]
    return entries


def resolve_model_file(model_dir):
    for name in ["best_model_patched.keras", "best_model.keras"]:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return path
    return None


def load_class_names(model_dir):
    """Data-driven class order for a model (see docstring precedence list)."""
    config_path = os.path.join(model_dir, "experiment_config.json")
    meta_path = os.path.join(model_dir, "class_names.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, encoding="utf-8") as f:
                names = json.load(f).get("class_names")
            if names:
                return list(names)
        except (json.JSONDecodeError, OSError):
            pass
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                names = json.load(f).get("class_names")
            if names:
                return list(names)
        except (json.JSONDecodeError, OSError):
            pass
    print(f"    ! No class_names metadata for {os.path.basename(model_dir)}; "
          f"falling back to cfg.CLASS_NAMES.")
    return list(cfg.CLASS_NAMES)


def load_sigmoid_thresholds(model_dir):
    thresholds = [0.5] * 4
    metrics_path = os.path.join(model_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        print("    -> No metrics.csv found. Using default thresholds [0.5]*4.")
        return thresholds
    try:
        df_m = pd.read_csv(metrics_path)
        if "Thresholds_Assigned" in df_m.columns:
            th_val = df_m["Thresholds_Assigned"].iloc[0]
            floats = [float(x) for x in re.findall(r"0\.\d+", str(th_val))]
            if len(floats) == len(thresholds):
                thresholds = floats
                print(f"    -> Loaded thresholds from metrics.csv: {thresholds}")
    except Exception as e:
        print(f"    -> Warning: failed to parse thresholds: {e}. Using [0.5]*4.")
    return thresholds

# =====================================================================
# TARGET DATA
# =====================================================================

def load_target_eval(dataset):
    print(f"\n[Data] Loading {dataset} mapped test split...")
    loader = DatasetLoader(dataset=dataset, label_scheme="mapped")
    X, y_onehot = loader.build_eval_split("test")
    names = [loader.class_names[idx] for idx in np.argmax(y_onehot, axis=1)]
    return X, names, loader.class_names

# =====================================================================
# EVALUATE A SINGLE MODEL ON A TARGET
# =====================================================================

def evaluate_model(model_path, model_type, X, y_true_names, class_names, thresholds):
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"StochasticDepth": StochasticDepth},
        compile=False
    )
    y_prob = model.predict(X, batch_size=64, verbose=0)
    del model
    gc.collect()
    tf.keras.backend.clear_session()

    # Encode reference labels in the *model's* class order.
    lb = LabelBinarizer()
    lb.fit(class_names)
    y_onehot = lb.transform(y_true_names).astype(np.float32)
    y_true = np.argmax(y_onehot, axis=1)

    if model_type == "sigmoid":
        y_bin = np.zeros_like(y_prob)
        for i in range(len(class_names)):
            y_bin[:, i] = (y_prob[:, i] >= thresholds[i]).astype(int)
        acc = accuracy_score(y_onehot, y_bin)
        rec = recall_score(y_onehot, y_bin, average="macro", zero_division=0)
        f1 = f1_score(y_onehot, y_bin, average="macro", zero_division=0)
        rep = classification_report(y_onehot, y_bin, target_names=class_names, zero_division=0)
        cm_fig = _multilabel_cm_figure(y_onehot, y_bin, class_names, os.path.basename(model_path))
        return {
            "Accuracy": acc, "Macro_Recall": rec, "Macro_F1": f1,
            "Thresholds": str(thresholds), "report": rep, "cm_fig": cm_fig
        }

    y_pred = np.argmax(y_prob, axis=1)
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    rep = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False, cmap=plt.cm.Blues)
    plt.tight_layout()
    return {
        "Accuracy": acc, "Macro_Recall": rec, "Macro_F1": f1,
        "Thresholds": "N/A", "report": rep, "cm_fig": fig
    }


def _multilabel_cm_figure(y_true, y_bin, class_names, title):
    mcm = multilabel_confusion_matrix(y_true, y_bin)
    n = len(class_names)
    rows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(10, 5 * rows))
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
    fig.suptitle(f"Multilabel Confusion Matrix - {os.path.basename(title)}", fontweight="bold", y=0.98)
    plt.tight_layout()
    return fig

# =====================================================================
# DIRECTION RUNNER
# =====================================================================

def run_direction(direction, per_dataset=False):
    DIRECTIONS = {
        "PTBXL_to_CHAPMAN": ("PTBXL", "CHAPMAN"),
        "CHAPMAN_to_PTBXL": ("CHAPMAN", "PTBXL"),
        "PTBXL_to_PTBXL": ("PTBXL", "PTBXL"),
        "CHAPMAN_to_CHAPMAN": ("CHAPMAN", "CHAPMAN"),
    }
    if direction not in DIRECTIONS:
        raise ValueError(f"Unknown direction '{direction}'.")

    _, target = DIRECTIONS[direction]
    print("\n" + "=" * 80)
    print(f"DIRECTION: {direction}")
    print("=" * 80)

    X, y, class_names = load_target_eval(target)
    entries = active_registry_models()
    if not entries:
        print("[WARN] No models match the registry filter. Nothing to evaluate.")
        return

    out_dir = os.path.join(OUTPUT_DIR, direction)
    os.makedirs(out_dir, exist_ok=True)
    summary_records = []

    for idx, entry in enumerate(entries):
        model_dir = os.path.join(MODELS_DIR, entry["path"])
        model_file = resolve_model_file(model_dir)
        if model_file is None:
            print(f"  [Skip] No checkpoint found in {model_dir}")
            continue

        print(f"\n[{idx + 1}/{len(entries)}] {entry['id']} ({entry['type']})")
        model_class_names = load_class_names(model_dir)
        print(f"    Class order: {model_class_names}")

        if len(model_class_names) != len(class_names):
            print(f"    [Skip] class-count mismatch ({len(model_class_names)} vs {len(class_names)}).")
            continue

        thresholds = load_sigmoid_thresholds(model_dir) if entry["type"] == "sigmoid" else None
        try:
            result = evaluate_model(
                model_file,
                entry["type"],
                X,
                y,
                model_class_names,
                thresholds
            )
        except Exception as e:
            print(f"    [Error] {e}")
            continue

        model_out = os.path.join(out_dir, entry["id"].replace(os.sep, "__"))
        os.makedirs(model_out, exist_ok=True)

        with open(os.path.join(model_out, "classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(result["report"])
        result["cm_fig"].savefig(os.path.join(model_out, "confusion_matrix.png"), dpi=150)
        plt.close(result["cm_fig"])

        summary_records.append({
            "Direction": direction,
            "Model_Id": entry["id"],
            "Model_Name": entry["name"],
            "Model_Type": entry["type"],
            "Accuracy": result["Accuracy"],
            "Macro_Recall": result["Macro_Recall"],
            "Macro_F1": result["Macro_F1"],
            "Thresholds": result["Thresholds"]
        })
        print(f"    ACC={result['Accuracy']:.4f} REC={result['Macro_Recall']:.4f} "
              f"F1={result['Macro_F1']:.4f}")

    summary_df = pd.DataFrame(summary_records)
    summary_path = os.path.join(out_dir, f"cross_dataset_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print("\n" + "=" * 80)
    print(f"DIRECTION DONE: {direction}")
    print(f"Summary saved: {summary_path}")
    try:
        print(summary_df.to_string(index=False))
    except Exception:
        pass

# =====================================================================
# MAIN
# =====================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Bidirectional cross-dataset evaluation")
    parser.add_argument(
        "--direction",
        choices=["PTBXL_to_CHAPMAN", "CHAPMAN_to_PTBXL", "PTBXL_to_PTBXL", "CHAPMAN_to_CHAPMAN"],
        nargs="*",
        default=["PTBXL_to_CHAPMAN", "CHAPMAN_to_PTBXL"]
    )
    parser.add_argument(
        "--per-dataset",
        action="store_true",
        help="Also run PTBXL_to_PTBXL and CHAPMAN_to_CHAPMAN."
    )
    args = parser.parse_args()

    directions = list(args.direction)
    if args.per_dataset:
        for d in ["PTBXL_to_PTBXL", "CHAPMAN_to_CHAPMAN"]:
            if d not in directions:
                directions.append(d)

    for direction in directions:
        run_direction(direction)


if __name__ == "__main__":
    main()