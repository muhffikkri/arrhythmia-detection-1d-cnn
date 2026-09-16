# =====================================================================
# FILE: experiment_metadata.py
# GPU-REPRODUCIBLE EXPERIMENT RECORDS
# =====================================================================
# Standardizes every experiment record (research-progress schema):
# experiment_id, dataset_train, dataset_test, dataset_version,
# preprocessing_version, lead_configuration, label_configuration,
# model_version, random_seed, training_configuration, checkpoint,
# metrics, confusion_matrix, prediction_distribution,
# confidence_distribution, notes, interpretation, status.
# Also snapshots the runtime environment (OS / Python / TF / GPU / CUDA)
# so Windows and Linux (GPU) runs can be compared reproducibly.
# =====================================================================

import os
import sys
import platform
import json

import numpy as np
import pandas as pd

import tensorflow as tf

from src.config import config as cfg
from src.preprocessing import preprocessing as dsp

# =====================================================================
# ENVIRONMENT SNAPSHOT
# =====================================================================

def get_environment_metadata():
    """OS / Python / TensorFlow / GPU / CUDA snapshot for one training run."""
    env = {
        "os": platform.system(),
        "os_release": platform.release(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "gpu": [],
        "cuda": None,
        "cudnn": None,
        "cpu_count": os.cpu_count(),
    }

    try:
        build_info = tf.sysconfig.get_build_info()
        env["cuda"] = build_info.get("cuda_version")
        env["cudnn"] = build_info.get("cudnn_version")
    except Exception:
        env["cuda"] = None

    for dev in tf.config.list_physical_devices("GPU"):
        info = {
            "name": dev.name,
            "device_type": dev.device_type,
        }
        try:
            details = tf.config.experimental.get_device_details(dev)
            info["device_name"] = details.get("device_name")
            info["compute_capability"] = details.get("compute_capability")
        except Exception:
            pass
        env["gpu"].append(info)

    return env


# =====================================================================
# EXPERIMENT ID
# =====================================================================

def infer_experiment_id(train_dataset, test_dataset, folder_key=None):
    """Research-phase experiment ID from the run configuration.

    Spec (docs/research-progress.md):
      RAW_PTBXL_3L / RAW_CHAPMAN_3L / CLEAN_PTBXL_3L / CLEAN_CHAPMAN_3L
      CROSS_PTBXL_TO_CHAPMAN / CROSS_CHAPMAN_TO_PTBXL
      FOCUSED_*_3L / DERIVED_*_3L / RULE_*_3L (via Config.EXPERIMENT_ID)
    """
    train_dataset = str(train_dataset).upper()
    test_dataset = str(test_dataset).upper()

    if train_dataset != test_dataset:
        return f"CROSS_{train_dataset}_TO_{test_dataset}"

    kind = "RAW" if (folder_key and "raw" in str(folder_key).lower()) else "CLEAN"
    return f"{kind}_{train_dataset}_3L"


def build_experiment_id(train_dataset, test_dataset, folder_key, override=None):
    if override:
        return str(override).upper()
    return infer_experiment_id(train_dataset, test_dataset, folder_key)


# =====================================================================
# DISTRIBUTION ARTIFACTS
# =====================================================================

def save_prediction_distributions(exp_dir, y_true_idx, y_pred_idx, y_prob,
                                  class_names):
    """Persist per-class prediction counts and confidence distribution."""
    names = list(class_names)

    pred_counts = pd.Series(y_pred_idx).value_counts().sort_index()
    true_counts = pd.Series(y_true_idx).value_counts().sort_index()

    pred_df = pd.DataFrame({
        "class": names,
        "y_true_count": [true_counts.get(i, 0) for i in range(len(names))],
        "y_pred_count": [pred_counts.get(i, 0) for i in range(len(names))],
        "support": int(len(y_true_idx)),
    })

    pred_path = os.path.join(exp_dir, "prediction_distribution.csv")
    pred_df.to_csv(pred_path, index=False)

    y_prob = np.asarray(y_prob, dtype=np.float32)
    conf_rows = []
    for i, name in enumerate(names):
        col = y_prob[:, i]
        if col.size == 0:
            conf_rows.append({"class": name, "n": 0})
            continue
        conf_rows.append({
            "class": name,
            "n": int(col.size),
            "mean": float(col.mean()),
            "median": float(np.median(col)),
            "std": float(col.std()),
            "p05": float(np.percentile(col, 5)),
            "p95": float(np.percentile(col, 95)),
            "max": float(col.max()),
        })

    conf_path = os.path.join(exp_dir, "confidence_distribution.csv")
    pd.DataFrame(conf_rows).to_csv(conf_path, index=False)

    return pred_path, conf_path


# =====================================================================
# EXPERIMENT RECORD + REGISTRY
# =====================================================================

def build_experiment_record(
    experiment_id,
    dataset_train,
    dataset_test,
    folder_key,
    sampling_rate,
    signal_len,
    label_scheme,
    class_names,
    model_version,
    arch_config,
    random_seed,
    training_configuration,
    checkpoint,
    metrics_path,
    metrics_summary,
    confusion_matrix_path,
    prediction_distribution_path,
    confidence_distribution_path,
    notes="",
    interpretation="",
    status="completed",
):
    record = {
        "experiment_id": experiment_id,
        "dataset_train": dataset_train,
        "dataset_test": dataset_test,
        "dataset_version": {
            "folder": folder_key,
            "sampling_rate_hz": sampling_rate,
            "signal_len": signal_len,
        },
        "preprocessing_version": {
            "pipeline": dsp.PIPELINE_VERSION,
            "cleaning": dsp.cleaning_pipeline_description(),
        },
        "lead_configuration": {
            "leads": list(cfg.LEAD_NAMES),
            "lead_indices": list(cfg.LEAD_INDICES),
        },
        "label_configuration": {
            "scheme": label_scheme,
            "num_classes": len(class_names),
            "class_names": list(class_names),
        },
        "model_version": model_version,
        "architecture": arch_config,
        "random_seed": random_seed,
        "training_configuration": training_configuration,
        "checkpoint": checkpoint,
        "metrics": {
            "file": metrics_path,
            "summary": metrics_summary,
        },
        "confusion_matrix": confusion_matrix_path,
        "prediction_distribution": prediction_distribution_path,
        "confidence_distribution": confidence_distribution_path,
        "notes": notes,
        "interpretation": interpretation,
        "status": status,
        "environment": get_environment_metadata(),
    }
    return record


def write_experiment_metadata(exp_dir, record):
    path = os.path.join(exp_dir, "experiment_metadata.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)
    return path


def flatten_record(record):
    """Flatten a nested record into one CSV row (registry-friendly)."""
    row = {}
    row["experiment_id"] = record["experiment_id"]
    row["dataset_train"] = record["dataset_train"]
    row["dataset_test"] = record["dataset_test"]
    row["dataset_version"] = record["dataset_version"]["folder"] + "@" + str(
        record["dataset_version"]["sampling_rate_hz"]) + "hz"
    row["preprocessing_version"] = record["preprocessing_version"]["pipeline"] + "-" + \
        record["preprocessing_version"]["cleaning"]
    row["lead_configuration"] = ",".join(record["lead_configuration"]["leads"])
    row["label_configuration"] = record["label_configuration"]["scheme"] + ":" + \
        str(record["label_configuration"]["num_classes"]) + "classes"
    row["model_version"] = record["model_version"]
    row["random_seed"] = record["random_seed"]
    row["training_configuration"] = json.dumps(
        record["training_configuration"], sort_keys=True
    )
    row["checkpoint"] = record["checkpoint"]
    row["metrics"] = record["metrics"]["file"]
    row["confusion_matrix"] = record["confusion_matrix"]
    row["prediction_distribution"] = record["prediction_distribution"]
    row["confidence_distribution"] = record["confidence_distribution"]
    row["notes"] = record["notes"]
    row["interpretation"] = record["interpretation"]
    row["status"] = record["status"]
    row["os"] = record["environment"]["os"]
    row["python"] = record["environment"]["python"]
    row["tensorflow"] = record["environment"]["tensorflow"]
    row["gpu"] = "; ".join(g.get("device_name") or g.get("name")
                           for g in record["environment"]["gpu"]) or "cpu"
    row["cuda"] = record["environment"].get("cuda")
    return row


def append_experiment_registry(registry_path, record):
    """Append one record to the master experiment registry CSV."""
    row = flatten_record(record)
    headers = list(row.keys())

    write_header = not os.path.exists(registry_path) or os.path.getsize(registry_path) == 0

    import csv as _csv
    with open(registry_path, "a", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=headers)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return registry_path