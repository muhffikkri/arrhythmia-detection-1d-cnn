# =====================================================================
# FILE: tracker.py
# =====================================================================

import os
import json
import datetime

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# =====================================================================
# CREATE EXPERIMENT DIRECTORY
# =====================================================================

def create_experiment_dir(
    base_dir,
    experiment_group,
    experiment_name
):
    timestamp = datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    exp_dir = os.path.join(
        base_dir,
        experiment_group,
        f"{timestamp}__{experiment_name}"
    )

    os.makedirs(exp_dir, exist_ok=True)

    os.makedirs(
        os.path.join(exp_dir, "weights"),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(exp_dir, "gradcam"),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(exp_dir, "misclassified"),
        exist_ok=True
    )

    return exp_dir


# =====================================================================
# SAVE CONFIG
# =====================================================================

def save_experiment_config(
    config_dict,
    save_dir
):
    save_path = os.path.join(
        save_dir,
        "experiment_config.json"
    )

    # Menggunakan utf-8 untuk penulisan JSON konfigurasi yang aman
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            config_dict,
            f,
            indent=4
        )


# =====================================================================
# SAVE TRAINING HISTORY
# =====================================================================

def save_training_history(
    history,
    exp_dir
):
    history_df = pd.DataFrame(history)

    csv_path = os.path.join(
        exp_dir,
        "training_history.csv"
    )

    history_df.to_csv(
        csv_path,
        index=False
    )

    # =========================================================
    # LOSS CURVE
    # =========================================================

    if (
        "loss" in history_df.columns
        and
        "val_loss" in history_df.columns
    ):
        plt.figure(figsize=(10, 5))

        plt.plot(
            history_df["loss"],
            label="train_loss"
        )

        plt.plot(
            history_df["val_loss"],
            label="val_loss"
        )

        plt.legend()
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)

        plt.savefig(
            os.path.join(
                exp_dir,
                "loss_curve.png"
            )
        )
        plt.close()

    # =========================================================
    # ACCURACY CURVE
    # =========================================================

    if (
        "accuracy" in history_df.columns
        and
        "val_accuracy" in history_df.columns
    ):
        plt.figure(figsize=(10, 5))

        plt.plot(
            history_df["accuracy"],
            label="train_accuracy"
        )

        plt.plot(
            history_df["val_accuracy"],
            label="val_accuracy"
        )

        plt.legend()
        plt.title("Accuracy Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)

        plt.savefig(
            os.path.join(
                exp_dir,
                "accuracy_curve.png"
            )
        )
        plt.close()


# =====================================================================
# SAVE CONFUSION MATRIX
# =====================================================================

def save_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    save_dir,
    normalize=False
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        normalize="true" if normalize else None
    )

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(
        ax=ax,
        xticks_rotation=45,
        colorbar=False
    )

    filename = (
        "confusion_matrix_normalized.png"
        if normalize
        else
        "confusion_matrix.png"
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, filename)
    )
    plt.close()


# =====================================================================
# SAVE CLASSIFICATION REPORT
# =====================================================================

def save_classification_report(
    y_true,
    y_pred,
    class_names,
    save_dir
):
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )

    save_path = os.path.join(
        save_dir,
        "classification_report.txt"
    )

    # Kunci Utama 1: Proteksi berkas laporan teks dari masalah encoding Windows
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)


# =====================================================================
# SAVE MODEL SUMMARY
# =====================================================================

def save_model_summary(
    model,
    save_dir
):
    save_path = os.path.join(
        save_dir,
        "model_summary.txt"
    )

    # Kunci Utama 2: Menggunakan encoding="utf-8" mutlak untuk menangani box-drawing characters Keras
    with open(save_path, "w", encoding="utf-8") as f:
        model.summary(
            print_fn=lambda x: f.write(str(x) + "\n")
        )


# =====================================================================
# UPDATE MASTER TRACKER
# =====================================================================

def update_master_tracker(
    metrics_dict,
    master_csv
):
    df_new = pd.DataFrame(
        [metrics_dict]
    )

    if os.path.exists(master_csv):
        df_old = pd.read_csv(master_csv)
        df = pd.concat(
            [df_old, df_new],
            ignore_index=True
        )
    else:
        df = df_new

    df.to_csv(
        master_csv,
        index=False
    )