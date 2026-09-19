# =====================================================================
# FILE: evaluator.py
# FINAL VERSION — ECG RESEARCH EVALUATOR
# =====================================================================

import time
import numpy as np
import pandas as pd

import tensorflow.keras.backend as K

from tensorflow.keras import layers

from sklearn.metrics import (

    classification_report,

    confusion_matrix,

    balanced_accuracy_score,

    precision_score,

    recall_score,

    f1_score
)

from src.config import config as cfg

# =====================================================================
# HARDWARE EVALUATION
# =====================================================================

def evaluate_hardware_efficiency(
    model,
    X_sample,
    exp_name
):

    """
    Evaluate:
    - parameter count
    - model size
    - MACs estimation
    - inference latency
    """

    trainable_params = np.sum([

        K.count_params(w)

        for w in model.trainable_weights
    ])

    model_size_mb = (

        trainable_params * 4

    ) / (1024 * 1024)

    total_macs = 0

    # ================================================================
    # MACs ESTIMATION
    # ================================================================

    for layer in model.layers:

        # ------------------------------------------------------------
        # Conv1D
        # ------------------------------------------------------------

        if isinstance(layer, layers.Conv1D):

            try:

                out_seq_len = int(
                    layer.output.shape[1]
                )

                in_channels = int(
                    layer.input.shape[-1]
                )

                macs = (

                    layer.filters

                    * in_channels

                    * layer.kernel_size[0]

                    * out_seq_len
                )

                total_macs += int(macs)

            except:
                pass

        # ------------------------------------------------------------
        # Dense
        # ------------------------------------------------------------

        elif isinstance(layer, layers.Dense):

            try:

                macs = int(

                    layer.input.shape[-1]

                    * layer.units
                )

                total_macs += macs

            except:
                pass

    # ================================================================
    # INFERENCE LATENCY
    # ================================================================

    sample = X_sample[:1]

    # warmup
    _ = model(sample, training=False)

    start_time = time.perf_counter()

    for _ in range(100):

        _ = model(sample, training=False)

    end_time = time.perf_counter()

    infer_time_ms = (

        ((end_time - start_time) / 100)

        * 1000
    )

    return {

        "Experiment":
            exp_name,

        "Total_Params":
            int(trainable_params),

        "Model_Size_MB":
            round(model_size_mb, 3),

        "MACs_Millions":
            round(total_macs / 1e6, 3),

        "Inference_Time_ms":
            round(infer_time_ms, 3),
    }

# =====================================================================
# CONFUSION MATRIX
# =====================================================================

def build_confusion_matrix(
    model,
    X_test,
    y_test
):

    y_pred_prob = model.predict(
        X_test,
        verbose=0
    )

    y_pred = np.argmax(
        y_pred_prob,
        axis=1
    )

    y_true = np.argmax(
        y_test,
        axis=1
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    return cm

# =====================================================================
# CLASSIFICATION REPORT
# =====================================================================

def build_classification_report(
    model,
    X_test,
    y_test
):

    y_pred_prob = model.predict(
        X_test,
        verbose=0
    )

    y_pred = np.argmax(
        y_pred_prob,
        axis=1
    )

    y_true = np.argmax(
        y_test,
        axis=1
    )

    report = classification_report(

        y_true,

        y_pred,

        labels=np.arange(len(cfg.CLASS_NAMES)),

        target_names=cfg.CLASS_NAMES,

        output_dict=True,

        zero_division=0
    )

    return pd.DataFrame(report).transpose()

# =====================================================================
# MISCLASSIFIED SAMPLES
# =====================================================================

def get_misclassified_samples(
    model,
    X_test,
    y_test
):

    y_pred_prob = model.predict(
        X_test,
        verbose=0
    )

    y_pred = np.argmax(
        y_pred_prob,
        axis=1
    )

    y_true = np.argmax(
        y_test,
        axis=1
    )

    misclassified_idx = np.where(
        y_true != y_pred
    )[0]

    return {

        "indices": misclassified_idx,

        "y_true": y_true[misclassified_idx],

        "y_pred": y_pred[misclassified_idx],

        "confidence":

            np.max(
                y_pred_prob[misclassified_idx],
                axis=1
            )
    }

# =====================================================================
# MAIN METRICS
# =====================================================================

def calculate_ml_metrics(
    model,
    X_test,
    y_test,
    exp_name,
):

    """
    Main multiclass ECG metrics
    """

    y_pred_prob = model.predict(
        X_test,
        verbose=0
    )

    y_pred = np.argmax(
        y_pred_prob,
        axis=1
    )

    y_true = np.argmax(
        y_test,
        axis=1
    )

    # ================================================================
    # MAIN METRICS
    # ================================================================

    bal_acc = balanced_accuracy_score(
        y_true,
        y_pred
    )

    macro_precision = precision_score(

        y_true,
        y_pred,

        average='macro',

        zero_division=0
    )

    macro_recall = recall_score(

        y_true,
        y_pred,

        average='macro',

        zero_division=0
    )

    macro_f1 = f1_score(

        y_true,
        y_pred,

        average='macro',

        zero_division=0
    )

    report = classification_report(

        y_true,

        y_pred,

        labels=np.arange(len(cfg.CLASS_NAMES)),

        target_names=cfg.CLASS_NAMES,

        output_dict=True,

        zero_division=0
    )

    metrics = {

        "Experiment":
            exp_name,

        "Balanced_Accuracy":
            float(bal_acc),

        "Macro_Precision":
            float(macro_precision),

        "Macro_Recall":
            float(macro_recall),

        "Macro_F1":
            float(macro_f1),
    }

    # ================================================================
    # PER CLASS
    # ================================================================

    for cls in cfg.CLASS_NAMES:

        if cls in report:

            metrics[f"Sens_{cls}"] = float(
                report[cls]["recall"]
            )

            metrics[f"Precision_{cls}"] = float(
                report[cls]["precision"]
            )

            metrics[f"F1_{cls}"] = float(
                report[cls]["f1-score"]
            )

    return metrics