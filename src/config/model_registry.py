# =====================================================================
# FILE: src/config/model_registry.py
# PURPOSE: Model registry configuration for cross-dataset validation
# =====================================================================

import os

# Registry of all models to be tested in Chapman cross-dataset evaluation.
# Feel free to toggle the "active" flag or specify "type" ("softmax" or "sigmoid").
MODEL_REGISTRY = [
    # =================================================================
    # PURE CNN SOFTMAX HEAD (MULTICLASS)
    # =================================================================
    {
        "id": "softmax_filtered_100to250_cnn",
        "path": os.path.join("Pure CNN Softmax Head", "softmax_filtered_100to250_cnn"),
        "name": "Softmax Filtered 100Hz to 250Hz CNN",
        "type": "softmax",
        "active": True
    },
    {
        "id": "softmax_filtered_100to250_cnnattention",
        "path": os.path.join("Pure CNN Softmax Head", "softmax_filtered_100to250_cnnattention"),
        "name": "Softmax Filtered 100Hz to 250Hz CNN Attention",
        "type": "softmax",
        "active": True
    },
    {
        "id": "softmax_filtered_100to250_lstm",
        "path": os.path.join("Pure CNN Softmax Head", "softmax_filtered_100to250_lstm"),
        "name": "Softmax Filtered 100Hz to 250Hz LSTM",
        "type": "softmax",
        "active": True
    },
    {
        "id": "softmax_filtered_500to250_cnn",
        "path": os.path.join("Pure CNN Softmax Head", "softmax_filtered_500to250_cnn"),
        "name": "Softmax Filtered 500Hz to 250Hz CNN",
        "type": "softmax",
        "active": True
    },
    {
        "id": "softmax_filtered_500to250_cnnattention",
        "path": os.path.join("Pure CNN Softmax Head", "softmax_filtered_500to250_cnnattention"),
        "name": "Softmax Filtered 500Hz to 250Hz CNN Attention",
        "type": "softmax",
        "active": True
    },
    {
        "id": "softmax_filtered_500to250_lstm",
        "path": os.path.join("Pure CNN Softmax Head", "softmax_filtered_500to250_lstm"),
        "name": "Softmax Filtered 500Hz to 250Hz LSTM",
        "type": "softmax",
        "active": True
    },
    {
        "id": "softmax_raw_100to250_cnn",
        "path": os.path.join("Pure CNN Softmax Head", "softmax_raw_100to250_cnn"),
        "name": "Softmax Raw 100Hz to 250Hz CNN",
        "type": "softmax",
        "active": True
    },
    {
        "id": "softmax_raw_500to250_cnn",
        "path": os.path.join("Pure CNN Softmax Head", "softmax_raw_500to250_cnn"),
        "name": "Softmax Raw 500Hz to 250Hz CNN",
        "type": "softmax",
        "active": True
    },
    # =================================================================
    # PURE CNN SIGMOID HEAD (MULTILABEL)
    # =================================================================
    {
        "id": "multilabel_100to250",
        "path": os.path.join("Pure CNN Sigmoid Head", "multilabel_100to250"),
        "name": "Sigmoid Multilabel 100Hz to 250Hz",
        "type": "sigmoid",
        "active": True
    },
    {
        "id": "multilabel_500to250",
        "path": os.path.join("Pure CNN Sigmoid Head", "multilabel_500to250"),
        "name": "Sigmoid Multilabel 500Hz to 250Hz",
        "type": "sigmoid",
        "active": True
    }
]

# Config to filter which models to test.
# Set to an empty list `[]` or `None` to evaluate all "active" models.
# Example: EVALUATE_ONLY_IDS = ["softmax_filtered_100to250_cnn", "multilabel_100to250"]
EVALUATE_ONLY_IDS = []
