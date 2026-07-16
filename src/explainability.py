# =====================================================================
# FILE: explainability.py
# =====================================================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage


# =====================================================================
# GRAD-CAM 1D
# =====================================================================

def compute_gradcam_1d(
    model,
    signal,
    class_index,
    layer_name=None
):
    """
    Grad-CAM untuk sinyal ECG 1D.

    Parameters
    ----------
    model : tf.keras.Model

    signal : ndarray
        Shape:
            (2500, 3)

    class_index : int
        Target class

    layer_name : str or None
        Jika None -> otomatis mencari Conv1D terakhir
    """

    # =========================================================
    # AUTO FIND LAST CONV1D
    # =========================================================

    if layer_name is None:

        for layer in reversed(model.layers):

            if isinstance(layer, tf.keras.layers.Conv1D):
                layer_name = layer.name
                break

    if layer_name is None:
        raise ValueError(
            "Tidak ditemukan layer Conv1D pada model."
        )

    # =========================================================
    # BUILD GRAD MODEL
    # =========================================================

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(layer_name).output,
            model.output
        ]
    )

    # =========================================================
    # PREPARE INPUT
    # =========================================================

    input_tensor = tf.convert_to_tensor(
        signal[np.newaxis, ...],
        dtype=tf.float32
    )

    # =========================================================
    # GRADIENT COMPUTATION
    # =========================================================

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            input_tensor,
            training=False
        )

        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    # =========================================================
    # GLOBAL AVERAGE POOLING
    # =========================================================

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1)
    )

    conv_outputs = conv_outputs[0]

    # Weighted activation map
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    # =========================================================
    # RELU
    # =========================================================

    heatmap = tf.nn.relu(heatmap)

    # =========================================================
    # NORMALIZATION
    # =========================================================

    max_heat = tf.reduce_max(heatmap)

    if max_heat <= 1e-8:
        return np.zeros_like(heatmap.numpy())

    heatmap = heatmap / max_heat

    return heatmap.numpy()


# =====================================================================
# VISUALIZATION
# =====================================================================

def plot_gradcam_ecg(
    signal,
    heatmap,
    save_path,
    fs=250.0
):
    """
    Visualisasi Grad-CAM ECG 3-lead.

    signal:
        shape = (2500, 3)

    heatmap:
        shape = (~625,)
    """

    # =========================================================
    # UPSAMPLE HEATMAP
    # =========================================================

    zoom_factor = len(signal) / len(heatmap)

    heatmap_resized = ndimage.zoom(
        heatmap,
        zoom_factor,
        order=1
    )

    # Proteksi panjang
    heatmap_resized = heatmap_resized[:len(signal)]

    # =========================================================
    # TIME AXIS
    # =========================================================

    time_axis = np.arange(len(signal)) / fs

    # =========================================================
    # COLORMAP
    # =========================================================

    cmap = plt.get_cmap('jet')

    # =========================================================
    # FIGURE
    # =========================================================

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15, 8),
        sharex=True
    )

    lead_names = [
        'Lead I',
        'Lead II',
        'Lead III'
    ]

    for i in range(3):

        ax = axes[i]

        sig = signal[:, i]

        # =====================================================
        # ECG SIGNAL
        # =====================================================

        ax.plot(
            time_axis,
            sig,
            color='black',
            linewidth=1.2,
            zorder=2
        )

        # =====================================================
        # HEATMAP OVERLAY
        # =====================================================

        ymin = np.min(sig) - 0.5
        ymax = np.max(sig) + 0.5

        ax.imshow(
            np.expand_dims(heatmap_resized, axis=0),
            aspect='auto',
            cmap=cmap,
            extent=[
                time_axis[0],
                time_axis[-1],
                ymin,
                ymax
            ],
            alpha=0.40,
            zorder=1
        )

        ax.set_ylabel(
            lead_names[i],
            fontweight='bold'
        )

        ax.grid(
            linestyle=':',
            alpha=0.5
        )

    axes[-1].set_xlabel(
        "Time (Seconds)",
        fontweight='bold'
    )

    plt.suptitle(
        "Grad-CAM ECG Attention Visualization",
        fontsize=16,
        fontweight='bold'
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=250,
        bbox_inches='tight'
    )

    plt.close()