# =====================================================================
# FILE: loss_factory.py
# PURPOSE:
# - STABLE FOCAL LOSS
# - LABEL SMOOTHING SUPPORT
# - ECG MULTI-CLASS LOSS FACTORY
# =====================================================================

import tensorflow as tf
import tensorflow.keras.backend as K


# =====================================================================
# STABLE FOCAL LOSS
# =====================================================================

def get_stable_focal_loss(
    gamma=2.0,
    alpha=0.5,
    label_smoothing=0.0
):
    """
    Numerically Stable Multi-class Focal Loss.

    Suitable for:
    - ECG arrhythmia imbalance
    - minority AF detection
    - multi-class ECG classification

    Formula:
        FL(pt) =
        - alpha * (1 - pt)^gamma * log(pt)
    """

    def focal_loss_fn(y_true, y_pred):

        # =========================================================
        # LABEL SMOOTHING
        # =========================================================

        if label_smoothing > 0.0:

            num_classes = tf.cast(
                tf.shape(y_true)[-1],
                tf.float32
            )

            smooth_positives = (
                1.0 - label_smoothing
            )

            smooth_negatives = (
                label_smoothing / num_classes
            )

            y_true_smoothed = (
                y_true * smooth_positives
                + smooth_negatives
            )

        else:

            y_true_smoothed = y_true

        # =========================================================
        # NUMERICAL STABILITY
        # =========================================================

        y_pred_clipped = tf.clip_by_value(

            y_pred,

            K.epsilon(),

            1.0 - K.epsilon()
        )

        # =========================================================
        # CROSS ENTROPY
        # =========================================================

        ce = (
            -y_true_smoothed
            * tf.math.log(y_pred_clipped)
        )

        ce = tf.reduce_sum(
            ce,
            axis=-1
        )

        # =========================================================
        # PT
        # =========================================================

        pt = tf.reduce_sum(

            y_true_smoothed
            * y_pred_clipped,

            axis=-1
        )

        # =========================================================
        # FOCAL WEIGHT
        # =========================================================

        focal_weight = (

            alpha
            * tf.pow(1.0 - pt, gamma)

        )

        # =========================================================
        # FINAL LOSS
        # =========================================================

        loss = focal_weight * ce

        return tf.reduce_mean(loss)

    return focal_loss_fn


# =====================================================================
# LOSS FACTORY
# =====================================================================

def get_loss_function(
    strategy='ce',
    gamma=2.0,
    alpha=0.5,
    label_smoothing=0.0
):
    """
    Loss Factory

    Available:
    - ce
    - focal
    """

    strategy = strategy.lower()

    # =========================================================
    # FOCAL LOSS
    # =========================================================

    if strategy == 'focal':

        print(
            f"[Loss] Using Focal Loss | "
            f"gamma={gamma} | "
            f"alpha={alpha} | "
            f"label_smoothing={label_smoothing}"
        )

        return get_stable_focal_loss(

            gamma=gamma,

            alpha=alpha,

            label_smoothing=label_smoothing
        )

    # =========================================================
    # STANDARD CE
    # =========================================================

    print(
        f"[Loss] Using CategoricalCrossentropy | "
        f"label_smoothing={label_smoothing}"
    )

    return tf.keras.losses.CategoricalCrossentropy(

        label_smoothing=label_smoothing

    )
