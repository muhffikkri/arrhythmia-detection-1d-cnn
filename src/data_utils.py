# FILE: data_utils.py
# PURPOSE:
# - SAFE ECG AUGMENTATION
# - MIXUP REGULARIZATION
# - NO DEFAULT SMOTE
# - FAST TF.DATA PIPELINE
# - ECG OVERFITTING REDUCTION

import numpy as np
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight

# OPTIONAL ONLY

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

# =====================================================================

# GLOBAL CONSTANT

# =====================================================================

ECG_FS = 250.0
NUM_CLASSES = 5

# =====================================================================

# CLASS WEIGHT

# =====================================================================

def generate_class_weights(y_onehot):
    y_labels = np.argmax(y_onehot, axis=1)

    classes = np.unique(y_labels)

    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_labels
    )

    class_weights = {
        int(c): float(w)
        for c, w in zip(classes, weights)
    }

    print("\n[Class Weight]")
    print(class_weights)

    return class_weights

# SAFE PHYSIOLOGICAL NOISE

def inject_physiological_noise(signal):

    signal = signal.astype(np.float32)

    # ---------------------------------------------------------
    # SMALL GAUSSIAN NOISE
    # ---------------------------------------------------------
    noise_std = np.random.uniform(0.001, 0.004)

    noise = np.random.normal(
        loc=0.0,
        scale=noise_std,
        size=signal.shape
    ).astype(np.float32)

    augmented = signal + noise

    # ---------------------------------------------------------
    # VERY SMALL BASELINE WANDER
    # ---------------------------------------------------------
    if np.random.rand() < 0.20:

        seq_len = signal.shape[0]

        t = np.arange(seq_len) / ECG_FS

        freq = np.random.uniform(0.10, 0.25)

        phase = np.random.uniform(0.0, 2 * np.pi)

        amplitude = np.random.uniform(0.005, 0.02)

        baseline = amplitude * np.sin(
            2 * np.pi * freq * t + phase
        )

        baseline = np.expand_dims(
            baseline,
            axis=-1
        )

        augmented = augmented + baseline

    # ---------------------------------------------------------
    # SMALL GAIN SCALING
    # ---------------------------------------------------------
    if np.random.rand() < 0.20:

        gain = np.random.uniform(0.99, 1.01)

        augmented = augmented * gain

    return augmented.astype(np.float32)

# OFFLINE MINORITY AUGMENTATION
def balance_minority_classes(
    X_train,
    y_train,
    target_ratio=0.75,
    random_state=42
    ):

    print("\n[Augmentation] Minority Class Balancing")

    np.random.seed(random_state)

    y_labels = np.argmax(y_train, axis=1)

    classes, counts = np.unique(
        y_labels,
        return_counts=True
    )

    max_count = np.max(counts)

    print(f"Original Distribution : {dict(zip(classes, counts))}")

    X_balanced = list(X_train)
    y_balanced = list(y_train)

    for cls in classes:

        cls_indices = np.where(
            y_labels == cls
        )[0]

        current_count = len(cls_indices)

        desired_count = int(
            max_count * target_ratio
        )

        if current_count >= desired_count:
            continue

        deficit = desired_count - current_count

        print(
            f" -> Augmenting class {cls} "
            f"with {deficit} samples"
        )

        sampled_indices = np.random.choice(
            cls_indices,
            size=deficit,
            replace=True
        )

        for idx in sampled_indices:

            signal = X_train[idx]

            augmented = inject_physiological_noise(
                signal
            )

            X_balanced.append(augmented)
            y_balanced.append(y_train[idx])

    X_balanced = np.array(
        X_balanced,
        dtype=np.float32
    )

    y_balanced = np.array(
        y_balanced,
        dtype=np.float32
    )

    shuffle_idx = np.random.permutation(
        len(X_balanced)
    )

    X_balanced = X_balanced[shuffle_idx]
    y_balanced = y_balanced[shuffle_idx]

    print("\nBalanced Distribution")

    print(
        np.bincount(
            np.argmax(y_balanced, axis=1)
        )
    )

    return X_balanced, y_balanced

# OPTIONAL SMOTE

def apply_smote_tomek(
    X_train,
    y_train,
    method='smote_tomek'
    ):
    print(
        f"\n[WARNING] Experimental {method.upper()}"
    )

    samples, seq_len, leads = X_train.shape

    X_flat = X_train.reshape(
        samples,
        seq_len * leads
    )

    y_labels = np.argmax(
        y_train,
        axis=1
    )

    if method == 'smote':

        balancer = SMOTE(
            random_state=42
        )

    elif method == 'smote_tomek':

        balancer = SMOTETomek(
            random_state=42
        )

    else:

        return X_train, y_train

    X_bal_flat, y_bal_labels = balancer.fit_resample(
        X_flat,
        y_labels
    )

    X_balanced = X_bal_flat.reshape(
        -1,
        seq_len,
        leads
    ).astype(np.float32)

    y_balanced = tf.keras.utils.to_categorical(
        y_bal_labels,
        num_classes=NUM_CLASSES
    ).astype(np.float32)

    return X_balanced, y_balanced

# MIXUP
@tf.function
# =====================================================================
# MIXUP ECG
# =====================================================================

@tf.function
def mixup_ecg(signal, label, alpha=0.3):

    label = tf.cast(label, tf.float32)
    batch_size = tf.shape(signal)[0]
    apply_mix = tf.random.uniform([]) < 0.7

    def do_mixup():

        indices = tf.random.shuffle(
            tf.range(batch_size)
        )

        shuffled_signal = tf.gather(
            signal,
            indices
        )

        shuffled_label = tf.gather(
            label,
            indices
        )

        lam = tf.random.uniform(
            shape=[batch_size, 1, 1],
            minval=alpha,
            maxval=1.0
        )

        lam_label = tf.reshape(
            lam,
            [batch_size, 1]
        )

        mixed_signal = (
            lam * signal
            + (1.0 - lam) * shuffled_signal
        )

        mixed_label = (
            lam_label * label
            + (1.0 - lam_label) * shuffled_label
        )

        return mixed_signal, mixed_label

    return tf.cond(
        apply_mix,
        do_mixup,
        lambda: (signal, label)
    )


# ONLINE ECG AUGMENTATION

@tf.function
def dynamic_ecg_augmentation(
    signal,
    label
    ):

# LEAD DROPOUT
    if tf.random.uniform([]) < 0.03:

        lead_idx = tf.random.uniform(
            [],
            minval=0,
            maxval=3,
            dtype=tf.int32
        )

        mask = 1.0 - tf.one_hot(
            lead_idx,
            3,
            dtype=tf.float32
        )

        signal = signal * mask

    # ---------------------------------------------------------
    # SMALL NOISE
    # ---------------------------------------------------------
    if tf.random.uniform([]) < 0.20:

        noise = tf.random.normal(
            shape=tf.shape(signal),
            mean=0.0,
            stddev=0.003
        )

        signal = signal + noise

    # ---------------------------------------------------------
    # GAIN
    # ---------------------------------------------------------
    if tf.random.uniform([]) < 0.15:

        gain = tf.random.uniform(
            [],
            minval=0.99,
            maxval=1.01
        )

        signal = signal * gain

    return signal, label

# TF.DATA PIPELINE
def create_tf_dataset(
    X,
    y,
    batch_size=64,
    is_training=False,
    use_augmentation=False,
    use_mixup=False,
    mixup_alpha=0.3
):
    """
    Optimized tf.data pipeline
    with augmentation + mixup support.
    """

    # =========================================================
    # ENSURE FLOAT32
    # =========================================================

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # =========================================================
    # DATASET
    # =========================================================

    dataset = tf.data.Dataset.from_tensor_slices(
        (X, y)
    )

    # =========================================================
    # TRAINING PIPELINE
    # =========================================================

    if is_training:

        dataset = dataset.shuffle(
            buffer_size=len(X),
            reshuffle_each_iteration=True
        )

        # -----------------------------------------------------
        # ONLINE ECG AUGMENTATION
        # -----------------------------------------------------

        if use_augmentation:

            dataset = dataset.map(
                dynamic_ecg_augmentation,
                num_parallel_calls=tf.data.AUTOTUNE
            )

    # =========================================================
    # BATCH
    # =========================================================

    dataset = dataset.batch(batch_size)

    # =========================================================
    # MIXUP AFTER BATCHING
    # =========================================================

    if is_training and use_mixup:

        dataset = dataset.map(

            lambda signals, labels: mixup_ecg(
                signals,
                labels,
                alpha=mixup_alpha
            ),

            num_parallel_calls=tf.data.AUTOTUNE
        )

    # =========================================================
    # PREFETCH
    # =========================================================

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset

# DATASET DISTRIBUTION
def print_dataset_distribution(
    y_onehot,
    title="Dataset"
    ):

    y_labels = np.argmax(
        y_onehot,
        axis=1
    )

    classes, counts = np.unique(
        y_labels,
        return_counts=True
    )

    print(f"\n[{title}]")

    total = np.sum(counts)

    for cls, cnt in zip(classes, counts):

        ratio = cnt / total

        print(
            f"Class {cls}: "
            f"{cnt} samples "
            f"({ratio:.4f})"
        )
