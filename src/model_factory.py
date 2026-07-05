# =====================================================================
# FILE: model_factory.py
# PURPOSE:
# - ECG CNN BACKBONE
# - STOCHASTIC DEPTH
# - SEPARABLE CONV SUPPORT
# - MULTI TEMPORAL HEAD
# - OVERFITTING REDUCTION
# =====================================================================

import tensorflow as tf

from tensorflow.keras import (
    layers,
    models,
    regularizers
)

from experiment_configs import Config


# =====================================================================
# STOCHASTIC DEPTH
# =====================================================================

class StochasticDepth(layers.Layer):
    """
    Residual branch dropping.

    Helps reduce overfitting
    on deeper residual architectures.
    """

    def __init__(
        self,
        survival_probability=1.0,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.survival_probability = survival_probability

    def call(
        self,
        x,
        residual,
        training=None
    ):

        if training:

            binary_tensor = tf.cast(
                tf.random.uniform([]) <
                self.survival_probability,
                tf.float32
            )

            x = (
                binary_tensor * x
            ) / self.survival_probability

        return x + residual


# =====================================================================
# CONV WRAPPER
# =====================================================================

def conv1d_wrapper(
    x,
    filters,
    kernel_size,
    dilation_rate=1,
    use_separable=True,
    l2_weight=3e-4
):
    """
    Wrapper for Conv1D / SeparableConv1D
    with proper regularization support.
    """

    if use_separable:

        x = layers.SeparableConv1D(

            filters=filters,

            kernel_size=kernel_size,

            padding='same',

            dilation_rate=dilation_rate,

            depthwise_regularizer=
            regularizers.l2(l2_weight),

            pointwise_regularizer=
            regularizers.l2(l2_weight)

        )(x)

    else:

        x = layers.Conv1D(

            filters=filters,

            kernel_size=kernel_size,

            padding='same',

            dilation_rate=dilation_rate,

            kernel_regularizer=
            regularizers.l2(l2_weight)

        )(x)

    return x


# =====================================================================
# RESIDUAL BLOCK
# =====================================================================

def residual_conv_block(
    x,
    filters,
    kernel_size,
    dilation_rate=1,
    dropout_rate=0.30,
    use_separable=True,
    stochastic_depth_rate=0.0
):
    """
    ECG Residual Block
    optimized for morphology learning.
    """

    shortcut = x

    survival_probability = (
        1.0 - stochastic_depth_rate
    )

    # =========================================================
    # CONV 1
    # =========================================================

    x = conv1d_wrapper(

        x=x,

        filters=filters,

        kernel_size=kernel_size,

        dilation_rate=dilation_rate,

        use_separable=use_separable
    )

    x = layers.BatchNormalization()(x)

    # ReLU lebih stabil untuk ECG CNN
    x = layers.Activation('relu')(x)

    # =========================================================
    # CONV 2
    # =========================================================

    x = conv1d_wrapper(

        x=x,

        filters=filters,

        kernel_size=kernel_size,

        dilation_rate=dilation_rate,

        use_separable=use_separable
    )

    x = layers.BatchNormalization()(x)

    # =========================================================
    # RESIDUAL PROJECTION
    # =========================================================

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(
            filters,
            kernel_size=1,
            padding='same',
            kernel_regularizer=regularizers.l2(3e-4)
        )(shortcut)
        
        # skala shortcut sejajar dengan keluaran cabang utama (x)
        shortcut = layers.BatchNormalization()(shortcut)

    # =========================================================
    # STOCHASTIC DEPTH
    # =========================================================

    x = StochasticDepth(
        survival_probability=
        survival_probability
    )(x, shortcut)

    x = layers.Activation('relu')(x)

    # Spatial dropout cocok untuk ECG
    x = layers.SpatialDropout1D(
        dropout_rate
    )(x)

    return x


# =====================================================================
# MAIN MODEL FACTORY
# =====================================================================

def build_dynamic_cnn(
    filters,
    kernels,
    dilations=None,
    temporal_mode="Pure_CNN",
    use_separable=True,
    stochastic_depth_rate=0.10
):
    """
    Dynamic ECG CNN
    for Arrhythmia Classification.
    """

    if dilations is None:

        dilations = [1] * len(filters)

    # =========================================================
    # SAFETY CHECK
    # =========================================================

    if not (
        len(filters)
        == len(kernels)
        == len(dilations)
    ):

        raise ValueError(
            f"Mismatch length:"
            f" filters={len(filters)},"
            f" kernels={len(kernels)},"
            f" dilations={len(dilations)}"
        )

    # =========================================================
    # INPUT
    # =========================================================

    inputs = layers.Input(
        shape=Config.INPUT_SHAPE
    )

    x = inputs

    # =========================================================
    # CNN BACKBONE
    # =========================================================

    for i, (f, k, d) in enumerate(

        zip(
            filters,
            kernels,
            dilations
        )
    ):

        current_sd_rate = (
            stochastic_depth_rate
            * (i + 1)
            / len(filters)
        )

        x = residual_conv_block(

            x=x,

            filters=f,

            kernel_size=k,

            dilation_rate=d,

            dropout_rate=0.15,

            use_separable=use_separable,

            stochastic_depth_rate=
            current_sd_rate
        )

        # Moderate pooling
        if i < 3:

            x = layers.MaxPooling1D(

                pool_size=2,

                strides=2,

                padding='same'

            )(x)

    # =========================================================
    # TEMPORAL MODELING
    # =========================================================

    if temporal_mode == "CNN_BiLSTM":

        x = layers.Bidirectional(

            layers.LSTM(

                64,

                return_sequences=True,

                dropout=0.20,

                recurrent_dropout=0.10

            )

        )(x)

        x = layers.LayerNormalization()(x)

    elif temporal_mode == "CNN_Attention":

        attn_out = layers.MultiHeadAttention(

            num_heads=4,

            key_dim=32,

            dropout=0.20

        )(x, x)

        x = layers.Add()([
            x,
            attn_out
        ])

        x = layers.LayerNormalization()(x)

    # =========================================================
    # CLASSIFICATION HEAD
    # =========================================================

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(
        128,
        use_bias=False, 
        kernel_regularizer=regularizers.l2(3e-4)
    )(x)
    
    x = layers.BatchNormalization()(x)

    x = layers.Activation('relu')(x)    

    x = layers.Dropout(0.20)(x)

    outputs = layers.Dense(

        Config.CLASSES,

        activation='softmax'

    )(x)

    # =========================================================
    # BUILD MODEL
    # =========================================================

    model = models.Model(

        inputs,

        outputs,

        name=f"ECG_{temporal_mode}"

    )

    return model