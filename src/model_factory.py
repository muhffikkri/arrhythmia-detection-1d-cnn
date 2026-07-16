# =====================================================================
# FILE: model_factory.py
# PURPOSE:
# - ECG CNN BACKBONE WITH SQUEEZE-AND-EXCITATION (SE)
# - STOCHASTIC DEPTH & SEPARABLE CONV
# - MULTI TEMPORAL HEAD (ATTENTION / BILSTM)
# - MULTI-LABEL TRANSITION (4-CLASS SIGMOID OUTPUT)
# =====================================================================

import tensorflow as tf
from tensorflow.keras import (
    layers,
    models,
    regularizers
)
from experiment_configs import Config
import config_ml as ml_cfg

# =====================================================================
# STOCHASTIC DEPTH
# =====================================================================
class StochasticDepth(layers.Layer):
    """
    Residual branch dropping.
    Helps reduce overfitting on deeper residual architectures.
    """
    def __init__(self, survival_probability=1.0, **kwargs):
        super().__init__(**kwargs)
        self.survival_probability = survival_probability

    def call(self, x, residual, training=None):
        if training:
            binary_tensor = tf.cast(
                tf.random.uniform([]) < self.survival_probability,
                tf.float32
            )
            x = (binary_tensor * x) / self.survival_probability
        return x + residual

# =====================================================================
# SQUEEZE-AND-EXCITATION (SE BLOCK)
# =====================================================================
def squeeze_and_excitation_block(input_tensor, ratio=8):
    """
    Squeeze-and-Excitation (SE) Block untuk Konvolusi 1D EKG.
    Menekan noise pada channel datar dan memperkuat komponen kompleks QRS.
    """
    init_channels = input_tensor.shape[-1]
    
    # 1. Squeeze: Mereduksi dimensi temporal [Batch, Timesteps, Channels] -> [Batch, 1, Channels]
    squeeze = layers.GlobalAveragePooling1D()(input_tensor)
    squeeze = layers.Reshape((1, init_channels))(squeeze)
    
    # 2. Excitation: Bottleneck Dense Layer
    excitation = layers.Dense(
        units=init_channels // ratio,
        activation='relu',
        kernel_initializer='he_normal',
        use_bias=False
    )(squeeze)
    
    excitation = layers.Dense(
        units=init_channels,
        activation='sigmoid',  # Mengunci bobot skala antara 0 hingga 1
        kernel_initializer='he_normal',
        use_bias=False
    )(excitation)
    
    # 3. Scale: Mengalikan peta fitur asli dengan bobot channel atensi
    scaled_feature_maps = layers.Multiply()([input_tensor, excitation])
    return scaled_feature_maps

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
    Wrapper for Conv1D / SeparableConv1D with proper regularization support.
    """
    if use_separable:
        x = layers.SeparableConv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            dilation_rate=dilation_rate,
            depthwise_regularizer=regularizers.l2(l2_weight),
            pointwise_regularizer=regularizers.l2(l2_weight)
        )(x)
    else:
        x = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            dilation_rate=dilation_rate,
            kernel_regularizer=regularizers.l2(l2_weight)
        )(x)
    return x

# =====================================================================
# RESIDUAL BLOCK WITH CHANNEL ATTENTION (SE)
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
    ECG Residual Block dengan Modul Squeeze-and-Excitation terintegrasi
    untuk optimasi ekstraksi morfologi gelombang P-QRS-T.
    """
    shortcut = x
    survival_probability = 1.0 - stochastic_depth_rate

    # ---------------------------------------------------------
    # CONV 1
    # ---------------------------------------------------------
    x = conv1d_wrapper(
        x=x,
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        use_separable=use_separable
    )
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # ---------------------------------------------------------
    # CONV 2
    # ---------------------------------------------------------
    x = conv1d_wrapper(
        x=x,
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        use_separable=use_separable
    )
    x = layers.BatchNormalization()(x)

    # ---------------------------------------------------------
    # CHANNEL ATTENTION BLOCK (SQUEEZE-AND-EXCITATION)
    # ---------------------------------------------------------
    # Ditambahkan di akhir ekstraksi spasial sebelum residual connection
    # x = squeeze_and_excitation_block(x, ratio=8)

    # ---------------------------------------------------------
    # RESIDUAL PROJECTION
    # ---------------------------------------------------------
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(
            filters,
            kernel_size=1,
            padding='same',
            kernel_regularizer=regularizers.l2(3e-4)
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # ---------------------------------------------------------
    # RESIDUAL BLENDING & REGULARIZATION
    # ---------------------------------------------------------
    x = StochasticDepth(survival_probability=survival_probability)(x, shortcut)
    x = layers.Activation('relu')(x)
    x = layers.SpatialDropout1D(dropout_rate)(x)

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
    stochastic_depth_rate=0.10,
    scheme="multilabel"
):
    """
    Dynamic ECG Multi-Label CNN Backbone.
    Mendukung ekstraksi fitur adaptif spasio-temporal.
    """
    if dilations is None:
        dilations = [1] * len(filters)

    # Safety Check
    if not (len(filters) == len(kernels) == len(dilations)):
        raise ValueError(
            f"Mismatch length: filters={len(filters)}, kernels={len(kernels)}, dilations={len(dilations)}"
        )

    # Input Layer
    inputs = layers.Input(shape=Config.INPUT_SHAPE)
    x = inputs

    # CNN Backbone (Spatial Encoder + SE Attention Loops)
    for i, (f, k, d) in enumerate(zip(filters, kernels, dilations)):
        current_sd_rate = stochastic_depth_rate * (i + 1) / len(filters)

        x = residual_conv_block(
            x=x,
            filters=f,
            kernel_size=k,
            dilation_rate=d,
            dropout_rate=0.15,
            use_separable=use_separable,
            stochastic_depth_rate=current_sd_rate
        )

        if i < 3:
            x = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    # Temporal Modeling
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
        x = layers.Add()([x, attn_out])
        x = layers.LayerNormalization()(x)

    # Classification Head (Transisi Mutlak ke Multi-Label)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, use_bias=False, kernel_regularizer=regularizers.l2(3e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)    
    x = layers.Dropout(0.20)(x)

    # KUNCI TRANSISI: Menggunakan total neuron 4 dan aktivasi Sigmoid

    if scheme == "multilabel":
        outputs = layers.Dense(
            ml_cfg.NUM_CLASSES,
            activation='sigmoid',
            name="multilabel_output"
        )(x)
    else :
        outputs = layers.Dense(
            Config.CLASSES,
            activation='softmax',
            name="multiclass_output"
        )(x)

    model = models.Model(inputs, outputs, name=f"ECG_Multilabel_{temporal_mode}")
    return model