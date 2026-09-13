# =====================================================================
# DESAIN MODEL: 1D-CNN
# PIPELINE: AKUISISI, VALIDASI, PREPROCESSING, TRAINING, & EVALUASI
# =====================================================================
"""
Deskripsi: Berkas ini berisi implementasi menyeluruh (end-to-end) dari skema 
           klasifikasi sinyal EKG multi-lead untuk deteksi aritmia.
           Sistem mencakup akuisisi data, prapemrosesan DSP tingkat lanjut, 
           validasi stratifikasi data, arsitektur deep learning 1D-CNN, 
           hingga evaluasi kinerja model.
"""

import os
import gc
import sys
import random
import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.signal import resample_poly, butter, filtfilt, medfilt
import pywt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
import tensorflow.keras.backend as K
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# =====================================================================
# KONFIGURASI GLOBAL & REPRODUSIBILITAS
# =====================================================================
class GlobalConfig:
    # Sensor Analog Front-End (AFE) ADS1293 dikondisikan pada fs = 250 Hz
    FS = 250.0  
    INPUT_SHAPE = (2500, 3)  # Durasi 10 detik x 3 lead (I, II, III)
    TARGET_CLASSES = ["Normal", "AF", "Takikardia", "Bradikardia"]
    NUM_CLASSES = len(TARGET_CLASSES)
    
    # Parameter Pelatihan
    BATCH_SIZE = 64
    EPOCHS = 35  
    LEARNING_RATE = 3e-4
    COSINE_DECAY_STEPS = 3000
    COSINE_ALPHA = 1e-2
    STOCHASTIC_DEPTH_RATE = 0.0
    USE_SEPARABLE_CONV = False
    
    # Folder Keluaran
    OUTPUT_DIR = "./output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_reproducibility(seed=42):
    """Menjamin hasil eksperimen yang konsisten melalui seeding acak."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_reproducibility(42)


# =====================================================================
# TAHAP 1: AKUISISI DATA & STRATIFIKASI VALIDASI
# =====================================================================
def load_dataset_manifest(manifest_csv_path=None, data_folder=None):
    """
    Mengakuisisi berkas manifest rekaman EKG dan membagi data secara
    stratifikasi berbasis strat_fold untuk memisahkan Train, Val, dan Test.
    """
    if not manifest_csv_path or not os.path.exists(manifest_csv_path):
        raise FileNotFoundError(f"Manifest file tidak ditemukan di: {manifest_csv_path}")
    if not data_folder or not os.path.exists(data_folder):
        raise FileNotFoundError(f"Folder data tidak ditemukan di: {data_folder}")
        
    print(f"[Akuisisi Data] Membaca manifest fisik dari: {manifest_csv_path}")
    df = pd.read_csv(manifest_csv_path)
    
    # Filter kelas target
    df_filtered = df[df["target_class"].isin(GlobalConfig.TARGET_CLASSES)].reset_index(drop=True)
    
    X_splits = {"train": [], "val": [], "test": []}
    y_splits = {"train": [], "val": [], "test": []}
    
    for _, row in df_filtered.iterrows():
        file_path = os.path.join(data_folder, row["filename_npy"])
        if not os.path.exists(file_path):
            continue
        
        try:
            # Membaca berkas numpy rekaman sinyal EKG
            signal_data = np.load(file_path).astype(np.float32)
            
            # Konstruksi label multi-hot / one-hot
            label = np.zeros(GlobalConfig.NUM_CLASSES, dtype=np.float32)
            class_idx = GlobalConfig.TARGET_CLASSES.index(row["target_class"])
            label[class_idx] = 1.0
            
            fold = int(row["strat_fold"])
            if fold in range(1, 9):
                split = "train"
            elif fold == 9:
                split = "val"
            elif fold == 10:
                split = "test"
            else:
                continue
            
            X_splits[split].append(signal_data)
            y_splits[split].append(label)
        except Exception as e:
            continue
            
    return (
        np.array(X_splits["train"], dtype=np.float32),
        np.array(y_splits["train"], dtype=np.float32),
        np.array(X_splits["val"], dtype=np.float32),
        np.array(y_splits["val"], dtype=np.float32),
        np.array(X_splits["test"], dtype=np.float32),
        np.array(y_splits["test"], dtype=np.float32)
    )


# =====================================================================
# TAHAP 2: PRAPEMROSESAN SINYAL (DIGITAL SIGNAL PROCESSING ENGINE)
# =====================================================================
def sanitize_signal(signal_data):
    """Menangani nilai abnormal (NaN dan Infinity) dari instrumen akuisisi."""
    signal_data = np.asarray(signal_data, dtype=np.float32)
    return np.nan_to_num(signal_data, nan=0.0, posinf=0.0, neginf=0.0)

def ensure_length(signal_data, target_len=2500):
    """Mengondisikan panjang biosinyal lewat pemotongan tengah atau padding nol."""
    signal_data = sanitize_signal(signal_data)
    current_len = signal_data.shape[0]
    
    if current_len == target_len:
        return signal_data
    elif current_len > target_len:
        start = (current_len - target_len) // 2
        return signal_data[start:start + target_len, :]
    else:
        pad_len = target_len - current_len
        return np.pad(signal_data, ((0, pad_len), (0, 0)), mode='constant', constant_values=0.0)

def apply_poly_resample(signal_data, src_fs, target_fs=250.0):
    """Penyelarasan frekuensi sampling menggunakan Polyphase FIR Resampling."""
    signal_data = sanitize_signal(signal_data)
    if src_fs == target_fs:
        return signal_data
    gcd = np.gcd(int(src_fs), int(target_fs))
    up = int(target_fs // gcd)
    down = int(src_fs // gcd)
    resampled = resample_poly(signal_data, up, down, axis=0)
    return sanitize_signal(resampled)

def apply_wavelet_denoising(signal_data, wavelet='db4', level=4):
    """
    Menghilangkan noise listrik dan otot menggunakan DWT (Discrete Wavelet Transform)
    Daubechies 4 (db4) dengan ambang batas soft-thresholding adaptif.
    """
    signal_data = sanitize_signal(signal_data)
    out = np.zeros_like(signal_data)
    
    for i in range(signal_data.shape[1]):
        max_lvl = pywt.dwt_max_level(signal_data.shape[0], pywt.Wavelet(wavelet).dec_len)
        safe_level = min(level, max_lvl)
        
        coeffs = pywt.wavedec(signal_data[:, i], wavelet, level=safe_level)
        threshold = np.std(coeffs[-1]) / 2.0  
        
        coeffs[1:] = [pywt.threshold(c, value=threshold, mode='soft') for c in coeffs[1:]]
        reconstructed = pywt.waverec(coeffs, wavelet)
        out[:, i] = reconstructed[:signal_data.shape[0]]
        
    return sanitize_signal(out)

def apply_median_baseline(signal_data, kernel_size=51):
    """Koreksi pergeseran garis dasar (Baseline Wander) menggunakan filter median."""
    signal_data = sanitize_signal(signal_data)
    if kernel_size % 2 == 0:
        kernel_size += 1
    out = np.zeros_like(signal_data)
    for i in range(signal_data.shape[1]):
        baseline = medfilt(signal_data[:, i], kernel_size=kernel_size)
        out[:, i] = signal_data[:, i] - baseline
    return sanitize_signal(out)

def apply_butter_bandpass(signal_data, fs, lowcut=0.5, highcut=45.0, order=4):
    """
    Penyaringan pita frekuensi lolos (Bandpass) Butterworth Orde 4 
    untuk menyeleksi frekuensi klinis representasi morfologi P-QRS-T EKG.
    """
    signal_data = sanitize_signal(signal_data)
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.99)
    if low >= high:
        return signal_data
    try:
        b, a = butter(order, [low, high], btype='band')
        filtered = filtfilt(b, a, signal_data, axis=0)
        return sanitize_signal(filtered)
    except:
        return signal_data

def apply_zscore_clip(signal_data, clip_val=5.0):
    """Z-Score Normalization per channel dan clipping batas ekstrim outlier."""
    signal_data = sanitize_signal(signal_data)
    mean = np.mean(signal_data, axis=0)
    std = np.std(signal_data, axis=0) + 1e-8
    norm = (signal_data - mean) / std
    return np.clip(norm, -clip_val, clip_val)

def advanced_cleaning_pipeline(raw_signal, src_fs, target_fs=250.0):
    """
    Pipeline Utama DSP: Wavelet Denoising -> Median Baseline Correction ->
    Butterworth Bandpass -> Resampling -> Z-score Normalization.
    """
    x = sanitize_signal(raw_signal)
    
    # 1. Wavelet Denoising
    x = apply_wavelet_denoising(x, wavelet='db4', level=4)
    
    # 2. Baseline Wander Removal
    kernel_size = 101 if src_fs >= 500 else 51
    x = apply_median_baseline(x, kernel_size=kernel_size)
    
    # 3. Butterworth Filter
    highcut = 100.0 if src_fs >= 250 else 45.0
    x = apply_butter_bandpass(x, fs=src_fs, lowcut=0.5, highcut=highcut, order=4)
    
    # 4. Resample
    if src_fs != target_fs:
        x = apply_poly_resample(x, src_fs, target_fs)
        
    # 5. Normalization & Clipping
    x = apply_zscore_clip(x)
    
    return sanitize_signal(x)


# =====================================================================
# TAHAP 3: DATA VALIDATION & PIPELINE TRAINING
# =====================================================================
def create_tf_dataset(X, y, batch_size=64, is_training=False):
    """Konstruksi pipeline input tf.data.Dataset yang teroptimasi."""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
        
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(tf.data.AUTOTUNE)


# =====================================================================
# TAHAP 4: ARSITEKTUR MODEL DEEP LEARNING (1D-CNN)
# =====================================================================

@tf.keras.utils.register_keras_serializable(package="model_factory")
class StochasticDepth(layers.Layer):
    """Stochastic Depth (Residual Branch Dropping) untuk regularisasi."""
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

    def get_config(self):
        config = super().get_config()
        config.update({"survival_probability": self.survival_probability})
        return config

def conv1d_wrapper(x, filters, kernel_size, dilation_rate=1, use_separable=False, l2_weight=3e-4):
    """Fungsi pembungkus blok konvolusi 1D standar / Separable Conv."""
    if use_separable:
        return layers.SeparableConv1D(
            filters=filters, kernel_size=kernel_size, padding='same',
            dilation_rate=dilation_rate,
            depthwise_regularizer=regularizers.l2(l2_weight),
            pointwise_regularizer=regularizers.l2(l2_weight)
        )(x)
    else:
        return layers.Conv1D(
            filters=filters, kernel_size=kernel_size, padding='same',
            dilation_rate=dilation_rate,
            kernel_regularizer=regularizers.l2(l2_weight)
        )(x)

def residual_conv_block(x, filters, kernel_size, dilation_rate=1, dropout_rate=0.30, use_separable=False, stochastic_depth_rate=0.0):
    """Blok Residual spasial residual conv 1D."""
    shortcut = x
    survival_prob = 1.0 - stochastic_depth_rate
    
    # Conv 1
    x = conv1d_wrapper(x, filters, kernel_size, dilation_rate, use_separable)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Conv 2
    x = conv1d_wrapper(x, filters, kernel_size, dilation_rate, use_separable)
    x = layers.BatchNormalization()(x)
    
    # Residual Projection
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, kernel_size=1, padding='same', kernel_regularizer=regularizers.l2(3e-4))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
        
    x = StochasticDepth(survival_probability=survival_prob)(x, shortcut)
    x = layers.Activation('relu')(x)
    x = layers.SpatialDropout1D(dropout_rate)(x)
    return x

def build_cnn(input_shape, num_classes):
    """Model 1D-CNN (Morfologi Spasial untuk Klasifikasi EKG)."""
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    filters = [64, 128, 256, 256, 512]
    kernels = [15, 11, 7, 5, 3]
    dilations = [1, 2, 4, 8, 16]
    
    for i, (f, k, d) in enumerate(zip(filters, kernels, dilations)):
        sd_rate = GlobalConfig.STOCHASTIC_DEPTH_RATE * (i + 1) / len(filters)
        x = residual_conv_block(x, f, k, d, dropout_rate=0.15, use_separable=GlobalConfig.USE_SEPARABLE_CONV, stochastic_depth_rate=sd_rate)
        if i < 3:
            x = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
            
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, use_bias=False, kernel_regularizer=regularizers.l2(3e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.20)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid', name="sigmoid_output")(x)
    
    return models.Model(inputs, outputs, name="ECG_1D_CNN")

# -------------------------------------------------------------
# NUMERICALLY STABLE FOCAL LOSS
# -------------------------------------------------------------
def get_stable_focal_loss(gamma=2.0, alpha=0.5, label_smoothing=0.0):
    """Focal loss multi-kelas stabil untuk menangani ketidakseimbangan dataset."""
    def focal_loss_fn(y_true, y_pred):
        if label_smoothing > 0.0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true_smoothed = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)
        else:
            y_true_smoothed = y_true
            
        y_pred_clipped = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        
        ce = -y_true_smoothed * tf.math.log(y_pred_clipped)
        ce = tf.reduce_sum(ce, axis=-1)
        
        pt = tf.reduce_sum(y_true_smoothed * y_pred_clipped, axis=-1)
        focal_weight = alpha * tf.pow(1.0 - pt, gamma)
        
        return tf.reduce_mean(focal_weight * ce)
    return focal_loss_fn


# =====================================================================
# TAHAP 5: EVALUASI KINERJA & OPTIMALISASI THRESHOLD
# =====================================================================
def perform_threshold_tuning(model, X_val, y_val):
    """Optimalisasi batas keputusan (threshold tuning) adaptif per kelas."""
    print("[Evaluasi] Mengoptimasi Threshold Keputusan per Kelas pada Set Validasi...")
    y_val_pred_prob = model.predict(X_val, batch_size=GlobalConfig.BATCH_SIZE, verbose=0)
    
    best_thresholds = []
    threshold_range = np.linspace(0.1, 0.9, 81)
    
    for cl_idx, class_name in enumerate(GlobalConfig.TARGET_CLASSES):
        best_f1 = 0.0
        best_th = 0.5
        for th in threshold_range:
            y_pred_bin = (y_val_pred_prob[:, cl_idx] >= th).astype(int)
            y_true_bin = y_val[:, cl_idx].astype(int)
            score = f1_score(y_true_bin, y_pred_bin, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_th = th
        print(f"  -> Threshold Optimal {class_name}: {best_th:.2f} (F1 Validasi: {best_f1:.4f})")
        best_thresholds.append(best_th)
    return best_thresholds

def calculate_metrics_report(y_true, y_pred_prob, thresholds):
    """Menghitung metrik evaluasi lengkap per kelas: F1, Precision, Recall, AUC-ROC."""
    y_pred_bin = np.zeros_like(y_pred_prob)
    for idx in range(GlobalConfig.NUM_CLASSES):
        y_pred_bin[:, idx] = (y_pred_prob[:, idx] >= thresholds[idx]).astype(int)
        
    metrics = {}
    for i, cls in enumerate(GlobalConfig.TARGET_CLASSES):
        metrics[cls] = {
            "F1-Score": f1_score(y_true[:, i], y_pred_bin[:, i], zero_division=0),
            "Precision": precision_score(y_true[:, i], y_pred_bin[:, i], zero_division=0),
            "Recall": recall_score(y_true[:, i], y_pred_bin[:, i], zero_division=0),
            "AUC-ROC": roc_auc_score(y_true[:, i], y_pred_prob[:, i])
        }
    return metrics, y_pred_bin


# =====================================================================
# BLOK UTAMA JALANNYA EKSPERIMEN (EXECUTION ENGINE)
# =====================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("MEMULAI PIPELINE UJI COBA EKSPERIMEN HAK CIPTA")
    print("="*70)
    
    # 1. Akuisisi dan Validasi Pemisah Data
    manifest_file = "./dataset/resample/manifest_ptbxl.csv"
    data_folder = "./dataset/resample/cleaned_100_to_250"
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset_manifest(
        manifest_csv_path=manifest_file,
        data_folder=data_folder
    )
    
    print(f"[Info Dataset] Ukuran Data Terakuisisi:")
    print(f"  Train : X={X_train.shape}, y={y_train.shape}")
    print(f"  Val   : X={X_val.shape}, y={y_val.shape}")
    print(f"  Test  : X={X_test.shape}, y={y_test.shape}")
    
    # Preprocessing simulasi pada set data uji
    print("\n[DSP Demo] Menjalankan Pipeline Preprocessing pada Sampel Tunggal...")
    sample_signal = X_train[0]
    processed_sample = advanced_cleaning_pipeline(sample_signal, src_fs=100.0, target_fs=250.0)
    print(f"  Input Shape: {sample_signal.shape} -> Output Shape: {processed_sample.shape}")
    
    # Konstruksi Dataset TF
    train_ds = create_tf_dataset(X_train, y_train, batch_size=GlobalConfig.BATCH_SIZE, is_training=True)
    val_ds = create_tf_dataset(X_val, y_val, batch_size=GlobalConfig.BATCH_SIZE, is_training=False)
    
    # Membangun Model Utama (1D-CNN)
    print(f"\nMembangun arsitektur model 1D-CNN...")
    model = build_cnn(GlobalConfig.INPUT_SHAPE, GlobalConfig.NUM_CLASSES)
    
    # Optimizer scheduler Cosine Decay
    lr_schedule = CosineDecay(
        initial_learning_rate=GlobalConfig.LEARNING_RATE,
        decay_steps=GlobalConfig.COSINE_DECAY_STEPS,
        alpha=GlobalConfig.COSINE_ALPHA
    )
    
    # Kompilasi model dengan Custom Stable Focal Loss
    loss_fn = get_stable_focal_loss(gamma=2.0, alpha=0.5)
    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss=loss_fn,
        metrics=['binary_accuracy']
    )
    
    # Menampilkan ringkasan model
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=os.path.join(GlobalConfig.OUTPUT_DIR, "best_model_cnn.keras"), monitor="val_loss", save_best_only=True, verbose=1)
    ]
    
    # Training Model
    print(f"\nMelatih model 1D-CNN...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=GlobalConfig.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Threshold tuning berbasis Validation Set
    best_thresholds = perform_threshold_tuning(model, X_val, y_val)
    
    # Evaluasi Akhir pada Isolated Test Set
    print(f"\n[Evaluasi] Menguji Model 1D-CNN pada Test Set...")
    y_test_pred_prob = model.predict(X_test, batch_size=GlobalConfig.BATCH_SIZE, verbose=0)
    
    metrics, y_pred_bin = calculate_metrics_report(y_test, y_test_pred_prob, best_thresholds)
    
    # Cetak laporan klasifikasi skikit-learn
    print(f"\nClassification Report untuk 1D-CNN:")
    print(classification_report(y_test, y_pred_bin, target_names=GlobalConfig.TARGET_CLASSES, zero_division=0))
    
    # Bersihkan memori GPU/RAM
    del model
    K.clear_session()
    gc.collect()

    print("\n" + "="*70)
    print("Seluruh skema eksperimen telah berhasil dieksekusi secara tuntas.")
    print("="*70 + "\n")
