# =====================================================================
# FILE 2: preprocessing.py — ADVANCED DSP ENGINE (RESEARCH-GRADE v5.0)
# =====================================================================

import numpy as np
from scipy.signal import (
    resample_poly,
    butter,
    filtfilt,
    medfilt
)
import pywt


# =========================================================
# GLOBAL CONFIG
# =========================================================

PIPELINE_VERSION = "v5.0_research_grade"

DEFAULT_CLIP_MIN = -5.0
DEFAULT_CLIP_MAX = 5.0


# =========================================================
# SAFETY UTILITIES
# =========================================================

def sanitize_signal(signal):
    """
    Membersihkan NaN / Inf sebelum DSP.
    """
    signal = np.asarray(signal, dtype=np.float32)

    return np.nan_to_num(
        signal,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


def validate_signal_shape(signal):
    """
    Memastikan format:
    [timesteps, channels]
    """
    if signal.ndim != 2:
        raise ValueError(
            f"Signal harus 2D [T,C], didapat shape={signal.shape}"
        )

    if signal.shape[0] < 32:
        raise ValueError(
            "Signal terlalu pendek untuk diproses."
        )

    return signal


# =========================================================
# LENGTH HANDLER
# =========================================================

def ensure_length(signal, target_len):
    """
    Mengondisikan panjang biosinyal.

    STRATEGI:
    - Center Crop
    - Zero Padding

    Zero padding dipilih agar baseline tetap stabil di 0.
    """

    signal = sanitize_signal(signal)

    current_len = signal.shape[0]

    if current_len == target_len:
        return signal

    elif current_len > target_len:
        start = (current_len - target_len) // 2

        return signal[start:start + target_len, :]

    else:
        pad_len = target_len - current_len

        return np.pad(
            signal,
            ((0, pad_len), (0, 0)),
            mode='constant',
            constant_values=0.0
        )


# =========================================================
# RESAMPLING
# =========================================================

def apply_poly_resample(signal, src_fs, target_fs):
    """
    Polyphase FIR Resampling.

    Lebih stabil dibanding:
    - linear interpolation
    - scipy.resample FFT

    untuk biosignal ECG.
    """

    signal = sanitize_signal(signal)

    if src_fs == target_fs:
        return signal

    gcd = np.gcd(int(src_fs), int(target_fs))

    up = int(target_fs // gcd)
    down = int(src_fs // gcd)

    resampled = resample_poly(
        signal,
        up,
        down,
        axis=0
    )

    return sanitize_signal(resampled)


# =========================================================
# WAVELET DENOISING
# =========================================================

def apply_wavelet_denoising(
    data,
    wavelet='db4',
    level=4
):
    """
    Adaptive Wavelet Denoising.

    Threshold:
        sigma(detail_final) / 2

    Literatur ECG:
    - db4 sangat umum untuk QRS morphology.
    """

    data = sanitize_signal(data)

    out = np.zeros_like(data)

    for i in range(data.shape[1]):

        max_lvl = pywt.dwt_max_level(
            data.shape[0],
            pywt.Wavelet(wavelet).dec_len
        )

        safe_level = min(level, max_lvl)

        coeffs = pywt.wavedec(
            data[:, i],
            wavelet,
            level=safe_level
        )

        threshold = np.std(coeffs[-1]) / 2.0

        coeffs[1:] = [
            pywt.threshold(
                c,
                value=threshold,
                mode='soft'
            )
            for c in coeffs[1:]
        ]

        reconstructed = pywt.waverec(
            coeffs,
            wavelet
        )

        out[:, i] = reconstructed[:data.shape[0]]

    return sanitize_signal(out)


# =========================================================
# BASELINE WANDER REMOVAL
# =========================================================

def apply_median_baseline(
    data,
    kernel_size=51
):
    """
    Baseline Wander Correction.

    Median filter dipakai karena:
    - robust terhadap spike
    - menjaga QRS morphology
    """

    data = sanitize_signal(data)

    # median kernel wajib odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    out = np.zeros_like(data)

    for i in range(data.shape[1]):

        baseline = medfilt(
            data[:, i],
            kernel_size=kernel_size
        )

        out[:, i] = data[:, i] - baseline

    return sanitize_signal(out)


# =========================================================
# BANDPASS FILTER
# =========================================================

def apply_butter_bandpass(
    signal,
    fs,
    lowcut=0.5,
    highcut=45.0,
    order=4
):
    """
    Butterworth Bandpass Filter.

    ECG typical:
    - lowcut : 0.5 Hz
    - highcut: 45 Hz

    Proteksi Nyquist otomatis.
    """

    signal = sanitize_signal(signal)

    nyq = 0.5 * fs

    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.99)

    if low >= high:
        return signal

    try:

        b, a = butter(
            order,
            [low, high],
            btype='band'
        )

        filtered = filtfilt(
            b,
            a,
            signal,
            axis=0
        )

        return sanitize_signal(filtered)

    except Exception:

        return signal


# =========================================================
# NORMALIZATION
# =========================================================

def apply_zscore_clip(
    data,
    epsilon=1e-8,
    clip_min=DEFAULT_CLIP_MIN,
    clip_max=DEFAULT_CLIP_MAX
):
    """
    Z-score normalization per channel.

    Kemudian:
    clipping untuk mencegah outlier ekstrem.
    """

    data = sanitize_signal(data)

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    norm = (data - mean) / (std + epsilon)

    norm = np.clip(
        norm,
        clip_min,
        clip_max
    )

    return sanitize_signal(norm)


# =========================================================
# MAIN DSP PIPELINE
# =========================================================

def advanced_cleaning_pipeline_offline(
    raw_signal,
    src_fs,
    target_fs
):
    """
    PIPELINE:
    native + downsampling

    Cocok:
    - 500 Hz -> 250 Hz
    - 250 Hz -> 250 Hz
    """

    x = sanitize_signal(raw_signal)
    x = validate_signal_shape(x)

    # =====================================================
    # 1. Wavelet Denoising
    # =====================================================

    x = apply_wavelet_denoising(
        x,
        wavelet='db4',
        level=4
    )

    # =====================================================
    # 2. Baseline Correction
    # =====================================================

    kernel_size = 101 if src_fs >= 500 else 51

    x = apply_median_baseline(
        x,
        kernel_size=kernel_size
    )

    # =====================================================
    # 3. Bandpass
    # =====================================================

    highcut = 100.0 if src_fs >= 250 else 45.0

    x = apply_butter_bandpass(
        x,
        fs=src_fs,
        lowcut=0.5,
        highcut=highcut,
        order=4
    )

    # =====================================================
    # 4. Resample
    # =====================================================

    if src_fs != target_fs:

        x = apply_poly_resample(
            x,
            src_fs,
            target_fs
        )

    # =====================================================
    # 5. Normalize
    # =====================================================

    x = apply_zscore_clip(x)

    return sanitize_signal(x)


# =========================================================
# UPSAMPLING PIPELINE
# =========================================================

def advanced_cleaning_pipeline_upsampling(
    raw_signal,
    src_fs,
    target_fs
):
    """
    PIPELINE:
    khusus upsampling

    Contoh:
    - 100 Hz -> 250 Hz
    """

    x = sanitize_signal(raw_signal)
    x = validate_signal_shape(x)

    # =====================================================
    # 1. Upsampling terlebih dahulu
    # =====================================================

    x = apply_poly_resample(
        x,
        src_fs,
        target_fs
    )

    # =====================================================
    # 2. Wavelet
    # =====================================================

    x = apply_wavelet_denoising(
        x,
        wavelet='db4',
        level=4
    )

    # =====================================================
    # 3. Baseline
    # =====================================================

    x = apply_median_baseline(
        x,
        kernel_size=51
    )

    # =====================================================
    # 4. Bandpass
    # =====================================================

    x = apply_butter_bandpass(
        x,
        fs=target_fs,
        lowcut=0.5,
        highcut=45.0,
        order=4
    )

    # =====================================================
    # 5. Normalize
    # =====================================================

    x = apply_zscore_clip(x)

    return sanitize_signal(x)


# =========================================================
# SMART ROUTER
# =========================================================

def advanced_cleaning_pipeline(
    raw_signal,
    src_fs,
    target_fs=250.0
):
    """
    SMART ROUTER.

    Otomatis memilih:
    - upsampling pipeline
    - offline/native pipeline
    """

    raw_signal = sanitize_signal(raw_signal)
    raw_signal = validate_signal_shape(raw_signal)

    if src_fs < target_fs:

        return advanced_cleaning_pipeline_upsampling(
            raw_signal=raw_signal,
            src_fs=src_fs,
            target_fs=target_fs
        )

    else:

        return advanced_cleaning_pipeline_offline(
            raw_signal=raw_signal,
            src_fs=src_fs,
            target_fs=target_fs
        )


# =========================================================
# QUICK METADATA EXTRACTOR
# =========================================================

def extract_signal_statistics(signal):
    """
    Statistik ringan untuk audit manifest.
    """

    signal = sanitize_signal(signal)

    return {
        "min": float(np.min(signal)),
        "max": float(np.max(signal)),
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "shape": tuple(signal.shape)
    }