import os
import sys

# Menambahkan root project ke sys.path agar modul 'src' dapat diimpor
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import tensorflow as tf
from src.models.model_factory import build_dynamic_cnn
from src.config.experiment_configs import Config

def main():
    print("=== Model Architecture Plotter ===")
    
    # 1. Konfigurasi default dari experiment_configs.py
    filters = Config.FILTER_SPACES["Medium"]
    kernels = Config.KERNEL_SPACES["Balanced"]
    dilations = Config.DILATION_SPACES["Progressive_Dilation"]
    
    # Mengambil mode temporal pertama (secara default "Pure_CNN")
    temporal_mode = Config.TEMPORAL_MODELS[0] if Config.TEMPORAL_MODELS else "Pure_CNN"
    use_separable = Config.USE_SEPARABLE_CONV
    stochastic_depth_rate = Config.STOCHASTIC_DEPTH_RATE
    
    print(f"Konfigurasi Model:")
    print(f" - Filters: {filters}")
    print(f" - Kernels: {kernels}")
    print(f" - Dilations: {dilations}")
    print(f" - Temporal Mode: {temporal_mode}")
    print(f" - Use Separable Conv: {use_separable}")
    print(f" - Stochastic Depth Rate: {stochastic_depth_rate}")
    
    # 2. Build Model
    print("\nMembangun model...")
    model = build_dynamic_cnn(
        filters=filters,
        kernels=kernels,
        dilations=dilations,
        temporal_mode=temporal_mode,
        use_separable=use_separable,
        stochastic_depth_rate=stochastic_depth_rate,
        scheme="multilabel"
    )
    
    # Menampilkan ringkasan model di terminal
    model.summary()
    
    # 3. Plot Model
    output_filename = "model.png"
    print(f"\nMembuat gambar arsitektur model ke '{output_filename}'...")
    try:
        tf.keras.utils.plot_model(
            model,
            to_file=output_filename,
            show_shapes=True,
            show_layer_names=True,
            expand_nested=True,
            dpi=96
        )
        print(f"Sukses! Gambar arsitektur disimpan sebagai: {os.path.abspath(output_filename)}")
    except Exception as e:
        print(f"\nGagal membuat gambar arsitektur model: {e}")
        print("\nCatatan: Fungsi 'plot_model' membutuhkan library 'pydot' dan 'graphviz'.")
        print("Silakan jalankan perintah berikut untuk menginstalnya:")
        print("  pip install pydot graphviz")
        print("\nSerta pastikan aplikasi Graphviz terinstall di sistem Anda dan foldernya masuk ke PATH sistem:")
        print("  Windows (via winget): winget install -e --id Graphviz.Graphviz")

if __name__ == "__main__":
    main()
