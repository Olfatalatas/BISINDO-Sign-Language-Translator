import os
import numpy as np
import tensorflow as tf

# ==========================================
# ⚙️ KONFIGURASI PATH
# ==========================================
# Setup Path Otomatis
current_dir = os.path.dirname(os.path.abspath(__file__)) # Folder 05_model_conversion
project_root = os.path.dirname(current_dir)              # Root Project

# Folder Models (Input & Output di satu tempat agar rapi)
models_dir = os.path.join(project_root, "models")
os.makedirs(models_dir, exist_ok=True)

# Input: Model .h5 hasil training LSTM
h5_model_path = os.path.join(models_dir, "best_model_lstm.h5")

# Output: Model .tflite siap pakai
tflite_out_path = os.path.join(models_dir, "model_lstm.tflite")

print(f"[INFO] Mencari Model di: {h5_model_path}")

# ------------------------------
# Helper untuk convert & save
# ------------------------------
def convert_and_save(converter, out_path):
    try:
        tflite_model = converter.convert()
        with open(out_path, "wb") as f:
            f.write(tflite_model)
        size_kb = os.path.getsize(out_path) / 1024
        print(f"✅ [SUKSES] Disimpan: {out_path} ({size_kb:.1f} KB)")
    except Exception as e:
        print(f"❌ [ERROR] Gagal konversi: {e}")

# ------------------------------
# 1. Load model & Convert
# ------------------------------
if os.path.exists(h5_model_path):
    print("[INFO] Loading model H5...")
    model = tf.keras.models.load_model(h5_model_path, compile=False)

    print("[INFO] Converting to TFLite (Float16)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    convert_and_save(converter, tflite_out_path)
    
    print("\n[INFO] Konversi LSTM Selesai!")
else:
    print(f"❌ Error: File model tidak ditemukan di {h5_model_path}")
    print("   Jalankan script 'train_lstm.py' terlebih dahulu (pakai sample mode juga boleh).")