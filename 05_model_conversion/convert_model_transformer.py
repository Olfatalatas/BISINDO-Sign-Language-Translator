import os
import numpy as np
import tensorflow as tf

# ==========================================
# ⚙️ KONFIGURASI PATH
# ==========================================
# Setup Path Otomatis
current_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.dirname(current_dir)

# Folder Models
models_dir = os.path.join(project_root, "models")
os.makedirs(models_dir, exist_ok=True)

# Input: Model .keras hasil training Transformer
keras_model_path = os.path.join(models_dir, "transformer_best.keras")

# Output: Model .tflite
tflite_model_path = os.path.join(models_dir, "model_transformer.tflite")

print(f"[INFO] Mencari Model di: {keras_model_path}")

# ------------------------------
# 1. Load model .keras
# ------------------------------
print("[INFO] Loading Keras model...")
model = tf.keras.models.load_model(keras_model_path, compile=False)
model.summary()

# ------------------------------
# 2. Convert to TFLite (Float16 quantization)
# ------------------------------
print("\n[INFO] Converting to TensorFlow Lite (float16 quantization)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Gunakan optimisasi default
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Targetkan tipe data float16 untuk bobot (lebih ringan & cepat di ARM)
converter.target_spec.supported_types = [tf.float16]
# Pastikan inference tetap float32 agar kompatibel
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

# Konversi dan simpan
tflite_model = converter.convert()
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

size_kb = os.path.getsize(tflite_model_path) / 1024
print(f"[OK] Model berhasil disimpan: {tflite_model_path} ({size_kb:.1f} KB)")

# ------------------------------
# 3. (Opsional) Cek input/output signature
# ------------------------------
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\n[INFO] Input Tensor:", input_details)
print("[INFO] Output Tensor:", output_details)
print("\n[INFO] Konversi selesai. Model siap digunakan di Raspberry Pi.")