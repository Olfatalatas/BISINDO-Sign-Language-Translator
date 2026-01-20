import numpy as np
import joblib
import os

# ------------------------------
# Konfigurasi Path (UPDATED)
# ------------------------------
# Mendeteksi folder root project
current_dir = os.path.dirname(os.path.abspath(__file__)) # Posisi: 05_model_conversion
project_root = os.path.dirname(current_dir)              # Posisi: Root Project

# Folder tempat Model & Label disimpan (Biasanya hasil training masuk ke sini)
models_dir = os.path.join(project_root, "models")

# Path File Input (Sesuaikan nama file dengan hasil output training Anda nanti)
rf_encoder_path = os.path.join(models_dir, "label_encoder_kcross.pkl")
transformer_labels_path = os.path.join(models_dir, "gestures_labels.npy")

# Path File Output
output_path = os.path.join(models_dir, "all_gestures_labels.npy")

print(f"📂 Mengambil label dari folder: {models_dir}")

# ------------------------------
# Load Label RF (statis)
# ------------------------------
if not os.path.exists(rf_encoder_path):
    raise FileNotFoundError(f"File {rf_encoder_path} tidak ditemukan.")

rf_encoder = joblib.load(rf_encoder_path)
rf_labels = list(rf_encoder.classes_)
print("[INFO] RF Labels (static):", rf_labels)

# ------------------------------
# Load Label transformer (dinamis)
# ------------------------------
if not os.path.exists(transformer_labels_path):
    raise FileNotFoundError(f"File {transformer_labels_path} tidak ditemukan.")

transformer_labels = np.load(transformer_labels_path, allow_pickle=True).tolist()
print("[INFO] transformer Labels (dynamic):", transformer_labels)

# ------------------------------
# Gabungkan
# ------------------------------
all_labels = {
    "static": rf_labels,
    "dynamic": transformer_labels
}

# Simpan ke file npy
np.save(output_path, all_labels)
print(f"[INFO] Label gabungan disimpan ke {output_path}")

# ------------------------------
# Tes load kembali
# ------------------------------
loaded = np.load(output_path, allow_pickle=True).item()
print("\n[INFO] Cek hasil load kembali:")
print("Static:", loaded["static"])
print("Dynamic:", loaded["dynamic"])