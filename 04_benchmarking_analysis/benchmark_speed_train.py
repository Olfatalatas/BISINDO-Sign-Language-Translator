import time
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os

# ==========================================
# ⚙️ KONFIGURASI BENCHMARK
# ==========================================
# Setup Path Otomatis
current_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.dirname(current_dir)

# 1. Dataset: Ambil dari folder sample_data
DATASET_PATH = os.path.join(project_root, "sample_data", "dynamic")

# 2. Output Laporan
SAVE_DIR = os.path.join(current_dir, "results_benchmark")
os.makedirs(SAVE_DIR, exist_ok=True)

# 3. Setting Training (Ringan untuk Sampel)
EPOCHS = 5          # Cukup 5 epoch untuk tes kecepatan
BATCH_SIZE = 4      # Perkecil batch size karena data sampel sedikit
SEQ_LENGTH = 20
FEATURES = 126
DROPOUT_RATE = 0.3 

print(f"[INFO] Mode Speed Test: SAMPLE DATA")
print(f"[INFO] Dataset Source : {DATASET_PATH}")
print(f"[INFO] Save Report to : {SAVE_DIR}")

# ==========================================
# 1. LOAD DATASET ASLI
# ==========================================
print("⏳ Memuat dataset asli...")
X_all, y_all = [], []
gestures = {}
class_counter = 0

def normalize_sequence(seq, target_len=SEQ_LENGTH):
    if seq.shape[0] < target_len:
        pad_width = target_len - seq.shape[0]
        padding = np.zeros((pad_width, seq.shape[1]))
        return np.vstack([seq, padding])
    elif seq.shape[0] > target_len:
        return seq[:target_len]
    else:
        return seq

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".npz"):
            path = os.path.join(root, file)
            try:
                data = np.load(path, allow_pickle=True)
                X, y = data["X"], data["y"]
                if len(X) == 0: continue

                X_norm = np.array([normalize_sequence(seq) for seq in X])
                gesture_name = file.replace("_dataset.npz", "")
                
                if gesture_name not in gestures:
                    gestures[gesture_name] = class_counter
                    class_counter += 1
                
                y_label = gestures[gesture_name]
                X_all.append(X_norm)
                y_all.append(np.full(len(X_norm), y_label))
            except Exception as e: pass

if not X_all:
    print("❌ Error: Dataset tidak ditemukan!")
    exit()

X = np.concatenate(X_all, axis=0)
y = np.concatenate(y_all, axis=0)
num_classes = len(gestures)
y_cat = to_categorical(y, num_classes=num_classes)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

print(f"✅ Dataset Siap. Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"   Total Kelas: {num_classes}")
print("-" * 50)

# ==========================================
# 2. DEFINISI MODEL (Setara)
# ==========================================
def build_lstm():
    model = keras.Sequential([
        layers.Input(shape=(SEQ_LENGTH, FEATURES)),
        layers.LSTM(64, return_sequences=False, unroll=True),
        layers.Dense(32, activation='relu'),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_transformer():
    inputs = keras.Input(shape=(SEQ_LENGTH, FEATURES))
    d_model = 64
    x = layers.Dense(d_model)(inputs)
    positions = tf.range(start=0, limit=SEQ_LENGTH, delta=1)
    x = x + layers.Embedding(input_dim=SEQ_LENGTH, output_dim=d_model)(positions)
    
    for _ in range(2):
        attn_out = layers.MultiHeadAttention(num_heads=4, key_dim=d_model, dropout=DROPOUT_RATE)(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)
        ffn_out = keras.Sequential([layers.Dense(64, activation="relu"), layers.Dense(d_model)])(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_out)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Callback Pencatat Waktu
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

# ==========================================
# 3. FUNGSI EKSEKUSI BENCHMARK
# ==========================================
def run_benchmark(name, model_builder):
    print(f"\n🚀 Memulai Training {name} ({EPOCHS} Epochs)...")
    
    tf.keras.backend.clear_session()
    model = model_builder()
    time_callback = TimeHistory()
    
    # Start Timer
    start_total = time.time()
    
    # Training tanpa EarlyStopping (Agar pas 20 epoch)
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[time_callback]
    )
    
    end_total = time.time()
    
    # Hitung Statistik
    avg_epoch = np.mean(time_callback.times)
    
    # PROYEKSI KE 5 EPOCH
    projected_total = avg_epoch * 5
    
    return {
        "name": name,
        "avg_epoch": avg_epoch,
        "projected_total": projected_total
    }

# ==========================================
# 4. JALANKAN PROGRAM UTAMA
# ==========================================
res_lstm = run_benchmark("LSTM", build_lstm)
res_trans = run_benchmark("Transformer", build_transformer)

# ==========================================
# 5. GENERATE LAPORAN & SIMPAN
# ==========================================
report_path = os.path.join(SAVE_DIR, "laporan_kecepatan_estimasi.txt")

def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

with open(report_path, "w", encoding="utf-8") as f:
    f.write("="*60 + "\n")
    f.write(f"LAPORAN BENCHMARK KECEPATAN (SAMPEL {EPOCHS} EPOCHS)\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"Tanggal           : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Konfigurasi       : Batch {BATCH_SIZE}, Dropout {DROPOUT_RATE}\n")
    f.write(f"Catatan           : Waktu total diestimasi untuk 5 Epochs.\n\n")
    
    # --- Tabel Hasil ---
    header = f"{'Model':<15} | {'Rata-rata/Epoch':<18} | {'ESTIMASI Total (5)':<25}"
    f.write("-" * len(header) + "\n")
    f.write(header + "\n")
    f.write("-" * len(header) + "\n")
    
    for res in [res_lstm, res_trans]:
        line = f"{res['name']:<15} | {res['avg_epoch']:.4f} sec{'':<8} | {format_time(res['projected_total'])}"
        f.write(line + "\n")
    
    f.write("-" * len(header) + "\n\n")
    
    # --- Analisis Perbandingan ---
    diff_epoch = abs(res_lstm['avg_epoch'] - res_trans['avg_epoch'])
    winner = "LSTM" if res_lstm['avg_epoch'] < res_trans['avg_epoch'] else "Transformer"
    
    f.write("ANALISIS:\n")
    f.write(f"1. Model tercepat: {winner}\n")
    f.write(f"2. Selisih per Epoch: {diff_epoch:.4f} detik\n")
    
    ratio = res_trans['avg_epoch'] / res_lstm['avg_epoch']
    if ratio > 1:
        f.write(f"3. Kesimpulan: Transformer {ratio:.2f}x lebih LAMBAT daripada LSTM.\n")
    else:
        f.write(f"3. Kesimpulan: Transformer {1/ratio:.2f}x lebih CEPAT daripada LSTM.\n")

print(f"\n✅ Selesai! Laporan estimasi tersimpan di:\n{report_path}")