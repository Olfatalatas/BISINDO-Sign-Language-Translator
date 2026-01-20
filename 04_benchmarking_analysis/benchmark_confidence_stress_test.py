import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# ⚙️ KONFIGURASI
# ==========================================
# Setup Path Otomatis
current_dir = os.path.dirname(os.path.abspath(__file__)) # Folder 04_benchmarking
project_root = os.path.dirname(current_dir)              # Root Project

# 1. Dataset: Ambil dari folder sample_data
DATASET_PATH = os.path.join(project_root, "sample_data", "dynamic")

# 2. Model: Ambil dari folder models (asumsi model sudah ada di sana)
PATH_LSTM = os.path.join(project_root, "models", "best_model_lstm.h5")
PATH_TRANSFORMER = os.path.join(project_root, "models", "transformer_best.keras")

# 3. Output Laporan: Simpan di folder benchmark ini
SAVE_REPORT_DIR = os.path.join(current_dir, "results_comparison")
os.makedirs(SAVE_REPORT_DIR, exist_ok=True)

SEQ_LENGTH = 20

print(f"[INFO] Mode Benchmark: SAMPLE DATA")
print(f"[INFO] Dataset Path : {DATASET_PATH}")
print(f"[INFO] Model LSTM   : {PATH_LSTM}")
print(f"[INFO] Model TF     : {PATH_TRANSFORMER}")

# ==========================================
# FUNGSI LOAD DATA
# ==========================================
def load_data_for_test():
    print("⏳ Memuat dataset...")
    X_all, y_all = [], []
    gestures = {}
    class_counter = 0
    
    def normalize_sequence(seq):
        if seq.shape[0] > SEQ_LENGTH: return seq[:SEQ_LENGTH]
        pad = np.zeros((SEQ_LENGTH - seq.shape[0], seq.shape[1]))
        return np.vstack([seq, pad])

    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if file.endswith(".npz"):
                path = os.path.join(root, file)
                try:
                    data = np.load(path, allow_pickle=True)
                    X_norm = np.array([normalize_sequence(s) for s in data["X"]])
                    g_name = file.replace("_dataset.npz", "")
                    if g_name not in gestures:
                        gestures[g_name] = class_counter
                        class_counter += 1
                    X_all.append(X_norm)
                    y_all.append(np.full(len(X_norm), gestures[g_name]))
                except: pass
    
    if not X_all:
        print("❌ Dataset tidak ditemukan!")
        exit()

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    
    # Ambil data test saja (20%)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_test, y_test

def get_avg_confidence(model, X, y):
    """Menghitung rata-rata probabilitas pada prediksi yang BENAR"""
    probs = model.predict(X, verbose=0)
    preds = np.argmax(probs, axis=1)
    correct_indices = np.where(preds == y)[0]
    if len(correct_indices) == 0: return 0.0
    correct_probs = np.max(probs[correct_indices], axis=1)
    return np.mean(correct_probs) * 100

def robustness_test(model, X, y, noise_level):
    """Menambah noise (gangguan) ke data dan cek akurasi"""
    noise = np.random.normal(0, noise_level, X.shape)
    X_noisy = X + noise
    probs = model.predict(X_noisy, verbose=0)
    preds = np.argmax(probs, axis=1)
    return accuracy_score(y, preds) * 100

# ==========================================
# EKSEKUSI & PENYIMPANAN
# ==========================================
X_test, y_test = load_data_for_test()
print(f"✅ Data Test Loaded: {X_test.shape}")

print("\n--- Memuat Model ---")
lstm_model = tf.keras.models.load_model(PATH_LSTM, compile=False)
trans_model = tf.keras.models.load_model(PATH_TRANSFORMER, compile=False)
print("✅ Model Loaded")

# Siapkan file laporan
# Siapkan file laporan
report_file = os.path.join(SAVE_REPORT_DIR, "hasil_analisis_model.txt")
f = open(report_file, "w", encoding="utf-8")  # <--- TAMBAHKAN INI

def log_print(text):
    """Cetak ke layar DAN simpan ke file"""
    print(text)
    f.write(text + "\n")

log_print("="*60)
log_print("LAPORAN PERBANDINGAN MODEL: LSTM vs TRANSFORMER")
log_print("="*60 + "\n")

# 1. Analisis Keyakinan (Confidence)
log_print("📊 1. ANALISIS KEYAKINAN (CONFIDENCE SCORE)")
conf_lstm = get_avg_confidence(lstm_model, X_test, y_test)
conf_trans = get_avg_confidence(trans_model, X_test, y_test)

log_print(f"   - LSTM Avg Confidence       : {conf_lstm:.2f}%")
log_print(f"   - Transformer Avg Confidence: {conf_trans:.2f}%")

if conf_trans > conf_lstm:
    diff = conf_trans - conf_lstm
    log_print(f"   👉 KESIMPULAN: Transformer lebih 'YAKIN' sebesar +{diff:.2f}%")
else:
    diff = conf_lstm - conf_trans
    log_print(f"   👉 KESIMPULAN: LSTM lebih 'YAKIN' sebesar +{diff:.2f}%")

# 2. Stress Test (Noise Robustness)
log_print("\n🌪️ 2. STRESS TEST (KETAHANAN TERHADAP NOISE)")
log_print("   (Mensimulasikan kamera buram atau tangan gemetar)")
log_print("-" * 65)
log_print(f"{'Noise Level':<15} | {'Akurasi LSTM':<15} | {'Akurasi Transformer':<20} | {'Selisih':<10}")
log_print("-" * 65)

noise_levels = [0.00, 0.01, 0.02, 0.03, 0.05, 0.10]
results_lstm = []
results_trans = []

for noise in noise_levels:
    acc_lstm = robustness_test(lstm_model, X_test, y_test, noise)
    acc_trans = robustness_test(trans_model, X_test, y_test, noise)
    diff = acc_trans - acc_lstm
    
    results_lstm.append(acc_lstm)
    results_trans.append(acc_trans)
    
    log_print(f"{noise:<15} | {acc_lstm:.2f}%{'':<8} | {acc_trans:.2f}%{'':<13} | {diff:+.2f}%")

log_print("-" * 65)

# Visualisasi Grafik Stress Test
plt.figure(figsize=(10, 6))
plt.plot(noise_levels, results_lstm, marker='o', linestyle='--', label='LSTM', color='blue')
plt.plot(noise_levels, results_trans, marker='s', linestyle='-', label='Transformer', color='orange')
plt.title("Noise Robustness Comparison")
plt.xlabel("Noise Level (Tingkat Gangguan)")
plt.ylabel("Akurasi (%)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Simpan Grafik
chart_path = os.path.join(SAVE_REPORT_DIR, "grafik_noise_robustness.png")
plt.savefig(chart_path)
log_print(f"\n✅ Grafik disimpan di: {chart_path}")

f.close()
print(f"\n✅ Laporan teks disimpan di: {report_file}")