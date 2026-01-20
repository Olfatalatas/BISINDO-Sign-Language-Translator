import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ===============================================
# 1. Konfigurasi Path
# ===============================================
# Setup Path Otomatis
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Input: Ambil dari folder sample_data/dynamic
DATASET_DIR = os.path.join(project_root, "sample_data", "dynamic")

# Output: Simpan ke folder models
SAVE_DIR = os.path.join(project_root, "models")
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"[INFO] Dataset Directory (Sample): {DATASET_DIR}")
print(f"[INFO] Save Directory: {SAVE_DIR}")

# Parameter Model
SEQ_LENGTH = 20
NUM_FOLDS = 5 # Untuk sampel mungkin akan warning jika data < 5, tapi biarkan saja standar
BATCH_SIZE = 16   
EPOCHS = 10       # KURANGI EPOCH! Untuk sampel di GitHub, 300 kelamaan. Cukup 10-20 buat tes.
DROPOUT_RATE = 0.3

# Buat folder simpan
os.makedirs(SAVE_DIR, exist_ok=True)

def normalize_sequence(seq, target_len=SEQ_LENGTH):
    if seq.shape[0] > target_len: return seq[:target_len]
    if seq.shape[0] < target_len:
        pad_width = target_len - seq.shape[0]
        padding = np.zeros((pad_width, seq.shape[1]))
        return np.vstack([seq, padding])
    return seq

X_all, y_all, gestures, class_counter = [], [], {}, 0

print("[INFO] Memuat dataset secara rekursif...")
for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        if file.endswith(".npz"):
            path = os.path.join(root, file)
            try:
                data = np.load(path, allow_pickle=True)
                X_sequences = data["X"]
                
                # Cek jika file kosong
                if len(X_sequences) == 0:
                    continue

                X_norm = np.array([normalize_sequence(seq) for seq in X_sequences])
                gesture_name = file.replace("_dataset.npz", "")
                
                if gesture_name not in gestures:
                    gestures[gesture_name] = class_counter
                    class_counter += 1
                
                y_label = gestures[gesture_name]
                X_all.append(X_norm)
                y_all.append(np.full(len(X_norm), y_label))
                # print(f"[INFO] Loaded {file}") 
            except Exception as e:
                print(f"[ERROR] Gagal load {file}: {e}")

if len(X_all) == 0:
    print("[CRITICAL] Tidak ada data ditemukan!")
    exit()

X = np.concatenate(X_all, axis=0)
y = np.concatenate(y_all, axis=0)
CLASS_NAMES = list(gestures.keys())
NUM_CLASSES = len(CLASS_NAMES)

print(f"[INFO] Total Data: {X.shape[0]} sequences, {NUM_CLASSES} kelas.")

# --- SPLIT 80% (train+val) dan 20% (test) ---
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ===============================================
# 2. Arsitektur Transformer (DIPERBAIKI)
# ===============================================
def build_transformer_model(input_shape, num_classes, d_model=64, num_heads=4, ff_dim=64, num_transformer_blocks=2, dropout=DROPOUT_RATE):
    inputs = keras.Input(shape=input_shape)
    x = layers.Dense(d_model, name="dense_projection")(inputs)
    
    # Positional Encoding
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    pos_embedding = keras.layers.Embedding(input_dim=input_shape[0], output_dim=d_model)(positions)
    x = x + pos_embedding
    
    # Transformer Blocks
    for _ in range(num_transformer_blocks):
        # Attention dengan Dropout variabel
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout
        )(query=x, value=x, key=x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # Feed Forward
        ffn_output = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model)
        ])(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

    # Classification Head
    x = layers.GlobalAveragePooling1D()(x)
    
    # --- PERBAIKAN: Menggunakan variabel dropout (0.3), bukan hardcode 0.2 ---
    x = layers.Dropout(dropout)(x) 
    # ------------------------------------------------------------------------
    
    x = layers.Dense(ff_dim, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    return keras.Model(inputs=inputs, outputs=outputs)

# ===============================================
# 3. K-Fold Cross-Validation (Setara LSTM)
# ===============================================
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
cv_scores = []
fold_no = 1

print("\n[INFO] Memulai 5-Fold Cross Validation...")
for train_index, val_index in skf.split(X_train_val, y_train_val):
    print(f"--- FOLD {fold_no}/{NUM_FOLDS} ---")
    
    X_train, X_val = X_train_val[train_index], X_train_val[val_index]
    y_train, y_val = y_train_val[train_index], y_train_val[val_index]
    
    INPUT_SHAPE = (X_train.shape[1], X_train.shape[2])
    
    # Build Model
    model = build_transformer_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, dropout=DROPOUT_RATE)
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
                  loss="sparse_categorical_crossentropy", 
                  metrics=["accuracy"])
    
    # Callbacks (Sama seperti LSTM: EarlyStopping + ReduceLR)
    callbacks_cv = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    # Train dengan Batch Size 16 & Epoch 300
    model.fit(X_train, y_train, 
              batch_size=BATCH_SIZE, 
              epochs=EPOCHS, 
              validation_data=(X_val, y_val), 
              callbacks=callbacks_cv,
              verbose=0) # verbose 0 agar rapi
    
    scores = model.evaluate(X_val, y_val, verbose=0)
    print(f"   Skor untuk fold {fold_no}: Akurasi {scores[1]*100:.2f}%")
    cv_scores.append(scores[1])
    fold_no += 1

print("\n--- Hasil Cross-Validation ---")
print(f"Akurasi Rata-rata: {np.mean(cv_scores)*100:.2f}% (+/- {np.std(cv_scores)*100:.2f}%)")

# ===============================================
# 4. Training Final & Evaluasi Akhir (Setara LSTM)
# ===============================================
print("\n[INFO] Melatih model final pada seluruh 80% data...")

INPUT_SHAPE = (X_train_val.shape[1], X_train_val.shape[2])
final_model = build_transformer_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, dropout=DROPOUT_RATE)

final_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])

# Callbacks Final (Sama seperti LSTM)
callbacks_final = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(os.path.join(SAVE_DIR, "transformer_best.keras"), save_best_only=True, monitor="val_accuracy"),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)
]

# Training Final (Gunakan validation_data=(X_test, y_test) seperti LSTM)
history = final_model.fit(
    X_train_val, y_train_val,
    batch_size=BATCH_SIZE,  # 16
    epochs=EPOCHS,          # 300
    validation_data=(X_test, y_test), # Konsisten dengan LSTM
    callbacks=callbacks_final,
    verbose=1
)

print("\n[INFO] Mengevaluasi model final pada 20% data test...")
test_loss, test_acc = final_model.evaluate(X_test, y_test)
print(f"\nAkurasi pada data uji final: {test_acc:.4f}")

# Simpan Model & Label
final_model.save(os.path.join(SAVE_DIR, "transformer_final.keras"))
np.save(os.path.join(SAVE_DIR, "labels.npy"), np.array(CLASS_NAMES))

# Prediksi
y_pred = np.argmax(final_model.predict(X_test), axis=1)

# Simpan Classification Report
report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
with open(os.path.join(SAVE_DIR, "classification_report.txt"), "w") as f: 
    f.write(report)

# ===============================================
# === CONFUSION MATRIX RESOLUSI TINGGI ===
# ===============================================
print("[INFO] Membuat Confusion Matrix Resolusi Tinggi...")
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(40, 40)) # Ukuran Besar
sns.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    cmap="Blues", 
    xticklabels=CLASS_NAMES, 
    yticklabels=CLASS_NAMES,
    ax=ax,
    annot_kws={"size": 6}
)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
ax.set_title("Confusion Matrix (Final Test)", fontsize=20)
ax.set_ylabel("True Label", fontsize=15)
ax.set_xlabel("Predicted Label", fontsize=15)

plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.svg"), bbox_inches='tight')
plt.close(fig)

# ===============================================
# === SIMPAN KURVA (SATU GAMBAR: ACCURACY & LOSS) ===
# ===============================================
print("[INFO] Menyimpan kurva training...")

plt.figure(figsize=(14, 6)) # Ukuran diperlebar agar muat dua grafik

# --- GRAFIK 1: ACCURACY (KIRI) ---
plt.subplot(1, 2, 1) # 1 Baris, 2 Kolom, Gambar ke-1
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Transformer Model Accuracy', fontsize=14)
plt.ylabel('Accuracy', fontsize=12)  # Sumbu Y: Accuracy
plt.xlabel('Epoch', fontsize=12)     # Sumbu X: Epoch
plt.legend(loc='lower right')
plt.grid(True, linestyle="--", alpha=0.6)

# --- GRAFIK 2: LOSS (KANAN) ---
plt.subplot(1, 2, 2) # 1 Baris, 2 Kolom, Gambar ke-2
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Transformer Model Loss', fontsize=14)
plt.ylabel('Loss', fontsize=12)      # Sumbu Y: Loss
plt.xlabel('Epoch', fontsize=12)     # Sumbu X: Epoch
plt.legend(loc='upper right')
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout() # Mengatur jarak otomatis agar tidak berdempetan
plt.savefig(os.path.join(SAVE_DIR, "transformer_training_curves.png"), dpi=300)
plt.close()

print(f"\n[INFO] Selesai! Gambar disimpan sebagai 'transformer_training_curves.png' di: {SAVE_DIR}")