import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ------------------------------
# Konfigurasi
# ------------------------------
# Setup Path Otomatis
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Input: Folder sampel dinamis
dataset_path = os.path.join(project_root, "sample_data", "dynamic")

# Output: Folder models
RESULTS_DIR = os.path.join(project_root, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"[INFO] Dataset Path (Sample): {dataset_path}")
print(f"[INFO] Output Path: {RESULTS_DIR}")

SEQ_LENGTH = 20
FEATURES = 126
K_FOLDS = 5
# Saran: Kurangi Epochs di bagian model.fit nanti jika hanya untuk tes sampel (misal jadi 20)

# ------------------------------
# Load dataset
# ------------------------------
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

print("[INFO] Memuat dataset...")
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".npz"):
            path = os.path.join(root, file)
            data = np.load(path, allow_pickle=True)
            X, y = data["X"], data["y"]
            X_norm = np.array([normalize_sequence(seq) for seq in X])
            gesture_name = file.replace("_dataset.npz", "")
            if gesture_name not in gestures:
                gestures[gesture_name] = class_counter
                class_counter += 1
            y_label = gestures[gesture_name]
            X_all.append(X_norm)
            y_all.append(np.full(len(X_norm), y_label))
            print(f"[INFO] Loaded {file} from {os.path.basename(root)}: {X_norm.shape}")

X_all = np.concatenate(X_all, axis=0)
y_all = np.concatenate(y_all, axis=0)
gesture_list = list(gestures.keys())
num_classes = len(gesture_list)
y_cat_all = to_categorical(y_all, num_classes=num_classes)

print(f"[INFO] Total Data: {X_all.shape}, Total Kelas: {num_classes}")

# ------------------------------
# Split dataset 80% train+val / 20% test
# ------------------------------
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_all, y_cat_all, test_size=0.2, stratify=y_all, random_state=42
)

# ------------------------------
# K-Fold Cross Validation pada train+val
# ------------------------------
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
fold_no = 1
histories = []

for train_idx, val_idx in kf.split(X_train_val):
    print(f"\n[INFO] Training fold {fold_no}/{K_FOLDS}")
    
    X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
    y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
    
    # Dropout di sini Anda set 0.3 pada kode sebelumnya, 
    # Anda bisa ubah ke 0.6 jika ingin kembali ke konfigurasi awal
    model = models.Sequential([
        layers.Input(shape=(SEQ_LENGTH, FEATURES)),
        layers.LSTM(64, return_sequences=False, unroll=True),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3), 
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint(f"{RESULTS_DIR}/best_model_fold{fold_no}.h5", monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=300,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    histories.append(history)
    
    # --- BAGIAN SIMPAN REPORT & MATRIX PER FOLD DIHAPUS ---
    
    print(f"[INFO] Fold {fold_no} selesai.")
    fold_no += 1

# ------------------------------
# Train final model pada seluruh train+val
# ------------------------------
print("\n[INFO] Melatih model final pada seluruh data train+val...")
# Pastikan nilai Dropout konsisten dengan yang Anda inginkan (misal 0.3 atau 0.6)
model_final = models.Sequential([
    layers.Input(shape=(SEQ_LENGTH, FEATURES)),
    layers.LSTM(64, return_sequences=False, unroll=True),
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.6),
    layers.Dense(num_classes, activation='softmax')
])
model_final.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

callbacks_final = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ModelCheckpoint(f"{RESULTS_DIR}/best_model_lstm.h5", monitor="val_loss", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)
]

history_final = model_final.fit(
    X_train_val, y_train_val,
    validation_data=(X_test, y_test),
    epochs=300,
    batch_size=16,
    callbacks=callbacks_final
)

# ------------------------------
# Save final model & gestures
# ------------------------------
model_final.save(os.path.join(RESULTS_DIR, "model_lstm.h5"))
np.save(os.path.join(RESULTS_DIR, "gestures_labels.npy"), gesture_list)

# ------------------------------
# Plot Kurva Akurasi & Loss
# ------------------------------
plt.figure(figsize=(12, 5))

# Kurva Akurasi
plt.subplot(1, 2, 1)
plt.plot(history_final.history["accuracy"], label="Train Accuracy", color='blue')
plt.plot(history_final.history["val_accuracy"], label="Validation Accuracy", color='orange')
plt.title("LSTM Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Kurva Loss
plt.subplot(1, 2, 2)
plt.plot(history_final.history["loss"], label="Train Loss", color='blue')
plt.plot(history_final.history["val_loss"], label="Validation Loss", color='orange')
plt.title("LSTM Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "training_curves_lstm.png"))
plt.show()

print(f"[INFO] Kurva akurasi dan loss disimpan di {RESULTS_DIR}/training_curves_lstm.png")

# ------------------------------
# Evaluasi akhir pada test set (TETAP DISIMPAN)
# ------------------------------
print("[INFO] Evaluasi akhir pada Test Set...")
y_test_true = np.argmax(y_test, axis=1)
y_test_pred = np.argmax(model_final.predict(X_test, verbose=0), axis=1)

# --- CONFUSION MATRIX FINAL (TEST SET) ---
cm_test = confusion_matrix(y_test_true, y_test_pred, labels=range(num_classes))

# Buat figure besar
fig, ax = plt.subplots(figsize=(40, 40))
sns.heatmap(
    cm_test, 
    annot=True, 
    fmt="d", 
    cmap="Blues", 
    xticklabels=gesture_list, 
    yticklabels=gesture_list,
    ax=ax,
    annot_kws={"size": 6}
)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
ax.set_title("Confusion Matrix (Test Set)", fontsize=20)
ax.set_ylabel("True Label", fontsize=15)
ax.set_xlabel("Predicted Label", fontsize=15)

# Simpan PNG & SVG
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_test_lstm_cross.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_test_lstm_cross.svg"), bbox_inches='tight')
plt.close(fig)

report_test = classification_report(y_test_true, y_test_pred, target_names=gesture_list, digits=4)
with open(os.path.join(RESULTS_DIR, "classification_report_test_lstm_cross.txt"), "w") as f:
    f.write(report_test)

print(f"[INFO] Evaluasi test set selesai. Confusion matrix & classification report disimpan di {RESULTS_DIR}")