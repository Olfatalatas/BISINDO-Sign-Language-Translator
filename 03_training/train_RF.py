import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1. Load dataset
# =========================
# Setup Path Otomatis
current_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.dirname(current_dir)

# Input: Langsung ambil dari folder sample_data
data_path = os.path.join(project_root, "sample_data", "static_landmarks.csv")

# Output: Simpan model ke folder models
results_dir = os.path.join(project_root, "models")
os.makedirs(results_dir, exist_ok=True)

print(f"[INFO] Memuat dataset SAMPEL dari: {data_path}")
print(f"[INFO] Output model akan disimpan di: {results_dir}")

# Cek file
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ File CSV tidak ditemukan di {data_path}. Pastikan Anda sudah menjalankan extract_landmark pada folder sampel.")

# Baca CSV
df = pd.read_csv(data_path, low_memory=False)

# Pisahkan fitur dan label
X = df.drop("label", axis=1).values

# --- REVISI PENTING DI SINI ---
# Paksa semua label menjadi String agar angka '5' dan huruf 'A' bisa diproses LabelEncoder
y = df["label"].astype(str).values 
# ------------------------------

# Encode label
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

print(f"[INFO] Dataset dimuat. Total data: {X.shape[0]}, Fitur: {X.shape[1]}")
print(f"[INFO] Jumlah kelas: {len(class_names)}")

# =========================
# 2. Split dataset (80/20)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# =========================
# 3. Train Random Forest (train-test split)
# =========================
print("[INFO] Melatih model Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=None,
    random_state=42,
    n_jobs=-1  # Gunakan semua core CPU agar lebih cepat
)
rf.fit(X_train, y_train)

# =========================
# 4. Evaluasi train-test split
# =========================
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Akurasi test (80/20 split): {acc:.2f}\n")

cm = confusion_matrix(y_test, y_pred)

# =========================
# 5. K-Fold Cross Validation & Learning Curve
# =========================
k = 5  # jumlah fold
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

print(f"[INFO] Menjalankan {k}-Fold Cross Validation...")
cv_scores = cross_val_score(rf, X, y_encoded, cv=skf, n_jobs=-1)
print(f"✅ K-Fold CV mean accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

y_pred_cv = cross_val_predict(rf, X, y_encoded, cv=skf, n_jobs=-1)
cv_cm = confusion_matrix(y_encoded, y_pred_cv)

# --- PLOT LEARNING CURVE (Kurva Pembelajaran) ---
print("[INFO] Membuat Learning Curve...")
train_sizes_abs, train_scores, val_scores = learning_curve(
    rf, X, y_encoded, cv=skf, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring="accuracy"
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

# Direktori penyimpanan
results_dir = r"E:\Dataset Penelitian Bahasa Isyarat Olfat 2025\Tugas Akhir\Source Code\results train RF AUGMENTED 2"
os.makedirs(results_dir, exist_ok=True)

plt.figure(figsize=(10, 6))
plt.title("Learning Curve (Random Forest)")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.grid(True, linestyle="--", alpha=0.6)
plt.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="blue")
plt.fill_between(train_sizes_abs, val_scores_mean - val_scores_std,
                 val_scores_mean + val_scores_std, alpha=0.1, color="orange")
plt.plot(train_sizes_abs, train_scores_mean, 'o-', color="blue", label="Training Score")
plt.plot(train_sizes_abs, val_scores_mean, 'o-', color="orange", label="Cross Validation Score")
plt.legend(loc="best")
plt.savefig(os.path.join(results_dir, "learning_curve_rf.png"))
plt.close()

# =========================
# 6. Simpan Model
# =========================
model_path = os.path.join(results_dir, "random_forest_bisindo_kcross.pkl")
with open(model_path, "wb") as f:
    pickle.dump(rf, f)

if le is not None:
    le_path = os.path.join(results_dir, "label_encoder_kcross.pkl")
    with open(le_path, "wb") as f:
        pickle.dump(le, f)

print(f"\n✅ Model disimpan di: {model_path}")

# =========================
# 7. Simpan Laporan & Confusion Matrix (SUPER JELAS)
# =========================

# --- A. Simpan Classification Reports ---
report_path = os.path.join(results_dir, "classification_report_split_rf.txt")
with open(report_path, "w") as f:
    f.write("=== Classification Report (80/20 Split) ===\n")
    f.write(classification_report(y_test, y_pred, target_names=class_names))

cv_report_path = os.path.join(results_dir, "classification_report_cv_rf.txt")
with open(cv_report_path, "w") as f:
    f.write(f"=== Classification Report ({k}-Fold Cross Validation) ===\n")
    f.write(classification_report(y_encoded, y_pred_cv, target_names=class_names))

# --- B. Simpan Confusion Matrix 80/20 Split (UKURAN BESAR) ---
print("[INFO] Menyimpan Confusion Matrix 80/20 (High Res)...")
fig_cm, ax_cm = plt.subplots(figsize=(40, 40))  # <--- Ukuran Raksasa

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax_cm, annot_kws={"size": 6}) # Font angka kecil agar muat

ax_cm.set_xticklabels(ax_cm.get_xticklabels(), rotation=90, fontsize=8)
ax_cm.set_yticklabels(ax_cm.get_yticklabels(), rotation=0, fontsize=8)
ax_cm.set_xlabel("Predicted Label", fontsize=15)
ax_cm.set_ylabel("True Label", fontsize=15)
ax_cm.set_title("Confusion Matrix (80/20 Split)", fontsize=20)

plt.savefig(os.path.join(results_dir, "confusion_matrix_split_rf.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(results_dir, "confusion_matrix_split_rf.svg"), bbox_inches='tight')
plt.close(fig_cm)

# --- C. Simpan Confusion Matrix Cross-Validation (UKURAN BESAR) ---
print("[INFO] Menyimpan Confusion Matrix CV (High Res)...")
fig_cvcm, ax_cvcm = plt.subplots(figsize=(40, 40)) # <--- Ukuran Raksasa

sns.heatmap(cv_cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax_cvcm, annot_kws={"size": 6}) # Font angka kecil agar muat

ax_cvcm.set_xticklabels(ax_cvcm.get_xticklabels(), rotation=90, fontsize=8)
ax_cvcm.set_yticklabels(ax_cvcm.get_yticklabels(), rotation=0, fontsize=8)
ax_cvcm.set_xlabel("Predicted Label", fontsize=15)
ax_cvcm.set_ylabel("True Label", fontsize=15)
ax_cvcm.set_title(f"Confusion Matrix ({k}-Fold Cross Validation)", fontsize=20)

plt.savefig(os.path.join(results_dir, "confusion_matrix_cv_rf.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(results_dir, "confusion_matrix_cv_rf.svg"), bbox_inches='tight')
plt.close(fig_cvcm)

print(f"\n✅ Semua hasil (Learning Curve, Conf Matrix Besar, Laporan) tersimpan di: {results_dir}")