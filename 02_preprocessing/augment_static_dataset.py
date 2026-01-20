import cv2
import os
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# ⚙️ KONFIGURASI
# ==========================================
# Setup Path Otomatis
current_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.dirname(current_dir)

# --- [YANG DIUBAH] ---
# Target folder: sample_data/static
DATASET_ROOT = os.path.join(project_root, "sample_data", "static")

# Multiplier: Untuk sampe
AUGMENT_MULTIPLIER = 3 
# ---------------------

print(f"📂 Target Augmentasi (Sample): {DATASET_ROOT}")

def safe_augment_image(image):
    h, w = image.shape[:2]
    
    # --- 1. ROTASI (-10 s/d 10 derajat) ---
    # Sudut diperkecil sedikit agar aman
    angle = random.uniform(-10, 10)
    center = (w // 2, h // 2)
    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, M_rot, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    
    # --- 2. SAFE ZOOM (Fokus Zoom Out) ---
    # Rentang scale: 0.85 (Jauh) sampai 1.05 (Sedikit Dekat)
    # Kebanyakan akan menjauh (Zoom Out) agar tangan tidak terpotong
    scale = random.uniform(0.85, 1.05)
    
    if scale < 1.0: 
        # === ZOOM OUT (Mengecil) ===
        # Gambar dikecilkan, lalu ditempel di tengah background hitam
        # Ini 100% AMAN, tangan tidak akan hilang
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        # Buat kanvas hitam seukuran asli
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Hitung posisi tengah
        y_off = (h - new_h) // 2
        x_off = (w - new_w) // 2
        
        # Tempel gambar kecil ke kanvas
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        image = canvas
        
    elif scale > 1.0:
        # === ZOOM IN (Membesar) ===
        # Dibatasi maksimal 5% agar tidak memotong jari
        new_h, new_w = int(h / scale), int(w / scale)
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        
        cropped = image[top:top+new_h, left:left+new_w]
        image = cv2.resize(cropped, (w, h))

    # --- 3. GESER SEDIKIT (Translation) ---
    # Geser maksimal 5% dari lebar/tinggi gambar
    tx = random.uniform(-0.05, 0.05) * w
    ty = random.uniform(-0.05, 0.05) * h
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, M_trans, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    return image

def process_file(file_info):
    root, filename = file_info
    
    if "_aug_" in filename:
        return 0

    img_path = os.path.join(root, filename)
    image = cv2.imread(img_path)
    
    if image is None: return 0
    
    count = 0
    base_name = os.path.splitext(filename)[0]

    for i in range(AUGMENT_MULTIPLIER):
        try:
            aug_img = safe_augment_image(image)
            
            new_filename = f"{base_name}_aug_{i}.jpg"
            save_path = os.path.join(root, new_filename)
            cv2.imwrite(save_path, aug_img)
            count += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    return count

def main():
    print(f"🚀 Memulai SAFE AUGMENTATION pada: {DATASET_ROOT}")
    
    all_files = []
    for root, dirs, files in os.walk(DATASET_ROOT):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_files.append((root, file))

    print(f"📄 Memproses {len(all_files)} file gambar...")

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(process_file, all_files)
        total = sum(results)

    print(f"\n✅ SELESAI! {total} gambar variasi aman dibuat.")

if __name__ == "__main__":
    main()