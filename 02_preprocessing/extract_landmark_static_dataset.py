import os
import cv2
import mediapipe as mp
import pandas as pd

# Inisialisasi Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.6)

def extract_landmarks_from_image(image_path, do_flip=True):
    """
    Ekstrak landmark dari 1 gambar.
    Bisa di-flip horizontal tanpa mengubah file aslinya.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    # FLIP gambar jika ingin konsisten dengan kamera
    if do_flip:
        image = cv2.flip(image, 1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    data = []
    for hand_landmarks in results.multi_hand_landmarks:
        for lm in hand_landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])

    # padding kalau hanya 1 tangan
    if len(results.multi_hand_landmarks) == 1:
        data.extend([0.0] * 63)

    if len(data) != 126:
        return None

    return data

def process_dataset(root_dir, output_csv):
    """Loop semua folder → ekstrak landmark → simpan ke CSV"""
    dataset = []

    for person in os.listdir(root_dir):
        person_path = os.path.join(root_dir, person)
        if not os.path.isdir(person_path):
            continue

        for gesture in os.listdir(person_path):
            gesture_path = os.path.join(person_path, gesture)
            if not os.path.isdir(gesture_path):
                continue

            for filename in os.listdir(gesture_path):
                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                file_path = os.path.join(gesture_path, filename)
                landmarks = extract_landmarks_from_image(file_path)

                if landmarks is not None:
                    dataset.append(landmarks + [gesture])

    # Buat DataFrame
    num_landmarks = 42 * 3  # 42 titik (2 tangan), tiap titik ada x,y,z
    columns = [f"{axis}{i}" for i in range(42) for axis in ["x", "y", "z"]]
    columns.append("label")

    df = pd.DataFrame(dataset, columns=columns)

    # Simpan ke CSV
    df.to_csv(output_csv, index=False)
    print(f"✅ Ekstraksi selesai, hasil disimpan di: {output_csv}")


# -------------------
# 📌 Contoh penggunaan
# -------------------
if __name__ == "__main__":
    # Setup Path Otomatis
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    project_root = os.path.dirname(current_dir)

    # --- [YANG DIUBAH] ---
    # Input: Ambil gambar dari folder sample_data
    root_dir = os.path.join(project_root, "sample_data", "static")
    
    # Output: Simpan CSV di folder sample_data juga (agar ikut ke-upload ke GitHub)
    output_csv = os.path.join(project_root, "sample_data", "static_landmarks.csv")
    # ---------------------

    print(f"📂 Input Sample Images: {root_dir}")
    print(f"💾 Output Sample CSV  : {output_csv}")

    # Cek folder
    if os.path.exists(root_dir):
        process_dataset(root_dir, output_csv)
    else:
        print(f"❌ Error: Folder sampel tidak ditemukan di {root_dir}")