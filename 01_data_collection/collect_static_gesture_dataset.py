import cv2
import os
import time

def capture_dataset_auto(save_path, person_name, max_images=200, delay_ms=100, start_delay=5000):
    """
    Capture dataset gestur BISINDO otomatis dari webcam
    
    Parameters:
    -----------
    save_path : str
        Path folder tempat menyimpan dataset (contoh: "E:/Dataset.../Nic/SIAPA")
    person_name : str
        Nama orang/responden (contoh: "NICO")
    max_images : int
        Target jumlah gambar
    delay_ms : int
        Jeda antar capture gambar (dalam milidetik)
    start_delay : int
        Jeda awal sebelum mulai capture (dalam milidetik)
    """
    
    # Cek apakah folder sudah ada
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"📁 Folder baru dibuat: {save_path}")
    
    # Inisialisasi webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Tidak bisa membuka webcam")
        return
    
    print(f"▶ Persiapkan gestur di folder: {save_path}")
    print(f"Capture akan dimulai dalam {start_delay/1000:.1f} detik...")
    
    # Tampilkan countdown awal
    start_time = time.time()
    while (time.time() - start_time) * 1000 < start_delay:
        ret, frame = cap.read()
        if not ret:
            print("❌ Gagal membaca frame dari webcam")
            cap.release()
            return
        frame_resized = cv2.resize(frame, (640, 480))
        remaining = int(start_delay/1000 - (time.time() - start_time))
        cv2.putText(frame_resized, f"Mulai dalam {remaining} detik...",
                    (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        cv2.imshow("Capture Dataset", frame_resized)
        cv2.waitKey(1)
    
    # Cari nomor terakhir agar tidak overwrite file lama
    existing_files = [f for f in os.listdir(save_path) if f.startswith(person_name)]
    count_start = len(existing_files)
    
    count = 0
    print(f"▶ Mulai auto-capture: {max_images} gambar | Delay: {delay_ms} ms per gambar")
    print("Tekan 'q' jika ingin berhenti lebih awal")

    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            print("❌ Gagal membaca frame dari webcam")
            break

        frame_resized = cv2.resize(frame, (640, 480))
        cv2.putText(frame_resized, f"{person_name} | {count+1}/{max_images}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Capture Dataset", frame_resized)

        # Simpan gambar dengan nama berurutan
        filename = f"{person_name}_({count_start + count + 1}).jpg"
        cv2.imwrite(os.path.join(save_path, filename), frame)
        print(f"✅ Disimpan: {filename}")
        count += 1

        # Delay dalam ms
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"📂 Selesai: {count} gambar disimpan di {save_path}")


# ... (kode fungsi def capture_dataset_auto di atas biarkan saja) ...

# -------------------
# 📌 Contoh penggunaan
# -------------------

if __name__ == "__main__":
    # 1. Tentukan nama orang dan gestur
    NAMA_ORANG = "MIGOZ"
    NAMA_GESTUR = "TIDUR"
    
    # 2. Setup Path Otomatis
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # --- [YANG DIUBAH] ---
    # Simpan ke folder 'sample_data/static' (bukan data/static_raw)
    save_path = os.path.join(project_root, "sample_data", "static", NAMA_ORANG, NAMA_GESTUR)
    # ---------------------

    print(f"📂 Lokasi penyimpanan SAMPEL: {save_path}")

    # Jalankan fungsi capture
    capture_dataset_auto(
        save_path=save_path, 
        person_name=NAMA_ORANG, 
        max_images=10,
        delay_ms=100, 
        start_delay=5000
    )