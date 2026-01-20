import cv2
import mediapipe as mp
import numpy as np
import os

# ---------------------------
# Konfigurasi
# ---------------------------
PERSON_NAME = "MIGOZ"
GESTURE_NAME = "UTARA"

# Setup Path Otomatis
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# --- [YANG DIUBAH] ---
# 1. Simpan ke folder 'sample_data/dynamic'
SAVE_DIR = os.path.join(project_root, "sample_data", "dynamic", PERSON_NAME)

# 2. Kurangi jumlah sequence (misal cuma 5 untuk sampel)
SEQUENCE_LENGTH = 20    
TOTAL_SEQUENCES = 5

# Print info path
print(f"📂 Lokasi penyimpanan: {SAVE_DIR}")

# pastikan folder ada
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------
# Init MediaPipe
# ---------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

# ---------------------------
# Kamera
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka")

print(f"[INFO] Mulai otomatis rekam {TOTAL_SEQUENCES} sequence untuk gestur: {GESTURE_NAME}")
print("[INFO] Jumlah tangan akan divalidasi otomatis berdasarkan deteksi MediaPipe")
print("Tekan 'q' untuk berhenti.")

all_sequences = []
seq_count = 0
frame_buffer = []
last_status = None  # simpan status valid/invalid terakhir

def validate_sequence(seq, threshold=0.9):
    """Validasi otomatis: pastikan sequence punya cukup frame valid"""
    seq = np.array(seq).reshape(SEQUENCE_LENGTH, 2, 21, 3)
    valid_frames = 0
    hand_counts = []

    for frame in seq:
        hands_detected = 0
        for hand in frame:
            if np.any(hand != 0):
                hands_detected += 1
        hand_counts.append(hands_detected)
        if hands_detected > 0:
            valid_frames += 1

    frame_valid_ratio = valid_frames / SEQUENCE_LENGTH
    avg_hands = round(np.mean(hand_counts))

    # valid jika deteksi tangan stabil (>= threshold)
    return frame_valid_ratio >= threshold, frame_valid_ratio, avg_hands

while seq_count < TOTAL_SEQUENCES:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    landmarks_both = np.zeros((2, 21, 3))

    if results.multi_hand_landmarks:
        for h_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if h_idx > 1:
                break
            for i, lm in enumerate(hand_landmarks.landmark):
                landmarks_both[h_idx, i] = [lm.x, lm.y, lm.z]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    frame_buffer.append(landmarks_both.flatten())

    # kalau buffer sudah SEQUENCE_LENGTH frame → cek dan simpan
    if len(frame_buffer) == SEQUENCE_LENGTH:
        is_valid, ratio, avg_hands = validate_sequence(frame_buffer)

        if is_valid:
            all_sequences.append(np.array(frame_buffer))
            seq_count += 1
            last_status = (f"VALID ({avg_hands} hand)", (0, 255, 0))
            print(f"[SAVED] Sequence {seq_count}/{TOTAL_SEQUENCES} | Ratio={ratio:.2f}, Hands={avg_hands}")
        else:
            last_status = ("RETAKE", (0, 0, 255))
            print(f"[RETAKE] Sequence tidak valid | Ratio={ratio:.2f}")

        frame_buffer = []  # reset buffer

    # tampilkan info status
    if last_status:
        msg, color = last_status
        cv2.putText(frame, f"{msg}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    cv2.putText(frame, f"{GESTURE_NAME} Seq {seq_count}/{TOTAL_SEQUENCES}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    cv2.imshow("Collect Dataset", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()

# ---------------------------
# Simpan semua ke 1 file .npz
# ---------------------------
X = np.array(all_sequences)  # (N, SEQUENCE_LENGTH, 126)
y = np.array([0]*len(all_sequences))  # label gestur ini = 0

save_path = os.path.join(SAVE_DIR, f"{GESTURE_NAME}_dataset.npz")
np.savez_compressed(save_path, X=X, y=y, gesture=GESTURE_NAME)

print("\n[INFO] Dataset selesai!")
print(f"Total sequence: {X.shape[0]} | Shape data: {X.shape}")
print("Disimpan ke:", save_path)