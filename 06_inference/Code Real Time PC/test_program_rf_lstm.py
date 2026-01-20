import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
import time
import os
from collections import deque

# ==========================================
# ⚙️ KONFIGURASI PATH (Relative)
# ==========================================
# Setup Path Otomatis
current_dir = os.path.dirname(os.path.abspath(__file__)) # Folder 06_inference
project_root = os.path.dirname(current_dir)              # Root Project
models_dir = os.path.join(project_root, "models")        # Folder models

print(f"[INFO] Memuat model dari: {models_dir}")

# ------------------------------
# Load Models & Labels
# ------------------------------
# 1. Random Forest (Static)
rf_path = os.path.join(models_dir, "random_forest_bisindo_kcross.pkl")
# 2. LSTM (Dynamic)
lstm_path = os.path.join(models_dir, "best_model_lstm.h5") # Pastikan nama file sesuai output training
# 3. Labels
label_path = os.path.join(models_dir, "all_gestures_labels.npy")

if not os.path.exists(rf_path) or not os.path.exists(lstm_path):
    raise FileNotFoundError("❌ Model tidak ditemukan! Pastikan Anda sudah training dan file ada di folder 'models'.")

rf_model = joblib.load(rf_path)
lstm_model = tf.keras.models.load_model(lstm_path)
labels = np.load(label_path, allow_pickle=True).item() 

print("[INFO] Model & Label berhasil dimuat.")

# ------------------------------
# CONFIG
# ------------------------------
SEQ_LENGTH = 20
MIN_SEQ_FOR_LSTM = 12
MOTION_THRESHOLD = 0.0025
HOLD_FRAMES = 6
COOLDOWN_TIME = 1.5
HAND_GRACE_TIME = 1.0

# ------------------------------
# State & Buffers
# ------------------------------
sequence_buffer = deque(maxlen=SEQ_LENGTH)
motion_scores = []
hold_counter = 0
last_pred_time = 0.0
last_result_text = ""
last_result_color = (255,255,255)
last_result_source = ""
prev_landmarks = None

# untuk grace time
hand_present = False
grace_start_time = None

print(f"[INFO] Models loaded. SEQ_LENGTH={SEQ_LENGTH}, MIN_SEQ_FOR_LSTM={MIN_SEQ_FOR_LSTM}")

# ------------------------------
# Mediapipe
# ------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ------------------------------
# Helpers
# ------------------------------
def extract_landmarks(results):
    if not results.multi_hand_landmarks:
        return None
    row = []
    for hl in results.multi_hand_landmarks:
        for lm in hl.landmark:
            row.extend([lm.x, lm.y, lm.z])
    if len(results.multi_hand_landmarks) == 1:
        row.extend([0.0]*63)
    if len(row) < 126:
        row.extend([0.0]*(126-len(row)))
    elif len(row) > 126:
        row = row[:126]
    return np.array(row, dtype=np.float32)

def predict_rf(feat):
    probs = rf_model.predict_proba([feat])[0]
    cid = int(np.argmax(probs))
    return labels["static"][cid]

def predict_lstm(seq):
    X = np.expand_dims(np.array(seq, dtype=np.float32), axis=0)
    probs = lstm_model.predict(X, verbose=0)[0]
    cid = int(np.argmax(probs))
    return labels["dynamic"][cid]

def draw_text_bg(img, text, pos=(10,40), font_scale=1.0, color=(255,255,255)):
    x,y = pos
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (w,h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(img, (x-6,y-6), (x+w+6, y+h+6), (0,0,0), -1)
    cv2.putText(img, text, (x, y+h-6), font, font_scale, color, thickness, cv2.LINE_AA)

# ------------------------------
# Main loop
# ------------------------------
cap = cv2.VideoCapture(0)
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # draw hands
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hl, mp_hands.HAND_CONNECTIONS)

        # cooldown prediksi (hasil tetap ditampilkan)
        if time.time() - last_pred_time < COOLDOWN_TIME:
            if last_result_text:
                draw_text_bg(image, f"{last_result_text} ({last_result_source})", pos=(10,10), color=last_result_color)
                remaining = COOLDOWN_TIME - (time.time() - last_pred_time)
                draw_text_bg(image, f"Next in {remaining:.1f}s", pos=(10,60), font_scale=0.8, color=(200,200,200))
            cv2.imshow("Gesture Recognition", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # landmarks extraction
        lm = extract_landmarks(results)
        if lm is None:
            # reset jika tangan hilang
            sequence_buffer.clear()
            motion_scores.clear()
            hold_counter = 0
            prev_landmarks = None
            hand_present = False
            grace_start_time = None

            draw_text_bg(image, "Show your hand (waiting)...", pos=(10,10), color=(200,200,200))
            cv2.imshow("Gesture Recognition", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        else:
            # ada tangan
            if not hand_present:
                hand_present = True
                grace_start_time = time.time()

            # grace time saat tangan baru muncul
            if grace_start_time is not None and (time.time() - grace_start_time < HAND_GRACE_TIME):
                remaining = HAND_GRACE_TIME - (time.time() - grace_start_time)
                draw_text_bg(image, f"Stabilizing... {remaining:.1f}s", pos=(10,10), color=(0,200,200))
                cv2.imshow("Gesture Recognition", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

        # append frame ke buffer
        sequence_buffer.append(lm)

        # hitung motion
        if prev_landmarks is not None:
            motion = np.mean(np.abs(lm - prev_landmarks))
        else:
            motion = 0.0
        prev_landmarks = lm

        motion_scores.append(motion)
        if len(motion_scores) > SEQ_LENGTH:
            motion_scores.pop(0)

        draw_text_bg(image, f"Recording {len(sequence_buffer)}/{SEQ_LENGTH}", pos=(10,10), color=(200,200,200))

        if len(sequence_buffer) >= 2:
            avg_motion = float(np.mean(motion_scores)) if motion_scores else 0.0

            # dynamic
            if avg_motion > MOTION_THRESHOLD and len(sequence_buffer) >= MIN_SEQ_FOR_LSTM:
                result = predict_lstm(list(sequence_buffer))
                last_result_text = result
                last_result_source = "LSTM"
                last_result_color = (0,255,0)
                last_pred_time = time.time()
                sequence_buffer.clear()
                motion_scores.clear()
                prev_landmarks = None
                hold_counter = 0
                draw_text_bg(image, f"{last_result_text} (LSTM)", pos=(10,10), color=last_result_color)

            # static
            elif avg_motion <= MOTION_THRESHOLD:
                hold_counter += 1
                draw_text_bg(image, f"Holding... {hold_counter}/{HOLD_FRAMES}", pos=(10,60), color=(200,200,200))
                if hold_counter >= HOLD_FRAMES and len(sequence_buffer) >= 1:
                    result = predict_rf(sequence_buffer[-1])
                    last_result_text = result
                    last_result_source = "RF"
                    last_result_color = (0,0,255)
                    last_pred_time = time.time()
                    sequence_buffer.clear()
                    motion_scores.clear()
                    prev_landmarks = None
                    hold_counter = 0
                    draw_text_bg(image, f"{last_result_text} (RF)", pos=(10,10), color=last_result_color)

        cv2.imshow("Gesture Recognition", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()