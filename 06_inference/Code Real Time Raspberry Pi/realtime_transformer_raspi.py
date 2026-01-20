import sys
# sys.path.append('/usr/lib/python3/dist-packages') 

import cv2
import numpy as np
import mediapipe as mp
import joblib
import time
import os
import tensorflow as tf
from picamera2 import Picamera2
from collections import deque

# ==========================================
# ⚙️ KONFIGURASI PATH (Relative)
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.dirname(current_dir)              
models_dir = os.path.join(project_root, "models")

print(f"[INFO] Memuat model Transformer TFLite dari: {models_dir}")

# ------------------------------
# Load Models
# ------------------------------
# 1. Random Forest
rf_path = os.path.join(models_dir, "random_forest_bisindo_kcross.pkl")

# 2. Transformer TFLite
tflite_path = os.path.join(models_dir, "model_transformer.tflite") # Nama file hasil convert step 5

# 3. Labels
label_path = os.path.join(models_dir, "all_gestures_labels.npy")

# Eksekusi Load
rf_model = joblib.load(rf_path)

interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = np.load(label_path, allow_pickle=True).item()

# ------------------------------
# CONFIG
# ------------------------------
SEQ_LENGTH = 20
MIN_SEQ_FOR_TRANSFORMER = 12
MOTION_THRESHOLD = 0.002
HOLD_FRAMES = 6
COOLDOWN_TIME = 1.5
HAND_GRACE_TIME = 1.0

print(f"[INFO] System Ready. Transformer TFLite Input: {input_details[0]['shape']}")

# =====================================================
# 3. STATE & BUFFER
# =====================================================
sequence_buffer = deque(maxlen=SEQ_LENGTH)
motion_scores = []
hold_counter = 0
last_pred_time = 0.0
last_result_text = ""
last_result_color = (255, 255, 255)
last_result_source = ""
prev_landmarks = None

hand_present = False
grace_start_time = None

# =====================================================
# 4. MEDIAPIPE
# =====================================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65
)
mp_draw = mp.solutions.drawing_utils

# =====================================================
# 5. HELPER FUNCTIONS
# =====================================================
def extract_landmarks(results):
    if not results.multi_hand_landmarks:
        return None
    row = []
    for hl in results.multi_hand_landmarks:
        for lm in hl.landmark:
            row.extend([lm.x, lm.y, lm.z])
    if len(results.multi_hand_landmarks) == 1:
        row.extend([0.0] * 63)
    if len(row) < 126:
        row.extend([0.0] * (126 - len(row)))
    elif len(row) > 126:
        row = row[:126]
    return np.array(row, dtype=np.float32)

def predict_rf(feat):
    probs = rf_model.predict_proba([feat])[0]
    cid = int(np.argmax(probs))
    return labels["static"][cid]

def predict_transformer_tflite(seq):
    seq = np.array(seq, dtype=np.float32)

    # padding jika panjang sequence < 20
    if len(seq) < SEQ_LENGTH:
        last_frame = seq[-1] if len(seq) > 0 else np.zeros((126,), dtype=np.float32)
        pad_len = SEQ_LENGTH - len(seq)
        pad_frames = np.repeat(last_frame[np.newaxis, :], pad_len, axis=0)
        seq = np.concatenate([seq, pad_frames], axis=0)

    X = np.expand_dims(seq, axis=0)  # (1, 20, 126)

    interpreter.set_tensor(input_details[0]['index'], X)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]['index'])[0]
    cid = int(np.argmax(probs))
    return labels["dynamic"][cid]

def draw_text_bg(img, text, pos=(10,40), font_scale=1.0, color=(255,255,255)):
    x, y = pos
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(img, (x-6, y-6), (x+w+6, y+h+6), (0,0,0), -1)
    cv2.putText(img, text, (x, y+h-6), font, font_scale, color, thickness, cv2.LINE_AA)

# =====================================================
# 6. INISIALISASI KAMERA
# =====================================================
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (640, 480)})
picam2.configure(config)
picam2.start()

print("[INFO] Sistem siap. Tekan 'q' untuk keluar.")

# =====================================================
# 7. LOOP UTAMA
# =====================================================
try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.flip(frame, 1)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hl, mp_hands.HAND_CONNECTIONS)

        # cooldown prediksi
        if time.time() - last_pred_time < COOLDOWN_TIME:
            if last_result_text:
                draw_text_bg(image, f"{last_result_text} ({last_result_source})", pos=(10,10), color=last_result_color)
                remaining = COOLDOWN_TIME - (time.time() - last_pred_time)
                draw_text_bg(image, f"Next in {remaining:.1f}s", pos=(10,60), font_scale=0.8, color=(200,200,200))
            cv2.imshow("Gesture Recognition", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        lm = extract_landmarks(results)
        if lm is None:
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
            if not hand_present:
                hand_present = True
                grace_start_time = time.time()

            if grace_start_time is not None and (time.time() - grace_start_time < HAND_GRACE_TIME):
                remaining = HAND_GRACE_TIME - (time.time() - grace_start_time)
                draw_text_bg(image, f"Stabilizing... {remaining:.1f}s", pos=(10,10), color=(0,200,200))
                cv2.imshow("Gesture Recognition", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

        # ------------------------------
        # Proses sequence & motion
        # ------------------------------
        sequence_buffer.append(lm)
        motion = np.mean(np.abs(lm - prev_landmarks)) if prev_landmarks is not None else 0.0
        prev_landmarks = lm
        motion_scores.append(motion)
        if len(motion_scores) > SEQ_LENGTH:
            motion_scores.pop(0)

        draw_text_bg(image, f"Recording {len(sequence_buffer)}/{SEQ_LENGTH}", pos=(10,10), color=(200,200,200))

        if len(sequence_buffer) >= 2:
            avg_motion = float(np.mean(motion_scores)) if motion_scores else 0.0

            # ------------------------------
            # Dynamic Gesture → Transformer
            # ------------------------------
            if avg_motion > MOTION_THRESHOLD and len(sequence_buffer) >= MIN_SEQ_FOR_TRANSFORMER:
                result = predict_transformer_tflite(list(sequence_buffer))
                last_result_text = result
                last_result_source = "Transformer"
                last_result_color = (0,255,0)
                last_pred_time = time.time()
                sequence_buffer.clear()
                motion_scores.clear()
                prev_landmarks = None
                hold_counter = 0
                draw_text_bg(image, f"{last_result_text} (Transformer)", pos=(10,10), color=last_result_color)

            # ------------------------------
            # Static Gesture → Random Forest
            # ------------------------------
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
    cv2.destroyAllWindows()
    hands.close()
    picam2.stop()