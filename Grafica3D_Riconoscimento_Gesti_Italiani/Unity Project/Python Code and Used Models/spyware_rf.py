
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import socket
import json
import time
from scipy.ndimage import binary_fill_holes
import pandas as pd
import joblib

# ----------- PARAMETRI CAMERA (come da MATLAB) ------------------
s = {
    "principal_point": [309.568, 245.154],
    "focal_length": [474.346, 474.346],
    "distortion_coeffs": [0.139663, 0.0914142, 0.00468509, 0.00220023, 0.0654529],
    "depth_scale": 0.000125
}

# ----------- LANDMARK E COPPIE ------------------
landmark_names = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

landmark_pairs = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5),
    (5, 9), (9, 13), (13, 17),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20)
]


THRESHOLD = 0.3 # Soglia ottimale determinata durante il training

# -------------- FUNZIONI --------------------
def load_raw_depth_map(path, width=640, height=480):
    raw = np.fromfile(path, dtype=np.uint16, count=width * height)
    if raw.size != width * height:
        raise ValueError(f"Raw size mismatch: {raw.size} vs {width*height}")
    return raw.reshape((height, width))

def fill_pixel_depth_image(img, i, j):
    h, w = img.shape
    neighbors = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0: continue
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w:
                v = img[ni, nj]
                if 0 < v < 65535:
                    neighbors.append(v)
    if neighbors:
        img[i, j] = int(np.mean(neighbors))

def process_depth_map(depth_map):
    bw = depth_map > 0
    filled = binary_fill_holes(bw)
    mask = filled.astype(np.uint8) - bw.astype(np.uint8)
    out = depth_map.copy()
    h, w = out.shape
    for i in range(1, h-1):
        for j in range(1, w-1):
            if mask[i, j]:
                fill_pixel_depth_image(out, i, j)
    return out

def rs2_deproject_pixel_to_point(s, pixel, depth):
    cx, cy = s["principal_point"]
    fx, fy = s["focal_length"]
    k1, k2, p1, p2, k3 = s["distortion_coeffs"]

    x = (pixel[0] - cx) / fx
    y = (pixel[1] - cy) / fy

    r2 = x * x + y * y
    f = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3

    ux = x * f + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    uy = y * f + 2 * p2 * x * y + p1 * (r2 + 2 * y * y)

    return np.array([depth * ux, depth * uy, depth])

def dist_3D(s, a, b):
    z1 = a[2] * s['depth_scale']
    z2 = b[2] * s['depth_scale']
    p1 = rs2_deproject_pixel_to_point(s, a[:2], z1)
    p2 = rs2_deproject_pixel_to_point(s, b[:2], z2)
    return np.linalg.norm(p2 - p1)

def load_fake_data():
    print(" Carico RGB e Depth fissi...")

    folder = r"C:\Users\Manuela\Desktop\ACQUISIZIONI"

    rgb_path = os.path.join(folder, "rgb.png")
    depth_path = os.path.join(folder, "depth.raw")

    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        print(" File rgb.png o depth.raw non trovati.")
        return None, None

    print(f" Userò RGB: {rgb_path}")
    print(f" Userò Depth: {depth_path}")

    rgb = cv2.imread(rgb_path)
    if rgb is None:
        print(f" {rgb_path} non valido.")
        return None, None
    else:
        print(f" RGB caricato: shape = {rgb.shape}")

    print(f" Carico {depth_path}...")
    depth = load_raw_depth_map(depth_path)
    depth = process_depth_map(depth)
    print(f" Depth caricata: shape = {depth.shape}, dtype = {depth.dtype}")

    #  Resize 
    rgb = cv2.resize(rgb, (640, 480))
    depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_NEAREST)

    # Rotazione e ribaltamento 
    rgb = cv2.rotate(rgb, cv2.ROTATE_180)
    rgb = cv2.flip(rgb, 1)

    # 
    depth = cv2.rotate(depth, cv2.ROTATE_180)
    depth = cv2.flip(depth, 1)

    return rgb, depth

def send_result_to_unity(predicted_class, confidence):
    print(f"📡 Invio a Unity: {predicted_class} ({confidence:.2f})")
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", 12345))
        data = {
            "frame_id": "test_manual",
            "predicted_class": predicted_class,
            "confidence": round(confidence, 3),
            "timestamp": time.time()
        }
        client.send((json.dumps(data) + "\n").encode("utf-8"))
        print(f" Dati inviati a Unity: {data}")
        client.close()
    except Exception as e:
        print(" Errore invio a Unity:", e)



def main():
    print(" Avvio script di test...")

    rgb, depth = load_fake_data()
    if rgb is None or depth is None:
        print(" Caricamento fallito. Uscita.")
        return

    print(" Inizializzazione MediaPipe...")
    model_path = "hand_landmarker.task"
    if not os.path.exists(model_path):
        print(f" Modello MediaPipe non trovato: {model_path}")
        return

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    # Salvataggio temporaneo del frame RGB e utilizzo MediaPipe da file
    tmp_path = "tmp_rgb.png"
    cv2.imwrite(tmp_path, rgb)
    mp_image = mp.Image.create_from_file(tmp_path)

    print(" Rilevamento mano in corso...")
    result = detector.detect(mp_image)

    # Pulizia file temporaneo
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    if not result.hand_landmarks:
        print(" Nessuna mano rilevata.")
        send_result_to_unity(5, 0.0)
        return
    print(" Mano rilevata.")
    
    landmarks = result.hand_landmarks[0]
    coords = []
    for lm in landmarks:
        x = round(lm.x * 639)
        y = round(lm.y * 479)
        z = int(depth[y, x]) if 0 <= x < 640 and 0 <= y < 480 else 0
        coords.append([x, y, z])
    print("Landmark coords:", coords)

    
    distances = {}
    for i, j in landmark_pairs:
        d = dist_3D(s, coords[i], coords[j]) * 100
        key = f"{landmark_names[i]}_{landmark_names[j]}"
        distances[key] = d
    
    sorted_keys = sorted(distances.keys())
    feature_vector = np.array([distances[k] for k in sorted_keys]).reshape(1, -1)
    print(" Feature vector pronto.")
   

    print(" Clipping e standardizzazione...")

    feature_vector = np.clip(feature_vector, 0, 15)
    print(" Feature vector dopo clipping:", feature_vector)


    scaler = joblib.load("scaler_rf.pkl")
    X_scaled = scaler.transform(feature_vector)

    clf = joblib.load("calibrated_rf_model.pkl")

    probs = clf.predict_proba(X_scaled)[0]
    pred = int(clf.predict(X_scaled)[0])         
    max_prob = float(np.max(probs))              


    if max_prob < THRESHOLD:
        final_class = 5
    else:
        final_class = pred

    print(f" Gesto classificato: {final_class} (confidence {max_prob:.2f})")
    send_result_to_unity(str(final_class), max_prob)

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    print(" Fine main(), classificazione completata.")

