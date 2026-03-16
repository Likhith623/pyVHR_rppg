import cv2
import numpy as np
import mediapipe as mp
import time
import sys
# fix path for imports
sys.path.append("neuro_pulse")
from src.roi_extractor import extract_rois, get_bvp_from_rois
from src.signal_processor import process_signal_buffer

def process_video(path):
    print(f"\nProcessing {path}...")
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    mp_face_mesh = mp.solutions.face_mesh
    fm = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    green_buffer = []
    roi_buffers = {"forehead": [], "left_cheek": [], "right_cheek": []}
    
    LIMIT = 200
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret or count > LIMIT: break
        count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)
        if res.multi_face_landmarks:
            rois = extract_rois(frame, res.multi_face_landmarks[0])
            if rois:
                bvp_val, rgbs = get_bvp_from_rois(rois)
                green_buffer.append(bvp_val)
                roi_buffers["forehead"].append(rgbs["forehead"][1])
                roi_buffers["left_cheek"].append(rgbs["left_cheek"][1])
                roi_buffers["right_cheek"].append(rgbs["right_cheek"][1])
    
    results = process_signal_buffer(green_buffer, fps, roi_buffers)
    print(f"Verdict: {results["verdict"]} ({results["confidence_pct"]}%)")
    print(f"SNR: {results["snr_db"]:.2f}")
    print(f"Purity: {results["spectral_purity"]:.2f}")
    print(f"Corr: {results["roi_correlation"]:.2f}")
    print(f"PeakQ: {results["peak_quality"]:.2f}")
    print(f"Periodicity: {results["periodicity"]:.2f}")
    return results

try:
    process_video("neuro_pulse/datasets/real/033.mp4")
    process_video("neuro_pulse/datasets/synthetic/033_097.mp4")
except Exception as e:
    print(e)
