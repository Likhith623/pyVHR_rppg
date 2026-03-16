import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "neuro_pulse"))
import cv2
import numpy as np
import mediapipe as mp
from src.roi_extractor import extract_roi_green_multi
from src.signal_processor import process_signal_buffer

def run_v(p):
    print(f"\nProcessing {p}")
    cap = cv2.VideoCapture(p)
    fm = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)
    g, b = [], {"forehead":[], "left_cheek":[], "right_cheek":[]}
    for i in range(300):
        ret, f_img = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(f_img, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)
        if res.multi_face_landmarks:
            roi_res = extract_roi_green_multi(f_img, res.multi_face_landmarks[0], f_img.shape[0], f_img.shape[1])
            if roi_res:
                g.append(roi_res["mean_green"])
                for k in ["forehead", "left_cheek", "right_cheek"]:
                    b[k].append(roi_res["roi_means"][k])
    
    if len(g) > 100:
        res = process_signal_buffer(g, 30.0, b)
        if res:
            for k in ['verdict', 'confidence_pct', 'snr_db', 'spectral_purity', 'roi_correlation', 'peak_quality', 'periodicity']:
                v = res.get(k)
                if isinstance(v, float): print(f"{k}: {v:.3f}")
                else: print(f"{k}: {v}")
    else:
        print("Not enough frames")

run_v("neuro_pulse/datasets/real/033.mp4")
run_v("neuro_pulse/datasets/synthetic/033_097.mp4")
