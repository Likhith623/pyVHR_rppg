import os
import sys
import numpy as np
import cv2
import mediapipe as mp

sys.path.insert(0, os.path.dirname(__file__))

from src.roi_extractor import extract_roi_green_multi
from src.signal_processor import compute_psd_welch, butterworth_bandpass, resample_signal

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)
cap = cv2.VideoCapture("../ff_downloads/original_sequences/youtube/c40/videos/033.mp4")
roi_buffers = {"forehead": [], "left_cheek": [], "right_cheek": []}
for _ in range(300):
    ret, frame = cap.read()
    if not ret: break
    h, w = frame.shape[:2]
    res = mp_face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if res.multi_face_landmarks:
        rv = extract_roi_green_multi(frame, res.multi_face_landmarks[0], h, w)
        if rv:
            for k in roi_buffers: roi_buffers[k].append(rv[k])

sig = np.array(roi_buffers["forehead"], dtype=np.float64)
filt = butterworth_bandpass(sig, fs=25.0)
resam = resample_signal(filt, original_fs=25.0, target_fs=256.0)
f, psd = compute_psd_welch(resam, fs=256.0)
print("Max freqs in cardiac band (0.7 - 4.0 Hz):")
mask = (f >= 0.7) & (f <= 4.0)
top_idx = np.argsort(psd[mask])[-5:][::-1]
for i in top_idx:
    print(f"Freq: {f[mask][i]:.2f} Hz, PSD: {psd[mask][i]:.4f}")
