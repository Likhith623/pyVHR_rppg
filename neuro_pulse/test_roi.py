import os
import sys
import numpy as np
import cv2
import mediapipe as mp
sys.path.insert(0, os.path.dirname(__file__))
from src.roi_extractor import extract_roi_green_multi
from src.signal_processor import compute_psd_welch, butterworth_bandpass, resample_signal, compute_spectral_purity

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)

def test_video(path):
    cap = cv2.VideoCapture(path)
    roi_buffers = {"forehead": [], "left_cheek": [], "right_cheek": []}
    for _ in range(300):
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        res = mp_face_mesh.process(rgb)
        if res.multi_face_landmarks:
            rv = extract_roi_green_multi(frame, res.multi_face_landmarks[0], h, w)
            if rv:
                for k in roi_buffers: roi_buffers[k].append(rv[k])
    
    for k, buf in roi_buffers.items():
        if len(buf) < 150: continue
        sig = np.array(buf, dtype=np.float64)
        filt = butterworth_bandpass(sig, fs=25.0)
        resam = resample_signal(filt, original_fs=25.0, target_fs=256.0)
        f, psd = compute_psd_welch(resam, fs=256.0)
        pur = compute_spectral_purity(f, psd)
        hr = f[np.argmax(psd)] * 60.0
        print(f"  {k} purity: {pur:.3f} HR: {hr:.1f}")

print("REAL video 033:")
test_video("../ff_downloads/original_sequences/youtube/c40/videos/033.mp4")
print("FAKE video 033_097:")
test_video("../ff_downloads/manipulated_sequences/Deepfakes/c40/videos/033_097.mp4")
