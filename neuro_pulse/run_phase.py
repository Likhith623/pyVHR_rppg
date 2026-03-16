import sys, os
import cv2
import numpy as np
import mediapipe as mp
import glob
from scipy import signal as scipy_signal

real_dir = '../ff_downloads/original_sequences/youtube/c40/videos/'
fake_dir = '../ff_downloads/manipulated_sequences/Deepfakes/c40/videos/'

real_vids = sorted(glob.glob(os.path.join(real_dir, '*.mp4')))[:5]
fake_vids = sorted(glob.glob(os.path.join(fake_dir, '*.mp4')))[:5]

mp_face_mesh = mp.solutions.face_mesh
LEFT_CHEEK_IDX = [234, 93, 132, 58, 172]
RIGHT_CHEEK_IDX = [454, 323, 361, 288, 397]

def extract_phase_diff(video_path):
    cap = cv2.VideoCapture(video_path)
    left_buf, right_buf = [], []
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        for _ in range(150):
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0]
                def get_mean(idx_list):
                    pts = np.array([[int(lm.landmark[i].x * w), int(lm.landmark[i].y * h)] for i in idx_list], dtype=np.int32)
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillConvexPoly(mask, pts, 255)
                    return np.mean(frame[:,:,1][mask==255])
                left_buf.append(get_mean(LEFT_CHEEK_IDX))
                right_buf.append(get_mean(RIGHT_CHEEK_IDX))
    cap.release()
    
    if len(left_buf) < 100: return 0
    left = scipy_signal.detrend(left_buf)
    right = scipy_signal.detrend(right_buf)
    nyq = 15.0
    b, a = scipy_signal.butter(4, [0.7/nyq, 4.0/nyq], btype='band')
    left = scipy_signal.filtfilt(b, a, left)
    right = scipy_signal.filtfilt(b, a, right)
    
    f, Pxy = scipy_signal.csd(left, right, fs=30.0, nperseg=len(left))
    mask = (f >= 0.7) & (f <= 4.0)
    if not np.any(mask): return 0
    idx = np.argmax(np.abs(Pxy[mask]))
    peak_phase = np.angle(Pxy[mask][idx])
    return abs(peak_phase)


print("PHASE DIFF (RADIANS)")
for v in real_vids:
    print(f"REAL {os.path.basename(v)}: {extract_phase_diff(v):.5f}")
for v in fake_vids:
    print(f"FAKE {os.path.basename(v)}: {extract_phase_diff(v):.5f}")
