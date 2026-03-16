import sys, os
import cv2
import numpy as np
import mediapipe as mp
import glob

real_dir = '../ff_downloads/original_sequences/youtube/c40/videos/'
fake_dir = '../ff_downloads/manipulated_sequences/Deepfakes/c40/videos/'

real_vids = sorted(glob.glob(os.path.join(real_dir, '*.mp4')))[:5]
fake_vids = sorted(glob.glob(os.path.join(fake_dir, '*.mp4')))[:5]

mp_face_mesh = mp.solutions.face_mesh
FOREHEAD_IDX = [10, 338, 297, 332, 284, 251, 389]

def get_texture_variance(video_path):
    cap = cv2.VideoCapture(video_path)
    variances = []
    
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        for _ in range(50):
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0]
                pts = []
                for idx in FOREHEAD_IDX:
                    pts.append([int(lm.landmark[idx].x * w), int(lm.landmark[idx].y * h)])
                pts = np.array(pts, dtype=np.int32)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mask, pts, 255)
                green = frame[:, :, 1]
                pixels = green[mask == 255]
                if len(pixels) > 0:
                    variances.append(np.std(pixels))
                    
    cap.release()
    return np.mean(variances) if variances else 0

print("TEXTURE VARIANCE (FOREHEAD)")
for v in real_vids:
    print(f"REAL {os.path.basename(v)}: {get_texture_variance(v):.2f}")
for v in fake_vids:
    print(f"FAKE {os.path.basename(v)}: {get_texture_variance(v):.2f}")
