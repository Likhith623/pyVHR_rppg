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

def get_landmark_jitter(video_path):
    cap = cv2.VideoCapture(video_path)
    dists = []
    
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        for _ in range(150):
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0]
                # distance between nose tip (1) and chin (152) or somewhere
                nose = lm.landmark[1]
                chin = lm.landmark[152]
                dist = np.sqrt((nose.x - chin.x)**2 + (nose.y - chin.y)**2)
                dists.append(dist)
                    
    cap.release()
    if len(dists) > 1:
        return np.std(dists) * 1000
    return 0

print("LANDMARK JITTER (*1000)")
for v in real_vids:
    print(f"REAL {os.path.basename(v)}: {get_landmark_jitter(v):.5f}")
for v in fake_vids:
    print(f"FAKE {os.path.basename(v)}: {get_landmark_jitter(v):.5f}")
