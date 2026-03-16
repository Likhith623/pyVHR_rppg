import sys
sys.path.append("neuro_pulse")
import cv2
from src.roi_extractor import extract_rois, get_bvp_from_rois
from src.signal_processor import process_signal_buffer
import mediapipe as mp

def run_v(p):
    print(f"\nProcessing {p}")
    cap = cv2.VideoCapture(p)
    fm = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)
    g, b = [], {"forehead":[], "left_cheek":[], "right_cheek":[]}
    for _ in range(300):
        ret, f_img = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(f_img, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)
        if res.multi_face_landmarks:
            r = extract_rois(f_img, res.multi_face_landmarks[0])
            if r:
                bv, rc = get_bvp_from_rois(r)
                g.append(bv)
                for k in ["forehead", "left_cheek", "right_cheek"]:
                    b[k].append(rc[k][1])
    res = process_signal_buffer(g, 30.0, b)
    if res:
        for k in ['verdict', 'confidence_pct', 'snr_db', 'spectral_purity', 'roi_correlation', 'peak_quality', 'periodicity']:
            v = res.get(k)
            if isinstance(v, float): print(f"{k}: {v:.3f}")
            else: print(f"{k}: {v}")

run_v("neuro_pulse/datasets/real/033.mp4")
run_v("neuro_pulse/datasets/synthetic/033_097.mp4")
