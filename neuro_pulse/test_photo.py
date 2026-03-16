import sys, os
from src.signal_processor import process_signal_buffer
import numpy as np

n_samples = 300
fs = 30.0
t = np.arange(n_samples) / fs
noise = 0.5 * np.sin(2 * np.pi * 1.5 * t) + np.random.normal(0, 0.2, n_samples)

photo_roi = {
    "forehead":    (120 + noise + np.random.normal(0, 0.05, n_samples)).tolist(),
    "left_cheek":  (118 + noise + np.random.normal(0, 0.05, n_samples)).tolist(),
    "right_cheek": (118 + noise + np.random.normal(0, 0.05, n_samples)).tolist(),
}
photo_combined = (np.mean([photo_roi["forehead"], photo_roi["left_cheek"], photo_roi["right_cheek"]], axis=0)).tolist()

res = process_signal_buffer(photo_combined, webcam_fps=fs, roi_buffers=photo_roi)
if res:
    print(f"SNR: {res['snr_db']:.2f}")
    print(f"Purity: {res.get('spectral_purity',0):.2f}")
    print(f"Corr: {res.get('roi_correlation',0):.2f}")
    print(f"Periodicity: {res.get('periodicity',0):.2f}")
    print(f"Verdict: {res['verdict']}")
