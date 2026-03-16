import sys, os
from src.signal_processor import process_signal_buffer
import numpy as np

n_samples = 300
fs = 30.0
t = np.arange(n_samples) / fs
noise = np.full(n_samples, 0.0) # PERFECTLY STILL PHOTO!

photo_roi = {
    "forehead":    (120 + noise).tolist(),
    "left_cheek":  (118 + noise).tolist(),
    "right_cheek": (118 + noise).tolist(),
}
photo_combined = (np.mean([photo_roi["forehead"], photo_roi["left_cheek"], photo_roi["right_cheek"]], axis=0)).tolist()

res = process_signal_buffer(photo_combined, webcam_fps=fs, roi_buffers=photo_roi)
if res:
    print(f"SNR: {res['snr_db']:.3f}, Purity: {res.get('spectral_purity',0):.3f}")
    print(f"Verdict: {res['verdict']}")
