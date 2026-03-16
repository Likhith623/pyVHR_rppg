"""
Test the fixed classify_liveness using saved webcam capture data.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from src.signal_processor import (
    butterworth_bandpass, resample_signal, compute_psd_welch,
    compute_snr_and_hr, compute_peak_quality, compute_spectral_purity,
    compute_peak_prominence, compute_autocorr_periodicity,
    compute_roi_correlation, compute_signal_strength, classify_liveness,
    process_signal_buffer,
)

def main():
    data = np.load("webcam_capture.npz")
    green = data["green"].tolist()
    red = data["red"].tolist()
    actual_fps = float(data["actual_fps"])
    roi_buffers = {
        "forehead": data["forehead"].tolist(),
        "left_cheek": data["left_cheek"].tolist(),
        "right_cheek": data["right_cheek"].tolist(),
    }

    print(f"Loaded webcam capture: {len(green)} frames at {actual_fps:.1f} FPS")

    # Run at actual FPS
    result = process_signal_buffer(
        green, webcam_fps=actual_fps, roi_buffers=roi_buffers, red_buffer=red
    )
    if result:
        print(f"\n  Verdict: {result['verdict']} ({result['confidence_pct']:.0f}%)")
        print(f"  HR: {result['hr_bpm']:.0f} BPM")
        print(f"  PeakQ: {result['peak_quality']:.3f}")
        print(f"  Periodicity: {result['periodicity']:.4f}")
        print(f"  Prominence: {result['peak_prominence']:.3f}")
        print(f"  ROI Corr: {result['roi_correlation']:.4f}")

    # Also at 30 FPS
    result30 = process_signal_buffer(
        green, webcam_fps=30.0, roi_buffers=roi_buffers, red_buffer=red
    )
    if result30:
        print(f"\n  At 30 FPS:")
        print(f"  Verdict: {result30['verdict']} ({result30['confidence_pct']:.0f}%)")
        print(f"  HR: {result30['hr_bpm']:.0f} BPM")
        print(f"  PeakQ: {result30['peak_quality']:.3f}")
        print(f"  Periodicity: {result30['periodicity']:.4f}")
        print(f"  Prominence: {result30['peak_prominence']:.3f}")

if __name__ == "__main__":
    main()
