"""
Deep diagnostic: Capture 300 frames, save raw buffers, and analyse everything.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
import mediapipe as mp

from src.roi_extractor import extract_roi_green_multi
from src.signal_processor import (
    butterworth_bandpass, resample_signal, compute_psd_welch,
    compute_snr_and_hr, compute_peak_quality, compute_spectral_purity,
    compute_peak_prominence, compute_autocorr_periodicity,
    compute_roi_correlation, compute_signal_strength,
)

mp_face_mesh = mp.solutions.face_mesh

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FPS, 30)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Webcam FPS reported: {fps}")

    green_buffer = []
    red_buffer = []
    roi_buffers = {"forehead": [], "left_cheek": [], "right_cheek": []}
    last_roi = {"forehead": 0.0, "left_cheek": 0.0, "right_cheek": 0.0}
    face_count = 0
    total_frames = 300  # 10 seconds at 30fps
    import time
    t0 = time.time()

    print(f"Capturing {total_frames} frames. Look at the camera and stay still...")

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=False,
    ) as face_mesh:
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                fl = results.multi_face_landmarks[0]
                roi_vals = extract_roi_green_multi(frame, fl, h, w)
                if roi_vals is not None:
                    green_buffer.append(roi_vals["combined"])
                    red_buffer.append(roi_vals.get("red_combined", 0.0))
                    for key in ["forehead", "left_cheek", "right_cheek"]:
                        roi_buffers[key].append(roi_vals[key])
                        last_roi[key] = roi_vals[key]
                    face_count += 1
                else:
                    green_buffer.append(green_buffer[-1] if green_buffer else 0.0)
                    red_buffer.append(red_buffer[-1] if red_buffer else 0.0)
                    for key in roi_buffers:
                        roi_buffers[key].append(last_roi[key])
            else:
                green_buffer.append(green_buffer[-1] if green_buffer else 0.0)
                red_buffer.append(red_buffer[-1] if red_buffer else 0.0)
                for key in roi_buffers:
                    roi_buffers[key].append(last_roi[key])

    elapsed = time.time() - t0
    actual_fps = total_frames / elapsed
    cap.release()

    print(f"\nCapture done in {elapsed:.1f}s")
    print(f"Actual FPS: {actual_fps:.1f}")
    print(f"Face detection: {face_count}/{total_frames} ({100*face_count/total_frames:.0f}%)")
    print(f"Green buffer: {len(green_buffer)} samples")

    # Analyse raw green signal
    gb = np.array(green_buffer, dtype=np.float64)
    print(f"\nRaw green signal stats:")
    print(f"  Mean: {np.mean(gb):.2f}")
    print(f"  Std:  {np.std(gb):.4f}")
    print(f"  Min:  {np.min(gb):.2f}")
    print(f"  Max:  {np.max(gb):.2f}")

    # Per-ROI stats
    print(f"\nPer-ROI raw stats:")
    for key in ["forehead", "left_cheek", "right_cheek"]:
        arr = np.array(roi_buffers[key], dtype=np.float64)
        print(f"  {key:15s}: mean={np.mean(arr):.2f}, std={np.std(arr):.4f}")

    # ROI correlation - check what happens
    print(f"\n--- ROI Correlation Debug ---")
    for key in ["forehead", "left_cheek", "right_cheek"]:
        buf = roi_buffers[key]
        print(f"  {key} len: {len(buf)}")
        if len(buf) >= 150:
            sig = butterworth_bandpass(np.array(buf, dtype=np.float64), fs=actual_fps)
            print(f"  {key} filtered std: {np.std(sig):.6f}")

    roi_corr = compute_roi_correlation(roi_buffers, fs=actual_fps)
    print(f"  ROI correlation: {roi_corr:.4f}")

    # Run signal processing at actual FPS
    print(f"\n--- Feature values at actual FPS ({actual_fps:.1f}) ---")
    filtered = butterworth_bandpass(gb, fs=actual_fps)
    resampled = resample_signal(filtered, original_fs=actual_fps, target_fs=256.0)
    freqs, psd = compute_psd_welch(resampled, fs=256.0)
    hr_bpm, snr_db = compute_snr_and_hr(freqs, psd)
    peak_qual = compute_peak_quality(freqs, psd)
    purity = compute_spectral_purity(freqs, psd)
    peak_prom = compute_peak_prominence(freqs, psd)
    periodicity = compute_autocorr_periodicity(filtered, fs=actual_fps)
    sig_str = compute_signal_strength(green_buffer, fs=actual_fps)

    print(f"  Heart Rate       : {hr_bpm:.1f} BPM")
    print(f"  SNR              : {snr_db:.1f} dB")
    print(f"  Peak Quality     : {peak_qual:.3f}")
    print(f"  Periodicity      : {periodicity:.4f}")
    print(f"  Peak Prominence  : {peak_prom:.3f}")
    print(f"  ROI Correlation  : {roi_corr:.4f}")
    print(f"  Spectral Purity  : {purity:.4f}")
    print(f"  Signal Strength  : {sig_str:.4f}")

    # Also test with 30 FPS assumption
    print(f"\n--- Feature values at assumed 30 FPS ---")
    filtered30 = butterworth_bandpass(gb, fs=30.0)
    resampled30 = resample_signal(filtered30, original_fs=30.0, target_fs=256.0)
    freqs30, psd30 = compute_psd_welch(resampled30, fs=256.0)
    hr30, snr30 = compute_snr_and_hr(freqs30, psd30)
    pq30 = compute_peak_quality(freqs30, psd30)
    pp30 = compute_peak_prominence(freqs30, psd30)
    per30 = compute_autocorr_periodicity(filtered30, fs=30.0)
    roi_corr30 = compute_roi_correlation(roi_buffers, fs=30.0)
    print(f"  Heart Rate       : {hr30:.1f} BPM")
    print(f"  Peak Quality     : {pq30:.3f}")
    print(f"  Periodicity      : {per30:.4f}")
    print(f"  Peak Prominence  : {pp30:.3f}")
    print(f"  ROI Correlation  : {roi_corr30:.4f}")

    # Save raw buffers for offline analysis
    np.savez(
        "webcam_capture.npz",
        green=gb,
        red=np.array(red_buffer),
        forehead=np.array(roi_buffers["forehead"]),
        left_cheek=np.array(roi_buffers["left_cheek"]),
        right_cheek=np.array(roi_buffers["right_cheek"]),
        actual_fps=actual_fps,
    )
    print(f"\nRaw data saved to webcam_capture.npz")


if __name__ == "__main__":
    main()
