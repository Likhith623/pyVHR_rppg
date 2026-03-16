"""
Diagnostic script: Capture 200 frames from webcam and print all features.
This shows exactly what classify_liveness sees for a real face.
Run: cd neuro_pulse && python diagnose_webcam.py
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
    compute_roi_correlation, compute_signal_strength, classify_liveness,
    process_signal_buffer,
)

mp_face_mesh = mp.solutions.face_mesh

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FPS, 30)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Webcam FPS: {fps}")

    green_buffer = []
    red_buffer = []
    roi_buffers = {"forehead": [], "left_cheek": [], "right_cheek": []}
    last_roi = {"forehead": 0.0, "left_cheek": 0.0, "right_cheek": 0.0}
    face_count = 0
    total_frames = 200  # ~6.7 seconds at 30fps

    print(f"Capturing {total_frames} frames. Look at the camera and stay still...")
    print()

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=False,
    ) as face_mesh:
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                print(f"Frame {i}: FAILED to read")
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

            if (i + 1) % 50 == 0:
                print(f"  Captured {i + 1}/{total_frames} frames (faces: {face_count})")

    cap.release()
    print(f"\nCapture done. {face_count} face frames out of {total_frames}")
    print(f"Green buffer size: {len(green_buffer)}")
    print()

    if len(green_buffer) < 150:
        print("ERROR: Not enough frames for analysis!")
        sys.exit(1)

    # === Run full pipeline manually to see every feature ===
    sig = np.array(green_buffer, dtype=np.float64)
    filtered = butterworth_bandpass(sig, fs=fps)
    resampled = resample_signal(filtered, original_fs=fps, target_fs=256.0)
    freqs, psd = compute_psd_welch(resampled, fs=256.0)
    hr_bpm, snr_db = compute_snr_and_hr(freqs, psd)

    roi_corr = compute_roi_correlation(roi_buffers, fs=fps)
    peak_qual = compute_peak_quality(freqs, psd)
    purity = compute_spectral_purity(freqs, psd)
    peak_prom = compute_peak_prominence(freqs, psd)
    periodicity = compute_autocorr_periodicity(filtered, fs=fps)
    sig_str = compute_signal_strength(green_buffer, fs=fps)

    # Red channel purity
    r_purity = 0.0
    if len(red_buffer) == len(green_buffer):
        try:
            r_filt = butterworth_bandpass(np.array(red_buffer, dtype=np.float64), fs=fps)
            r_resampled = resample_signal(r_filt, original_fs=fps, target_fs=256.0)
            r_freqs, r_psd = compute_psd_welch(r_resampled, fs=256.0)
            r_purity = compute_spectral_purity(r_freqs, r_psd)
        except Exception as e:
            print(f"Red purity error: {e}")

    print("=" * 60)
    print("FEATURE VALUES FROM YOUR REAL FACE")
    print("=" * 60)
    print(f"  Heart Rate       : {hr_bpm:.1f} BPM")
    print(f"  SNR              : {snr_db:.1f} dB")
    print(f"  Peak Quality     : {peak_qual:.3f}   (need >2.5 for LIVE)")
    print(f"  Periodicity      : {periodicity:.4f}   (need >0.15 for LIVE)")
    print(f"  Peak Prominence  : {peak_prom:.3f}    (need >0.5 for LIVE)")
    print(f"  ROI Correlation  : {roi_corr:.4f}")
    print(f"  Spectral Purity  : {purity:.4f}")
    print(f"  Red Purity       : {r_purity:.4f}")
    print(f"  Signal Strength  : {sig_str:.4f}")
    print()

    # === Show the composite score calculation ===
    pq_score = float(np.clip((peak_qual - 1.0) / 4.0 * 100.0, 0, 100))
    period_score = float(np.clip((periodicity / 0.5) * 100.0, 0, 100))
    prom_score = float(np.clip((peak_prom / 2.0) * 100.0, 0, 100))
    corr_bonus = 0.0
    if 0.15 < roi_corr < 0.96:
        corr_bonus = float(np.clip(roi_corr * 50.0, 0, 30))

    composite = (0.40 * pq_score + 0.35 * period_score + 0.15 * prom_score + 0.10 * corr_bonus)

    print("COMPOSITE SCORE BREAKDOWN")
    print("-" * 60)
    print(f"  PQ score     : {pq_score:.1f}/100  (weight 0.40 -> {0.40*pq_score:.1f})")
    print(f"  Period score : {period_score:.1f}/100  (weight 0.35 -> {0.35*period_score:.1f})")
    print(f"  Prom score   : {prom_score:.1f}/100  (weight 0.15 -> {0.15*prom_score:.1f})")
    print(f"  Corr bonus   : {corr_bonus:.1f}/30   (weight 0.10 -> {0.10*corr_bonus:.1f})")
    print(f"  COMPOSITE    : {composite:.1f}/100  (threshold: 30.0)")
    print(f"  is_live (before vetoes): {composite > 30.0}")
    print()

    # Check vetoes
    veto_screen = (r_purity > 0.45 and purity > 0.45)
    veto_shaken = (roi_corr > 0.97 and periodicity < 0.25)
    print("VETO CHECKS")
    print("-" * 60)
    print(f"  Screen veto (red>{0.45} AND green>{0.45}): red={r_purity:.3f}, green={purity:.3f} -> {'VETOED!' if veto_screen else 'pass'}")
    print(f"  Shaken veto (corr>{0.97} AND period<{0.25}): corr={roi_corr:.3f}, period={periodicity:.4f} -> {'VETOED!' if veto_shaken else 'pass'}")
    print()

    # === Run the actual classify_liveness ===
    verdict, confidence = classify_liveness(
        snr_db,
        roi_correlation=roi_corr,
        peak_quality=peak_qual,
        signal_strength=sig_str,
        spectral_purity=purity,
        peak_prominence=peak_prom,
        periodicity=periodicity,
        red_purity=r_purity,
    )

    print("=" * 60)
    print(f"FINAL VERDICT: {verdict} ({confidence:.0f}%)")
    print("=" * 60)

    if verdict == "SYNTHETIC":
        print()
        print("*** YOUR REAL FACE IS BEING CLASSIFIED AS SYNTHETIC ***")
        print("Root cause analysis:")
        if composite <= 30.0:
            print(f"  -> Composite score ({composite:.1f}) is below threshold (30.0)")
            if pq_score < 37.5:
                print(f"     -> Peak Quality too low: {peak_qual:.3f} (need ~2.5+)")
            if period_score < 30:
                print(f"     -> Periodicity too low: {periodicity:.4f} (need ~0.15+)")
            if prom_score < 25:
                print(f"     -> Peak Prominence too low: {peak_prom:.3f} (need ~0.5+)")
        if veto_screen:
            print(f"  -> SCREEN VETO triggered! Red purity={r_purity:.3f} and spectral purity={purity:.3f}")
        if veto_shaken:
            print(f"  -> SHAKEN PHOTO VETO triggered! ROI corr={roi_corr:.3f} and periodicity={periodicity:.4f}")

    # Also run process_signal_buffer for comparison
    print()
    print("--- process_signal_buffer result ---")
    result = process_signal_buffer(
        green_buffer, webcam_fps=fps, roi_buffers=roi_buffers, red_buffer=red_buffer
    )
    if result:
        print(f"  Verdict: {result['verdict']} ({result['confidence_pct']:.0f}%)")
        print(f"  HR: {result['hr_bpm']:.0f} BPM")
    else:
        print("  process_signal_buffer returned None!")


if __name__ == "__main__":
    main()
