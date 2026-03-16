"""
Extract liveness features from FF++ videos to find the real vs fake boundary.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
import mediapipe as mp
import glob

from src.roi_extractor import extract_roi_green_multi
from src.signal_processor import (
    butterworth_bandpass, resample_signal, compute_psd_welch,
    compute_snr_and_hr, compute_peak_quality, compute_spectral_purity,
    compute_peak_prominence, compute_autocorr_periodicity,
    compute_roi_correlation, compute_signal_strength,
)

mp_face_mesh = mp.solutions.face_mesh

def extract_liveness_features(video_path, max_frames=300):
    """Extract liveness features from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    green_buffer = []
    roi_buffers = {"forehead": [], "left_cheek": [], "right_cheek": []}
    last_roi = {"forehead": 0.0, "left_cheek": 0.0, "right_cheek": 0.0}

    with mp_face_mesh.FaceMesh(
        max_num_faces=1, min_detection_confidence=0.5,
        min_tracking_confidence=0.5, static_image_mode=False,
    ) as fm:
        count = 0
        while count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = fm.process(rgb)
            if results.multi_face_landmarks:
                fl = results.multi_face_landmarks[0]
                roi_vals = extract_roi_green_multi(frame, fl, h, w)
                if roi_vals is not None:
                    green_buffer.append(roi_vals["combined"])
                    for key in ["forehead", "left_cheek", "right_cheek"]:
                        roi_buffers[key].append(roi_vals[key])
                        last_roi[key] = roi_vals[key]
                else:
                    green_buffer.append(green_buffer[-1] if green_buffer else 0.0)
                    for key in roi_buffers:
                        roi_buffers[key].append(last_roi[key])
            else:
                green_buffer.append(green_buffer[-1] if green_buffer else 0.0)
                for key in roi_buffers:
                    roi_buffers[key].append(last_roi[key])
            count += 1
    cap.release()

    if len(green_buffer) < 150:
        return None

    sig = np.array(green_buffer, dtype=np.float64)
    filtered = butterworth_bandpass(sig, fs=fps)
    resampled = resample_signal(filtered, original_fs=fps, target_fs=256.0)
    freqs, psd = compute_psd_welch(resampled, fs=256.0)
    hr_bpm, snr_db = compute_snr_and_hr(freqs, psd)

    return {
        "peak_quality": compute_peak_quality(freqs, psd),
        "periodicity": compute_autocorr_periodicity(filtered, fs=fps),
        "peak_prominence": compute_peak_prominence(freqs, psd),
        "roi_correlation": compute_roi_correlation(roi_buffers, fs=fps),
        "spectral_purity": compute_spectral_purity(freqs, psd),
        "signal_strength": compute_signal_strength(green_buffer, fs=fps),
        "hr_bpm": hr_bpm,
        "snr_db": snr_db,
    }

def main():
    real_dir = "/Users/likhith./pyVHR_rppg/ff_downloads/original_sequences/youtube/c40/videos"
    fake_dir = "/Users/likhith./pyVHR_rppg/ff_downloads/manipulated_sequences/Deepfakes/c40/videos"

    real_files = sorted(glob.glob(os.path.join(real_dir, "*.mp4")))[:10]
    fake_files = sorted(glob.glob(os.path.join(fake_dir, "*.mp4")))[:10]

    print(f"Extracting liveness features from {len(real_files)} real + {len(fake_files)} fake videos")
    print()

    all_feats = {"real": [], "fake": []}

    for label, files in [("real", real_files), ("fake", fake_files)]:
        for f in files:
            name = os.path.basename(f)
            feats = extract_liveness_features(f)
            if feats is None:
                print(f"  {label:4s} {name:20s}: SKIPPED (not enough frames)")
                continue
            all_feats[label].append(feats)
            print(f"  {label:4s} {name:20s}: PQ={feats['peak_quality']:.2f}  "
                  f"Period={feats['periodicity']:.3f}  Prom={feats['peak_prominence']:.2f}  "
                  f"Corr={feats['roi_correlation']:.3f}  HR={feats['hr_bpm']:.0f}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY: Real vs Fake feature distributions")
    print("=" * 70)

    for feat_name in ["peak_quality", "periodicity", "peak_prominence", "roi_correlation"]:
        real_vals = [f[feat_name] for f in all_feats["real"]]
        fake_vals = [f[feat_name] for f in all_feats["fake"]]
        if real_vals and fake_vals:
            r_mean, r_std = np.mean(real_vals), np.std(real_vals)
            f_mean, f_std = np.mean(fake_vals), np.std(fake_vals)
            pooled_std = np.sqrt((r_std**2 + f_std**2) / 2) + 1e-10
            cohens_d = (r_mean - f_mean) / pooled_std
            print(f"  {feat_name:20s}: REAL={r_mean:.3f}+/-{r_std:.3f}  "
                  f"FAKE={f_mean:.3f}+/-{f_std:.3f}  Cohen's d={cohens_d:.2f}")

    # Also show what your webcam produced for comparison
    print("\n  YOUR WEBCAM (from diagnose_deep.py):")
    print(f"  {'peak_quality':20s}: ~1.8-2.3")
    print(f"  {'periodicity':20s}: ~0.16")
    print(f"  {'peak_prominence':20s}: ~0.8-1.3")
    print(f"  {'roi_correlation':20s}: ~0.35")


if __name__ == "__main__":
    main()
