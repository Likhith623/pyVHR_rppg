"""
Neuro-Pulse Deepfake Detector Module.

Provides video-level deepfake detection by combining:
  1. rPPG physiological features (spectral purity, peak quality, periodicity)
  2. Visual artifact features (Laplacian ratio, DCT high-frequency, noise)

The detector operates in two modes:
  - LIVENESS mode (webcam): Uses rPPG features only — detects photos, screens,
    masks.  Works via the classify_liveness() function in signal_processor.py.
  - DEEPFAKE mode (uploaded video): Extracts rPPG + visual features from the
    video and classifies using a Gradient-Boosting model trained on FF++ data.
"""

import os
import cv2
import numpy as np
import mediapipe as mp

from src.roi_extractor import (
    _get_roi_points,
    FOREHEAD_IDX,
    LEFT_CHEEK_IDX,
    RIGHT_CHEEK_IDX,
)
from src.signal_processor import (
    butterworth_bandpass,
    resample_signal,
    compute_psd_welch,
    compute_snr_and_hr,
    compute_spectral_purity,
    compute_peak_quality,
    compute_peak_prominence,
    compute_autocorr_periodicity,
)

mp_face_mesh = mp.solutions.face_mesh

# Jawline + face outline landmarks for inner / boundary Laplacian analysis
INNER_FACE_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# Expected feature order (must match training)
FEATURE_NAMES = [
    "amp_discord", "dct_hf", "dct_hf_std", "face_jitter",
    "face_jitter_std", "gr_purity_diff", "inner_lap_std",
    "lap_ratio", "noise_mean", "noise_std", "peak_prominence",
    "peak_quality", "periodicity", "purity", "red_purity",
    "temporal_diff", "temporal_diff_std",
]


def extract_video_features(video_path: str, max_frames: int = 300) -> dict | None:
    """Extract rPPG + visual features from a video file.

    Returns a dict mapping each feature name to its float value,
    or None if the video has insufficient face frames.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    green_f, green_l, green_r = [], [], []
    red_f, red_l, red_r = [], [], []
    inner_laps, boundary_laps = [], []
    noise_levels, dct_ratios = [], []
    temporal_diffs, face_consistency = [], []
    prev_gray = None
    face_count, frame_count = 0, 0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False,
    ) as fm:
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            results = fm.process(rgb)

            if results.multi_face_landmarks:
                fl = results.multi_face_landmarks[0]
                green, red = frame[:, :, 1], frame[:, :, 2]
                face_count += 1

                # ── Per-ROI colour extraction ──
                for indices, gb, rb in [
                    (FOREHEAD_IDX, green_f, red_f),
                    (LEFT_CHEEK_IDX, green_l, red_l),
                    (RIGHT_CHEEK_IDX, green_r, red_r),
                ]:
                    pts = _get_roi_points(fl, indices, h, w)
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillConvexPoly(mask, pts, 255)
                    gp = green[mask == 255]
                    rp = red[mask == 255]
                    if len(gp) > 0:
                        gb.append(float(np.mean(gp)))
                        rb.append(float(np.mean(rp)))

                # ── Visual artifact features ──
                pts = _get_roi_points(fl, INNER_FACE_IDX, h, w)
                inner_mask = np.zeros((h, w), dtype=np.uint8)
                hull = cv2.convexHull(pts)
                cv2.fillConvexPoly(inner_mask, hull, 255)

                kernel = np.ones((15, 15), np.uint8)
                dilated = cv2.dilate(inner_mask, kernel, iterations=1)
                eroded = cv2.erode(inner_mask, kernel, iterations=1)
                boundary_mask = dilated - eroded

                lap = cv2.Laplacian(gray, cv2.CV_64F)
                if np.any(inner_mask == 255):
                    inner_laps.append(float(np.var(lap[inner_mask == 255])))
                if np.any(boundary_mask == 255):
                    boundary_laps.append(float(np.var(lap[boundary_mask == 255])))

                b1 = cv2.GaussianBlur(gray, (3, 3), 0)
                b2 = cv2.GaussianBlur(gray, (7, 7), 0)
                dog = b1 - b2
                if np.any(inner_mask == 255):
                    noise_levels.append(float(np.std(dog[inner_mask == 255])))

                x2, y2, bw, bh = cv2.boundingRect(hull)
                if bw > 32 and bh > 32:
                    face_roi = gray[y2 : y2 + bh, x2 : x2 + bw]
                    face_roi = cv2.resize(face_roi, (64, 64))
                    dct = cv2.dct(face_roi.astype(np.float32))
                    hf = float(np.sum(np.abs(dct[32:, 32:])))
                    tt = float(np.sum(np.abs(dct)))
                    if tt > 0:
                        dct_ratios.append(hf / tt)

                if prev_gray is not None:
                    diff = np.abs(gray - prev_gray)
                    fd = diff[inner_mask == 255]
                    if len(fd) > 0:
                        temporal_diffs.append(float(np.mean(fd)))
                        face_consistency.append(float(np.std(fd)))

            prev_gray = gray.copy()
            frame_count += 1
    cap.release()

    min_len = min(len(green_f), len(green_l), len(green_r))
    if min_len < 90 or len(inner_laps) < 30:
        return None

    feats: dict = {}

    # ── rPPG features ──
    try:
        combined = [
            (f + l + r) / 3
            for f, l, r in zip(
                green_f[:min_len], green_l[:min_len], green_r[:min_len]
            )
        ]
        sig = np.array(combined, dtype=np.float64)
        filtered = butterworth_bandpass(sig, fs=fps)
        resampled = resample_signal(filtered, original_fs=fps, target_fs=256.0)
        freqs, psd = compute_psd_welch(resampled, fs=256.0)

        feats["purity"] = compute_spectral_purity(freqs, psd)
        feats["peak_quality"] = compute_peak_quality(freqs, psd)
        feats["peak_prominence"] = compute_peak_prominence(freqs, psd)
        feats["periodicity"] = compute_autocorr_periodicity(filtered, fs=fps)

        # Per-ROI amplitude discord
        f_sig = butterworth_bandpass(
            np.array(green_f[:min_len], dtype=np.float64), fs=fps
        )
        l_sig = butterworth_bandpass(
            np.array(green_l[:min_len], dtype=np.float64), fs=fps
        )
        r_sig = butterworth_bandpass(
            np.array(green_r[:min_len], dtype=np.float64), fs=fps
        )
        amps = [np.std(f_sig), np.std(l_sig), np.std(r_sig)]
        feats["amp_discord"] = float(np.std(amps) / (np.mean(amps) + 1e-10))

        # Red-channel spectral purity (chrominance check)
        r_combined = [
            (f + l + r) / 3
            for f, l, r in zip(
                red_f[:min_len], red_l[:min_len], red_r[:min_len]
            )
        ]
        r_filt = butterworth_bandpass(
            np.array(r_combined, dtype=np.float64), fs=fps
        )
        r_res = resample_signal(r_filt, original_fs=fps, target_fs=256.0)
        r_freqs, r_psd = compute_psd_welch(r_res, fs=256.0)
        feats["red_purity"] = compute_spectral_purity(r_freqs, r_psd)
        feats["gr_purity_diff"] = feats["purity"] - feats["red_purity"]
    except Exception:
        return None

    # ── Visual features ──
    feats["lap_ratio"] = float(
        np.mean(inner_laps) / (np.mean(boundary_laps) + 1e-10)
    )
    feats["inner_lap_std"] = float(np.std(inner_laps))
    feats["noise_mean"] = float(np.mean(noise_levels)) if noise_levels else 0.0
    feats["noise_std"] = float(np.std(noise_levels)) if noise_levels else 0.0
    feats["dct_hf"] = float(np.mean(dct_ratios)) if dct_ratios else 0.0
    feats["dct_hf_std"] = float(np.std(dct_ratios)) if dct_ratios else 0.0
    feats["temporal_diff"] = (
        float(np.mean(temporal_diffs)) if temporal_diffs else 0.0
    )
    feats["temporal_diff_std"] = (
        float(np.std(temporal_diffs)) if temporal_diffs else 0.0
    )
    feats["face_jitter"] = (
        float(np.mean(face_consistency)) if face_consistency else 0.0
    )
    feats["face_jitter_std"] = (
        float(np.std(face_consistency)) if face_consistency else 0.0
    )

    return feats


def classify_video(video_path: str) -> dict:
    """Classify a video file as REAL or FAKE (deepfake).

    Uses the trained Gradient-Boosting classifier on rPPG + visual features.
    Falls back to a rule-based heuristic if the model files are missing.

    Returns:
        Dict with keys: verdict, confidence_pct, features, method.
    """
    feats = extract_video_features(video_path)
    if feats is None:
        return {
            "verdict": "ERROR",
            "confidence_pct": 0.0,
            "features": {},
            "method": "insufficient_data",
        }

    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    clf_path = os.path.join(model_dir, "deepfake_clf.pkl")
    names_path = os.path.join(model_dir, "feature_names.pkl")

    if os.path.exists(clf_path) and os.path.exists(names_path):
        import joblib

        clf = joblib.load(clf_path)
        feature_names = joblib.load(names_path)
        X = np.array([[feats.get(k, 0.0) for k in feature_names]])
        pred = clf.predict(X)[0]
        proba = clf.predict_proba(X)[0] if hasattr(clf, "predict_proba") else None

        if proba is not None:
            conf = float(max(proba) * 100.0)
        else:
            conf = 70.0

        verdict = "REAL" if pred == 1 else "FAKE"
        return {
            "verdict": verdict,
            "confidence_pct": conf,
            "features": feats,
            "method": "ml_classifier",
        }

    # Fallback: heuristic based on lap_ratio (best single feature)
    lap_ratio = feats.get("lap_ratio", 0.5)
    verdict = "REAL" if lap_ratio > 0.65 else "FAKE"
    conf = float(min(99, max(50, abs(lap_ratio - 0.65) * 200 + 50)))
    return {
        "verdict": verdict,
        "confidence_pct": conf,
        "features": feats,
        "method": "heuristic_fallback",
    }
