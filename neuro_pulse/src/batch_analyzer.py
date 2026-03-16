import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

"""
Neuro-Pulse Batch Analyzer Module.

Analyses directories of real and fake videos using two detection modes:
  1. rPPG liveness (heartbeat-based) — for webcam / real-time
  2. ML deepfake classifier (rPPG + visual features) — for uploaded videos

Computes detection metrics and exports results to CSV.
"""

import argparse
import csv
import glob
import logging
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
import mediapipe as mp

from src.roi_extractor import extract_roi_green_multi
from src.signal_processor import process_signal_buffer
from src.deepfake_detector import classify_video as ml_classify_video

# ──────────────────────────────────────────────
# Module-level FaceMesh instance (initialised once)
# ──────────────────────────────────────────────
_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False,
)


def analyze_video(
    video_path: str,
    snr_threshold: float = 3.0,
) -> Dict:
    """Analyse a single video file for deepfake detection.

    Uses the ML deepfake classifier (rPPG + visual features) for video files.
    Falls back to rPPG-only liveness classification if the ML model is unavailable.
    """
    filename = os.path.basename(video_path)

    try:
        # Primary: ML classifier (rPPG + visual features)
        ml_result = ml_classify_video(video_path)
        if ml_result["verdict"] != "ERROR":
            # Map ML verdicts to pipeline verdicts
            verdict = "LIVE HUMAN" if ml_result["verdict"] == "REAL" else "SYNTHETIC"
            feats = ml_result.get("features", {})
            return {
                "filename": filename,
                "verdict": verdict,
                "snr_db": 0.0,
                "hr_bpm": 0.0,
                "confidence_pct": ml_result["confidence_pct"],
                "spectral_purity": feats.get("purity", 0.0),
                "periodicity": feats.get("periodicity", 0.0),
                "peak_prominence": feats.get("peak_prominence", 0.0),
                "roi_correlation": 0.0,
                "peak_quality": feats.get("peak_quality", 0.0),
                "signal_strength": 0.0,
                "lap_ratio": feats.get("lap_ratio", 0.0),
                "dct_hf": feats.get("dct_hf", 0.0),
                "frames_processed": 300,
                "fps_estimated": 30.0,
                "face_detection_rate": 1.0,
                "method": ml_result.get("method", "unknown"),
            }
    except Exception as e:
        logging.warning(f"ML classifier failed for {video_path}: {e}, falling back to rPPG")

    # Fallback: rPPG-only liveness detection
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        green_buffer: List[float] = []
        roi_buffers: Dict[str, List[float]] = {
            "forehead": [], "left_cheek": [], "right_cheek": [],
        }
        frames_processed = 0
        faces_detected = 0
        max_frames = 300

        last_roi = {"forehead": 0.0, "left_cheek": 0.0, "right_cheek": 0.0}

        while frames_processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = _face_mesh.process(rgb)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                roi_vals = extract_roi_green_multi(frame, face_landmarks, h, w)

                if roi_vals is not None:
                    green_buffer.append(roi_vals["combined"])
                    for key in ["forehead", "left_cheek", "right_cheek"]:
                        roi_buffers[key].append(roi_vals[key])
                        last_roi[key] = roi_vals[key]
                    faces_detected += 1
                else:
                    green_buffer.append(green_buffer[-1] if green_buffer else 0.0)
                    for key in ["forehead", "left_cheek", "right_cheek"]:
                        roi_buffers[key].append(last_roi[key])
            else:
                green_buffer.append(green_buffer[-1] if green_buffer else 0.0)
                for key in ["forehead", "left_cheek", "right_cheek"]:
                    roi_buffers[key].append(last_roi[key])

            frames_processed += 1

        cap.release()

        face_rate = faces_detected / frames_processed if frames_processed > 0 else 0.0

        result = process_signal_buffer(
            green_buffer,
            webcam_fps=fps,
            roi_buffers=roi_buffers,
            threshold=snr_threshold,
        )

        if result is not None:
            return {
                "filename": filename,
                "verdict": result["verdict"],
                "snr_db": result["snr_db"],
                "hr_bpm": result["hr_bpm"],
                "confidence_pct": result["confidence_pct"],
                "spectral_purity": result.get("spectral_purity", 0.0),
                "periodicity": result.get("periodicity", 0.0),
                "peak_prominence": result.get("peak_prominence", 0.0),
                "roi_correlation": result.get("roi_correlation", 0.0),
                "peak_quality": result.get("peak_quality", 0.0),
                "signal_strength": result.get("signal_strength", 0.0),
                "frames_processed": frames_processed,
                "fps_estimated": fps,
                "face_detection_rate": face_rate,
            }
        else:
            return {
                "filename": filename,
                "verdict": "ERROR",
                "snr_db": 0.0,
                "hr_bpm": 0.0,
                "confidence_pct": 0.0,
                "roi_correlation": 0.0,
                "peak_quality": 0.0,
                "signal_strength": 0.0,
                "frames_processed": frames_processed,
                "fps_estimated": fps,
                "face_detection_rate": face_rate,
            }

    except Exception as e:
        logging.error(f"Error analysing {video_path}: {e}")
        return {
            "filename": filename,
            "verdict": "ERROR",
            "snr_db": 0.0,
            "hr_bpm": 0.0,
            "confidence_pct": 0.0,
            "roi_correlation": 0.0,
            "peak_quality": 0.0,
            "signal_strength": 0.0,
            "frames_processed": 0,
            "fps_estimated": 0.0,
            "face_detection_rate": 0.0,
        }


def analyze_directory(
    directory: str,
    label: str,
    snr_threshold: float = 3.0,
) -> List[Dict]:
    """Analyse all video files in a directory."""
    video_extensions = ("*.mp4", "*.mov", "*.avi")
    video_files: List[str] = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(directory, ext)))

    video_files.sort()

    if len(video_files) == 0:
        print(f"  WARNING: No video files found in {directory}")
        return []

    print(f"  Found {len(video_files)} videos in {directory} (label={label})")

    results: List[Dict] = []

    try:
        from tqdm import tqdm
        iterator = tqdm(video_files, desc=f"  {label}")
    except ImportError:
        iterator = video_files

    for i, video_path in enumerate(iterator):
        result = analyze_video(video_path, snr_threshold=snr_threshold)
        result["ground_truth"] = label
        results.append(result)

        if not hasattr(iterator, "set_description") and (i + 1) % 5 == 0:
            print(f"    Processed {i + 1}/{len(video_files)}")

    return results


def compute_metrics(results: List[Dict]) -> Dict:
    """Compute detection accuracy metrics from batch results."""
    tp = tn = fp = fn = 0

    for r in results:
        verdict = r.get("verdict", "ERROR")
        truth = r.get("ground_truth", "UNKNOWN")

        if verdict == "ERROR":
            continue

        if verdict == "LIVE HUMAN" and truth == "REAL":
            tp += 1
        elif verdict == "SYNTHETIC" and truth == "FAKE":
            tn += 1
        elif verdict == "LIVE HUMAN" and truth == "FAKE":
            fp += 1
        elif verdict == "SYNTHETIC" and truth == "REAL":
            fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Save CSV
    os.makedirs("outputs", exist_ok=True)
    csv_path = "outputs/batch_results.csv"
    fieldnames = [
        "filename", "ground_truth", "verdict", "snr_db", "hr_bpm",
        "confidence_pct", "roi_correlation", "peak_quality", "signal_strength",
        "frames_processed", "fps_estimated", "face_detection_rate",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\n  Results saved to {csv_path}")

    print("\n  Detection Metrics")
    print("  " + "=" * 40)
    print(f"  {'TP (Live->Real)':<25s}: {tp}")
    print(f"  {'TN (Synthetic->Fake)':<25s}: {tn}")
    print(f"  {'FP (Live->Fake)':<25s}: {fp}")
    print(f"  {'FN (Synthetic->Real)':<25s}: {fn}")
    print("  " + "-" * 40)
    print(f"  {'Accuracy':<25s}: {accuracy:.4f}")
    print(f"  {'TPR (Sensitivity)':<25s}: {tpr:.4f}")
    print(f"  {'TNR (Specificity)':<25s}: {tnr:.4f}")
    print(f"  {'FPR':<25s}: {fpr:.4f}")
    print(f"  {'FNR':<25s}: {fnr:.4f}")
    print("  " + "=" * 40)

    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": accuracy, "tpr": tpr, "tnr": tnr, "fpr": fpr, "fnr": fnr,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Neuro-Pulse Batch Video Analyzer"
    )
    parser.add_argument(
        "--real_dir", type=str, required=True,
        help="Directory containing real (genuine) face videos",
    )
    parser.add_argument(
        "--fake_dir", type=str, required=True,
        help="Directory containing fake (deepfake) face videos",
    )
    parser.add_argument(
        "--threshold", type=float, default=3.0,
        help="SNR threshold for liveness classification (default: 3.0)",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/batch_results.csv",
        help="Output CSV path (default: outputs/batch_results.csv)",
    )
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)
    logging.basicConfig(
        filename="outputs/errors.log",
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("Neuro-Pulse Batch Analyzer")
    print("=" * 50)

    all_results: List[Dict] = []

    if os.path.isdir(args.real_dir):
        real_results = analyze_directory(args.real_dir, label="REAL", snr_threshold=args.threshold)
        all_results.extend(real_results)
    else:
        print(f"  WARNING: Real directory not found: {args.real_dir}")

    if os.path.isdir(args.fake_dir):
        fake_results = analyze_directory(args.fake_dir, label="FAKE", snr_threshold=args.threshold)
        all_results.extend(fake_results)
    else:
        print(f"  WARNING: Fake directory not found: {args.fake_dir}")

    if all_results:
        metrics = compute_metrics(all_results)
    else:
        print("\n  No results to compute metrics from.")
