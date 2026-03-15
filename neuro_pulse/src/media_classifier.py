import os
import sys
import argparse
from typing import Dict

import cv2
import mediapipe as mp

from src.batch_analyzer import analyze_video


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def _has_face(frame) -> bool:
    """Detects whether a face exists in a single image frame."""
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=True,
    ) as face_mesh:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        return bool(result.multi_face_landmarks)


def classify_media(path: str, threshold: float = 3.0) -> Dict:
    """Classify a single media file (video or image).

    Videos are routed through the full rPPG pipeline. Images cannot contain
    temporal rPPG, so any still image is deemed synthetic but we still check
    for the presence of a face to provide a clear reason.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()

    if ext in VIDEO_EXTS:
        return analyze_video(path, snr_threshold=threshold)

    if ext in IMAGE_EXTS:
        frame = cv2.imread(path)
        if frame is None:
            raise ValueError(f"Could not read image: {path}")

        has_face = _has_face(frame)
        verdict = "SYNTHETIC"
        reason = "Still image has no temporal rPPG information"
        if not has_face:
            reason += "; face not detected"

        return {
            "filename": os.path.basename(path),
            "verdict": verdict,
            "reason": reason,
            "snr_db": 0.0,
            "hr_bpm": 0.0,
            "confidence_pct": 0.0,
            "roi_correlation": 0.0,
            "peak_quality": 0.0,
            "signal_strength": 0.0,
            "frames_processed": 1,
            "fps_estimated": 0.0,
            "face_detection_rate": 1.0 if has_face else 0.0,
        }

    raise ValueError(f"Unsupported file type: {ext}")


def main():
    parser = argparse.ArgumentParser(description="Neuro-Pulse Media Classifier (video or image)")
    parser.add_argument("--path", required=True, help="Path to video or image file")
    parser.add_argument("--threshold", type=float, default=3.0, help="SNR threshold (dB) for video classification")
    args = parser.parse_args()

    result = classify_media(args.path, threshold=args.threshold)
    print(result)


if __name__ == "__main__":
    main()
