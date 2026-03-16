import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

"""
Neuro-Pulse Real-Time Pipeline.

Captures webcam frames (or video file), extracts per-ROI green-channel
BVP signals, processes through the multi-criteria signal pipeline,
and overlays live heart rate, SNR, ROI correlation, and liveness verdict.

Enhanced liveness detection uses four physics-based criteria:
  1. SNR — cardiac band signal power vs out-of-band noise
  2. Multi-ROI correlation — correlated cardiac signals prove blood flow
  3. Spectral peak quality — sharp peak = real heartbeat
  4. Signal strength — characteristic rPPG amplitude
"""

import argparse
import time
from collections import deque
from typing import Optional, Dict

import cv2
import numpy as np
import mediapipe as mp

from src.roi_extractor import extract_roi_green_multi, visualize_roi
from src.signal_processor import process_signal_buffer

mp_face_mesh = mp.solutions.face_mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Neuro-Pulse Real-Time Deepfake Detection Pipeline"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source: '0' for webcam, or path to video file (default: '0')",
    )
    parser.add_argument(
        "--threshold", type=float, default=3.0,
        help="SNR threshold for liveness classification in dB (default: 3.0)",
    )
    return parser.parse_args()


def overlay_text(
    frame: np.ndarray,
    text: str,
    position: tuple,
    color: tuple = (255, 255, 255),
    scale: float = 0.7,
    thickness: int = 2,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    cv2.rectangle(
        frame, (x, y - text_h - 5), (x + text_w + 5, y + baseline + 5),
        (0, 0, 0), cv2.FILLED
    )
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def open_camera(source):
    """Open camera with macOS M2 compatibility fallbacks."""
    attempts = [
        lambda: cv2.VideoCapture(source),
        lambda: cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION),
        lambda: cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION),
        lambda: cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION),
    ]
    for i, attempt in enumerate(attempts):
        try:
            cap = attempt()
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"  Camera opened (attempt {i+1})")
                    return cap
                cap.release()
        except Exception:
            continue
    print("ERROR: Cannot open camera. Tried all backends.")
    sys.exit(1)


def run_pipeline(source, threshold: float = 3.0) -> None:
    # Per-ROI rolling buffers for correlation analysis
    MAXLEN = 150  # 5 seconds at 30 FPS
    green_buffer: deque = deque(maxlen=MAXLEN)
    red_buffer: deque = deque(maxlen=MAXLEN)  # NEW: Red channel for anti-spoofing
    roi_buffers: Dict[str, deque] = {
        "forehead":    deque(maxlen=MAXLEN),
        "left_cheek":  deque(maxlen=MAXLEN),
        "right_cheek": deque(maxlen=MAXLEN),
    }

    # Open camera
    if isinstance(source, int):
        cap = open_camera(source)
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"ERROR: Cannot open video source: {source}")
            sys.exit(1)

    cap.set(cv2.CAP_PROP_FPS, 30)
    fps_actual = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"Neuro-Pulse Real-Time Pipeline")
    print(f"  Source    : {source}")
    print(f"  FPS       : {fps_actual}")
    print(f"  Threshold : {threshold} dB")
    print(f"  Press 'q' to quit.\n")

    last_result: Optional[Dict] = None
    frame_count = 0
    missing_face_frames = 0
    t_start = time.time()
    last_roi: Dict[str, float] = {"forehead": 0.0, "left_cheek": 0.0, "right_cheek": 0.0}

    verdict_history: deque = deque(maxlen=5)
    conf_history: deque = deque(maxlen=5)
    snr_history: deque = deque(maxlen=5)
    corr_history: deque = deque(maxlen=5)

    try:
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=False,
        ) as face_mesh:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(source, int):
                        continue  # retry for webcam
                    break  # end of video file

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]

                    # Extract per-ROI green values
                    roi_vals = extract_roi_green_multi(frame, face_landmarks, h, w)

                    missing_face_frames = 0
                    if roi_vals is not None:
                        green_buffer.append(roi_vals["combined"])
                        red_buffer.append(roi_vals.get("red_combined", 0.0))
                        for key in ["forehead", "left_cheek", "right_cheek"]:
                            roi_buffers[key].append(roi_vals[key])
                            last_roi[key] = roi_vals[key]
                    else:
                        # Face found but ROI empty — carry forward
                        green_buffer.append(green_buffer[-1] if green_buffer else 0.0)
                        red_buffer.append(red_buffer[-1] if red_buffer else 0.0)
                        for key in ["forehead", "left_cheek", "right_cheek"]:
                            roi_buffers[key].append(last_roi[key])

                    frame = visualize_roi(frame, face_landmarks, h, w)
                else:
                    # No face detected
                    missing_face_frames += 1
                    green_buffer.append(green_buffer[-1] if green_buffer else 0.0)
                    red_buffer.append(red_buffer[-1] if red_buffer else 0.0)
                    for key in ["forehead", "left_cheek", "right_cheek"]:
                        roi_buffers[key].append(last_roi[key])
                    overlay_text(
                        frame, "No face detected", (10, 30),
                        color=(0, 165, 255)
                    )

                    if missing_face_frames > 15:
                        green_buffer.clear()
                        for k in roi_buffers:
                            roi_buffers[k].clear()
                        last_result = None
                        verdict_history.clear()

                # Process signal if we have enough samples
                if len(green_buffer) >= 150:
                    roi_dict = {k: list(v) for k, v in roi_buffers.items()}
                    result = process_signal_buffer(
                        list(green_buffer),
                        webcam_fps=fps_actual,
                        roi_buffers=roi_dict,
                        threshold=threshold,
                        red_buffer=list(red_buffer)
                    )
                    if result is not None:
                        last_result = result
                        verdict_history.append(result["verdict"])
                        conf_history.append(result["confidence_pct"])
                        snr_history.append(result["snr_db"])
                        corr_history.append(result.get("roi_correlation", 0.0))

                # Overlay results
                if last_result is not None:
                    verdict = max(verdict_history, key=verdict_history.count) if verdict_history else last_result["verdict"]
                    hr = last_result["hr_bpm"]
                    snr = float(np.mean(snr_history)) if snr_history else last_result["snr_db"]
                    conf = float(np.mean(conf_history)) if conf_history else last_result["confidence_pct"]
                    corr = float(np.mean(corr_history)) if corr_history else last_result.get("roi_correlation", 0.0)
                    pq = last_result.get("peak_quality", 0.0)

                    verdict_color = (0, 255, 0) if verdict == "LIVE HUMAN" else (0, 0, 255)

                    overlay_text(frame, f"HR: {hr:.0f} BPM", (10, h - 150), color=(255, 255, 255))
                    overlay_text(frame, f"SNR: {snr:.1f} dB", (10, h - 120), color=(255, 255, 255))
                    
                    # Display Purity instead of just peak quality
                    purity = last_result.get("spectral_purity", 0.0)
                    overlay_text(frame, f"Purity: {purity:.2%}", (10, h - 90), color=(255, 255, 255))
                    
                    overlay_text(frame, f"ROI Corr: {corr:.2f}", (10, h - 60), color=(255, 255, 255))
                    overlay_text(frame, f"{verdict} ({conf:.0f}%)", (10, h - 30), color=verdict_color)
                else:
                    n = len(green_buffer)
                    overlay_text(
                        frame, f"Analysing ({n}/150)", (10, h - 30),
                        color=(0, 255, 255)
                    )

                cv2.imshow("Neuro-Pulse", frame)
                frame_count += 1

                if frame_count % 150 == 0:
                    elapsed = time.time() - t_start
                    avg_fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"  Frames: {frame_count} | Avg FPS: {avg_fps:.1f}")
                    if last_result:
                        print(f"    SNR={last_result['snr_db']:.1f}dB  "
                              f"Corr={last_result.get('roi_correlation',0):.2f}  "
                              f"PeakQ={last_result.get('peak_quality',0):.1f}  "
                              f"→ {last_result['verdict']}")

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        elapsed = time.time() - t_start
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"\nDone. {frame_count} frames in {elapsed:.1f}s ({avg_fps:.1f} FPS)")


if __name__ == "__main__":
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    run_pipeline(source=source, threshold=args.threshold)
