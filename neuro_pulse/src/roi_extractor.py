import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

"""
Neuro-Pulse ROI Extractor Module.

Extracts facial Regions of Interest (ROI) using MediaPipe FaceMesh
and computes the mean green-channel intensity for rPPG signal extraction.

The three ROIs (forehead, left cheek, right cheek) are selected because
they exhibit the strongest blood volume pulse (BVP) signal due to
high capillary density and minimal hair occlusion.
"""

from typing import Optional, List, Tuple, Dict

import cv2
import numpy as np
import mediapipe as mp
import importlib

# ──────────────────────────────────────────────
# MediaPipe FaceMesh Configuration
# ──────────────────────────────────────────────
# Locate MediaPipe FaceMesh module in a robust way. Try several import
# strategies because different mediapipe wheels expose the package
# structure differently on various platforms.
try:
    mp_face_mesh = mp.solutions.face_mesh
except Exception:
    try:
        mp_face_mesh = importlib.import_module("mediapipe.solutions.face_mesh")
    except Exception:
        try:
            mp_face_mesh = importlib.import_module("mediapipe.python.solutions.face_mesh")
        except Exception as e:
            raise ImportError(
                "Could not import MediaPipe FaceMesh (mediapipe.solutions.face_mesh). "
                "Ensure mediapipe is installed correctly (try `pip install mediapipe`)."
            ) from e

# ──────────────────────────────────────────────
# ROI Landmark Indices (MediaPipe 468-point mesh)
# ──────────────────────────────────────────────
FOREHEAD_IDX: List[int] = [10, 338, 297, 332, 284, 251, 389]
LEFT_CHEEK_IDX: List[int] = [234, 93, 132, 58, 172]
RIGHT_CHEEK_IDX: List[int] = [454, 323, 361, 288, 397]


def _get_roi_points(
    face_landmarks, indices: List[int], h: int, w: int
) -> np.ndarray:
    """Convert landmark indices to pixel coordinates.

    Args:
        face_landmarks: MediaPipe face landmarks object.
        indices: List of landmark indices for the ROI.
        h: Frame height in pixels.
        w: Frame width in pixels.

    Returns:
        NumPy array of shape (N, 2) with integer pixel coordinates.
    """
    points = []
    for idx in indices:
        lm = face_landmarks.landmark[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append([x, y])
    return np.array(points, dtype=np.int32)


def extract_roi_green(
    frame: np.ndarray,
    face_landmarks,
    h: int,
    w: int,
) -> Optional[float]:
    """Extract the mean green-channel intensity from three facial ROIs.

    For each ROI (forehead, left cheek, right cheek), a convex-polygon
    mask is created and the mean of the green channel (BGR index 1) is
    computed. The final value is the average of the three ROI means
    minus the top-left 40x40 background patch mean, which helps remove
    ambient illumination drift.

    Args:
        frame: BGR image as a NumPy array of shape (H, W, 3).
        face_landmarks: MediaPipe face landmarks for a single face.
        h: Frame height in pixels.
        w: Frame width in pixels.

    Returns:
        Float green-channel value (background-subtracted), or None if
        any ROI mask is empty.
    """
    roi_indices = [FOREHEAD_IDX, LEFT_CHEEK_IDX, RIGHT_CHEEK_IDX]
    green_channel = frame[:, :, 1]  # BGR index 1 = Green
    roi_means: List[float] = []

    for indices in roi_indices:
        pts = _get_roi_points(face_landmarks, indices, h, w)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts, 255)

        pixels = green_channel[mask == 255]
        if len(pixels) == 0:
            return None

        roi_means.append(float(np.mean(pixels)))

    # Background reference: top-left 40x40 corner
    bg_patch = green_channel[0:40, 0:40]
    bg_mean = float(np.mean(bg_patch)) if bg_patch.size > 0 else 0.0

    avg_roi = float(np.mean(roi_means))
    return avg_roi - bg_mean


def extract_roi_green_multi(
    frame: np.ndarray,
    face_landmarks,
    h: int,
    w: int,
) -> Optional[Dict[str, float]]:
    """Extract per-ROI green-channel intensity for multi-ROI correlation.

    Returns individual green values for forehead, left cheek, right cheek
    plus the combined (background-subtracted) average. This enables
    inter-ROI correlation analysis for liveness detection.

    Args:
        frame: BGR image as a NumPy array of shape (H, W, 3).
        face_landmarks: MediaPipe face landmarks for a single face.
        h: Frame height in pixels.
        w: Frame width in pixels.

    Returns:
        Dict with keys: 'forehead', 'left_cheek', 'right_cheek', 'combined'.
        Returns None if any ROI mask is empty.
    """
    roi_map = {
        "forehead": FOREHEAD_IDX,
        "left_cheek": LEFT_CHEEK_IDX,
        "right_cheek": RIGHT_CHEEK_IDX,
    }
    green_channel = frame[:, :, 1]
    roi_values: Dict[str, float] = {}

    for name, indices in roi_map.items():
        pts = _get_roi_points(face_landmarks, indices, h, w)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts, 255)
        pixels = green_channel[mask == 255]
        if len(pixels) == 0:
            return None
        roi_values[name] = float(np.mean(pixels))

    bg_patch = green_channel[0:40, 0:40]
    bg_mean = float(np.mean(bg_patch)) if bg_patch.size > 0 else 0.0

    avg_roi = float(np.mean(list(roi_values.values())))
    roi_values["combined"] = avg_roi - bg_mean
    
    # NEW RESEARCH ADDITION: Red Channel Reference for Anti-Spoofing
    # Real heartbeats exist in Green channel. Screens/Autofocus pulse in all RGB channels.
    red_channel = frame[:, :, 2] # BGR index 2 = Red
    red_means = []
    for name, indices in roi_map.items():
        pts = _get_roi_points(face_landmarks, indices, h, w)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts, 255)
        pixels = red_channel[mask == 255]
        if len(pixels) > 0:
            red_means.append(float(np.mean(pixels)))
    roi_values["red_combined"] = float(np.mean(red_means)) if red_means else 0.0

    return roi_values


def normalize_signal(signal_array: list) -> list:
    """Apply Z-score normalisation to the most recent samples.

    Uses the last min(len(signal_array), 256) samples to compute the
    mean and standard deviation, then returns the Z-scored values.

    Args:
        signal_array: List of float signal values.

    Returns:
        List of Z-score normalised float values.
    """
    if len(signal_array) == 0:
        return []

    window_size = min(len(signal_array), 256)
    window = np.array(signal_array[-window_size:], dtype=np.float64)
    mean = np.mean(window)
    std = np.std(window)

    if std < 1e-10:
        return [0.0] * len(window)

    normalised = ((window - mean) / std).tolist()
    return normalised


def visualize_roi(
    frame: np.ndarray,
    face_landmarks,
    h: int,
    w: int,
) -> np.ndarray:
    """Draw the three ROI polygons on a copy of the input frame.

    Draws the forehead, left cheek, and right cheek regions as green
    polygon outlines on the frame for visual debugging.

    Args:
        frame: BGR image as a NumPy array of shape (H, W, 3).
        face_landmarks: MediaPipe face landmarks for a single face.
        h: Frame height in pixels.
        w: Frame width in pixels.

    Returns:
        Annotated copy of the frame with ROI polygons drawn.
    """
    annotated = frame.copy()
    color = (0, 255, 0)  # Green in BGR

    for indices in [FOREHEAD_IDX, LEFT_CHEEK_IDX, RIGHT_CHEEK_IDX]:
        pts = _get_roi_points(face_landmarks, indices, h, w)
        cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)

    return annotated


# ──────────────────────────────────────────────
# Main: webcam demo
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Neuro-Pulse ROI Extractor — Webcam Demo")
    print("Press 'q' to quit.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        sys.exit(1)

    green_buffer: List[float] = []
    frame_count = 0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=False,
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame.")
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                green_val = extract_roi_green(frame, face_landmarks, h, w)

                if green_val is not None:
                    green_buffer.append(green_val)

                frame = visualize_roi(frame, face_landmarks, h, w)
            else:
                green_val = None

            frame_count += 1

            # Print stats every 30 frames
            if frame_count % 30 == 0:
                if len(green_buffer) > 0:
                    recent = green_buffer[-30:]
                    norm = normalize_signal(green_buffer)
                    print(
                        f"Frame {frame_count} | "
                        f"Buffer: {len(green_buffer)} | "
                        f"Green mean: {np.mean(recent):.2f} | "
                        f"Green std: {np.std(recent):.4f} | "
                        f"Norm[-1]: {norm[-1]:.4f}"
                    )
                else:
                    print(f"Frame {frame_count} | No face detected yet.")

            cv2.imshow("Neuro-Pulse ROI Extractor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Total frames: {frame_count}, Buffer size: {len(green_buffer)}")
