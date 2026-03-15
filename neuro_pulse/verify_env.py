#!/usr/bin/env python3
"""
Neuro-Pulse Environment Verification Script.

Imports all required libraries, prints their versions,
and runs a 5-frame webcam capture test.
"""

import sys


def check_imports() -> bool:
    """Import all required libraries and print their versions."""
    print("=" * 50)
    print("  Neuro-Pulse: Environment Verification")
    print("=" * 50)
    print()

    all_ok = True
    libraries = [
        ("cv2", "opencv-python"),
        ("mediapipe", "mediapipe"),
        ("scipy", "scipy"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("streamlit", "streamlit"),
        ("sklearn", "scikit-learn"),
        ("tqdm", "tqdm"),
    ]

    for module_name, pip_name in libraries:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "N/A")
            print(f"  [OK] {pip_name:20s} -> {version}")
        except ImportError:
            print(f"  [FAIL] {pip_name:20s} -> NOT INSTALLED")
            all_ok = False

    print()
    return all_ok


def webcam_test(num_frames: int = 5) -> bool:
    """Capture num_frames frames from the default webcam.

    Args:
        num_frames: Number of frames to capture (default 5).

    Returns:
        True if all frames were captured successfully, False otherwise.
    """
    import cv2

    print(f"Webcam Test: Capturing {num_frames} frames...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [WARN] Could not open webcam (index 0).")
        print("  This is expected in headless / CI environments.")
        cap.release()
        return False

    captured = 0
    for i in range(num_frames):
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"  Frame {i + 1}/{num_frames}: {w}x{h}")
            captured += 1
        else:
            print(f"  Frame {i + 1}/{num_frames}: FAILED to read")

    cap.release()

    success = captured == num_frames
    print(f"  Captured {captured}/{num_frames} frames.")
    return success


def main() -> None:
    """Run all verification checks and print PASS/FAIL."""
    imports_ok = check_imports()

    if not imports_ok:
        print("=" * 50)
        print("  RESULT: FAIL (missing libraries)")
        print("=" * 50)
        sys.exit(1)

    webcam_ok = webcam_test()

    print()
    print("=" * 50)
    if imports_ok and webcam_ok:
        print("  RESULT: PASS (all libraries + webcam OK)")
    elif imports_ok:
        print("  RESULT: PASS (all libraries OK, webcam unavailable)")
    else:
        print("  RESULT: FAIL")
    print("=" * 50)

    sys.exit(0 if imports_ok else 1)


if __name__ == "__main__":
    main()
