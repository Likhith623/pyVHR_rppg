import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

"""
Neuro-Pulse Streamlit Dashboard.

Interactive web dashboard for deepfake detection with three modes:
  1. Live Webcam  — rPPG liveness detection (real face vs photo/screen/mask)
  2. Analyse Video — ML deepfake classifier (rPPG + visual artifact features)
  3. Analyse Image — Face detection + static analysis (images lack rPPG data)
"""

import time
import tempfile
from typing import Optional, Dict

import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import mediapipe as mp

from src.roi_extractor import extract_roi_green_multi, visualize_roi
from src.signal_processor import process_signal_buffer
from src.deepfake_detector import classify_video as ml_classify_video
from src.media_classifier import classify_media

# ──────────────────────────────────────────────
# Page Configuration (MUST be first st call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Neuro-Pulse",
    layout="wide",
    page_icon="\U0001f9e0",
)

# ──────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False
if "green_buffer" not in st.session_state:
    st.session_state.green_buffer = []
if "roi_buffers" not in st.session_state:
    st.session_state.roi_buffers = {"forehead": [], "left_cheek": [], "right_cheek": []}
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "hr_history" not in st.session_state:
    st.session_state.hr_history = []
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0

mp_face_mesh = mp.solutions.face_mesh


# ──────────────────────────────────────────────
# Component Renderers
# ──────────────────────────────────────────────
def render_bvp_waveform(container, green_buffer: list) -> None:
    import pandas as pd
    if len(green_buffer) > 0:
        data = green_buffer[-180:]
        df = pd.DataFrame({"BVP Signal": data})
        container.line_chart(df)
    else:
        container.info("Waiting for BVP data...")


def render_fft_spectrum(container, last_result: Optional[Dict]) -> None:
    if last_result is None or "freqs" not in last_result:
        container.info("Waiting for FFT data...")
        return

    freqs = np.array(last_result["freqs"])
    psd = np.array(last_result["psd"])
    hr_bpm = last_result["hr_bpm"]
    peak_hz = hr_bpm / 60.0

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(freqs, psd, color="#1f77b4", linewidth=1.2)

    mask = (freqs >= 0.7) & (freqs <= 4.0)
    ax.fill_between(freqs[mask], psd[mask], alpha=0.3, color="green", label="Cardiac Band")
    ax.axvline(peak_hz, color="red", linestyle="--", linewidth=1.5, label=f"Peak: {peak_hz:.2f} Hz")

    ax.annotate(
        f"{peak_hz:.2f} Hz\n({hr_bpm:.0f} BPM)",
        xy=(peak_hz, np.max(psd[mask]) if np.any(mask) else 0),
        xytext=(peak_hz + 0.3, np.max(psd[mask]) * 0.8 if np.any(mask) else 0),
        fontsize=9, color="red",
        arrowprops=dict(arrowstyle="->", color="red"),
    )

    ax.set_xlim(0.5, 5.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title("FFT Power Spectrum")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    container.pyplot(fig)
    plt.close(fig)


def render_liveness_metrics(last_result: Optional[Dict]) -> None:
    """Render metrics for webcam liveness mode."""
    if last_result is None:
        st.info("Waiting for analysis...")
        return

    conf = last_result["confidence_pct"]
    pq = last_result.get("peak_quality", 0.0)
    period = last_result.get("periodicity", 0.0)
    corr = last_result.get("roi_correlation", 0.0)
    prom = last_result.get("peak_prominence", 0.0)
    verdict = last_result["verdict"]

    color = "#2D7D4E" if verdict == "LIVE HUMAN" else "#CC2222"
    level = "LIVE" if verdict == "LIVE HUMAN" else "FAKE"

    st.markdown(f"**Confidence: {conf:.0f}% ({level})**")
    st.progress(min(conf / 100.0, 1.0))
    st.markdown(
        f"<style>div.stProgress > div > div > div {{ background-color: {color}; }}</style>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Peak Quality", f"{pq:.1f}", delta="Pass" if pq > 2.5 else "Fail")
    col2.metric("Periodicity", f"{period:.3f}", delta="Pass" if period > 0.15 else "Fail")
    col3.metric("ROI Correlation", f"{corr:.2f}",
                delta="Pass" if 0.15 < corr < 0.96 else "Fail")
    col4.metric("Peak Prominence", f"{prom:.2f}", delta="Pass" if prom > 0.5 else "Fail")


def render_deepfake_metrics(result: Dict) -> None:
    """Render metrics for video deepfake detection mode."""
    conf = result.get("confidence_pct", 0.0)
    verdict = result.get("verdict", "ERROR")
    method = result.get("method", "unknown")

    if verdict in ("REAL", "LIVE HUMAN"):
        color = "#2D7D4E"
        label = "REAL"
    elif verdict in ("FAKE", "SYNTHETIC"):
        color = "#CC2222"
        label = "FAKE / DEEPFAKE"
    else:
        color = "#E6A817"
        label = "ERROR"

    st.markdown(f"**Verdict: {label} ({conf:.0f}% confidence)** | Method: `{method}`")
    st.progress(min(conf / 100.0, 1.0))
    st.markdown(
        f"<style>div.stProgress > div > div > div {{ background-color: {color}; }}</style>",
        unsafe_allow_html=True,
    )

    feats = result.get("features", {})
    if feats:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Lap Ratio", f"{feats.get('lap_ratio', 0):.3f}")
        col2.metric("DCT HF", f"{feats.get('dct_hf', 0):.4f}")
        col3.metric("Noise Std", f"{feats.get('noise_std', 0):.3f}")
        col4.metric("Purity Diff", f"{feats.get('gr_purity_diff', 0):.3f}")


def render_verdict(last_result: Optional[Dict]) -> None:
    if last_result is None:
        st.warning("Awaiting sufficient frames for analysis...")
        return

    verdict = last_result.get("verdict", "ERROR")
    hr = last_result.get("hr_bpm", 0.0)

    if verdict == "LIVE HUMAN":
        st.success(f"LIVE HUMAN | Heart Rate: {hr:.0f} BPM")
    elif verdict == "REAL":
        st.success("REAL VIDEO — No deepfake detected")
    elif verdict in ("SYNTHETIC", "FAKE"):
        st.error("DETECTED: Potential Deepfake / Photo / Synthetic Content")
    else:
        st.warning(f"Analysis inconclusive: {verdict}")


def render_hr_history(hr_history: list) -> None:
    import pandas as pd
    if len(hr_history) > 0:
        data = hr_history[-60:]
        df = pd.DataFrame({"Heart Rate (BPM)": data})
        st.line_chart(df)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
st.sidebar.title("Neuro-Pulse")
mode = st.sidebar.radio(
    "Mode",
    ["Live Webcam", "Analyse Video", "Analyse Image"],
)

if mode == "Live Webcam":
    start_stop = st.sidebar.button(
        "Stop" if st.session_state.running else "Start"
    )
    if start_stop:
        st.session_state.running = not st.session_state.running
        if not st.session_state.running:
            st.session_state.green_buffer = []
            st.session_state.roi_buffers = {"forehead": [], "left_cheek": [], "right_cheek": []}
            st.session_state.last_result = None
            st.session_state.hr_history = []
            st.session_state.frame_count = 0

# ──────────────────────────────────────────────
# Main Layout
# ──────────────────────────────────────────────
st.title("Neuro-Pulse: rPPG Deepfake Detection")

if mode == "Live Webcam":
    st.caption("Liveness detection: Checks for a real heartbeat via rPPG. "
               "Detects photos, screens, masks, and printed images.")

    left_col, right_col = st.columns(2)
    camera_placeholder = left_col.empty()
    bvp_placeholder = right_col.empty()
    fft_placeholder = right_col.empty()

    if st.session_state.running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open webcam.")
            st.session_state.running = False
        else:
            last_roi = {"forehead": 0.0, "left_cheek": 0.0, "right_cheek": 0.0}
            with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                refine_landmarks=False,
            ) as face_mesh:
                while st.session_state.running:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Failed to read frame.")
                        break

                    h, w = frame.shape[:2]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb)

                    if results.multi_face_landmarks:
                        face_landmarks = results.multi_face_landmarks[0]
                        roi_vals = extract_roi_green_multi(frame, face_landmarks, h, w)
                        if roi_vals is not None:
                            st.session_state.green_buffer.append(roi_vals["combined"])
                            for key in ["forehead", "left_cheek", "right_cheek"]:
                                st.session_state.roi_buffers[key].append(roi_vals[key])
                                last_roi[key] = roi_vals[key]
                        frame = visualize_roi(frame, face_landmarks, h, w)
                    else:
                        st.session_state.green_buffer.append(
                            st.session_state.green_buffer[-1] if st.session_state.green_buffer else 0.0
                        )
                        for key in ["forehead", "left_cheek", "right_cheek"]:
                            st.session_state.roi_buffers[key].append(last_roi[key])

                    st.session_state.frame_count += 1

                    if len(st.session_state.green_buffer) >= 150:
                        result = process_signal_buffer(
                            st.session_state.green_buffer,
                            webcam_fps=30.0,
                            roi_buffers=st.session_state.roi_buffers,
                        )
                        if result is not None:
                            st.session_state.last_result = result
                            if result["verdict"] == "LIVE HUMAN":
                                st.session_state.hr_history.append(result["hr_bpm"])
                                st.session_state.hr_history = st.session_state.hr_history[-60:]

                    if st.session_state.frame_count % 15 == 0:
                        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        camera_placeholder.image(display_frame, channels="RGB", use_column_width=True)

                        with bvp_placeholder.container():
                            render_bvp_waveform(st, st.session_state.green_buffer)

                        with fft_placeholder.container():
                            render_fft_spectrum(st, st.session_state.last_result)

            cap.release()

    render_liveness_metrics(st.session_state.last_result)
    render_verdict(st.session_state.last_result)

    if st.session_state.hr_history:
        st.subheader("Heart Rate History")
        render_hr_history(st.session_state.hr_history)


elif mode == "Analyse Video":
    st.caption("Deepfake detection: Analyses rPPG physiological features and "
               "visual artifacts (Laplacian, DCT, noise patterns) to detect face-swap deepfakes.")

    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=["mp4", "mov", "avi", "mkv"],
    )

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        left_col, right_col = st.columns(2)

        # Show a preview frame
        cap = cv2.VideoCapture(tmp_path)
        ret, preview_frame = cap.read()
        if ret:
            display = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
            left_col.image(display, channels="RGB", use_column_width=True)
        cap.release()

        with st.spinner("Analysing video with ML classifier (rPPG + visual features)..."):
            result = ml_classify_video(tmp_path)

        render_deepfake_metrics(result)
        render_verdict(result)

        # Also extract BVP for visualisation
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        green_buffer = []
        roi_buffers_local = {"forehead": [], "left_cheek": [], "right_cheek": []}
        last_roi = {"forehead": 0.0, "left_cheek": 0.0, "right_cheek": 0.0}

        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=False,
        ) as face_mesh:
            count = 0
            while count < 300:
                ret, frame = cap.read()
                if not ret:
                    break
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                if results.multi_face_landmarks:
                    fl = results.multi_face_landmarks[0]
                    roi_vals = extract_roi_green_multi(frame, fl, h, w)
                    if roi_vals is not None:
                        green_buffer.append(roi_vals["combined"])
                        for key in ["forehead", "left_cheek", "right_cheek"]:
                            roi_buffers_local[key].append(roi_vals[key])
                            last_roi[key] = roi_vals[key]
                count += 1
        cap.release()

        if green_buffer:
            render_bvp_waveform(right_col, green_buffer)
            if len(green_buffer) >= 150:
                rppg_result = process_signal_buffer(
                    green_buffer, webcam_fps=fps, roi_buffers=roi_buffers_local
                )
                if rppg_result:
                    render_fft_spectrum(right_col, rppg_result)

        os.unlink(tmp_path)


elif mode == "Analyse Image":
    st.caption("Image analysis: Still images lack temporal rPPG data. "
               "We check for face presence and report that images cannot be "
               "verified as live.")

    uploaded_file = st.file_uploader(
        "Upload an image file",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
    )

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        left_col, right_col = st.columns(2)

        # Show the image
        frame = cv2.imread(tmp_path)
        if frame is not None:
            display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            left_col.image(display, channels="RGB", use_column_width=True)

        with st.spinner("Analysing image..."):
            result = classify_media(tmp_path)

        verdict = result.get("verdict", "ERROR")
        reason = result.get("reason", "")
        face_rate = result.get("face_detection_rate", 0.0)

        if verdict == "SYNTHETIC":
            st.error(f"SYNTHETIC: {reason}")
        else:
            st.success(f"Result: {verdict}")

        right_col.markdown("### Analysis Details")
        right_col.markdown(f"- **Verdict**: {verdict}")
        right_col.markdown(f"- **Reason**: {reason}")
        right_col.markdown(f"- **Face detected**: {'Yes' if face_rate > 0 else 'No'}")
        right_col.info(
            "Still images cannot contain rPPG (heartbeat) signals. "
            "Any single image — real or fake — is classified as SYNTHETIC "
            "because liveness requires temporal video data."
        )

        os.unlink(tmp_path)


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown("*Neuro-Pulse | SRM AP University | CSE Section I*")
