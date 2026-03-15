import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

"""
Neuro-Pulse Streamlit Dashboard.

Interactive web dashboard for real-time and batch deepfake detection
using multi-criteria rPPG signal analysis. Displays live webcam feed
with ROI overlay, BVP waveform, FFT spectrum, multi-criteria confidence
meter, and liveness verdict.
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
from src.batch_analyzer import analyze_video

# ──────────────────────────────────────────────
# Page Configuration (MUST be first st call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Neuro-Pulse",
    layout="wide",
    page_icon="\U0001f9e0",
)

# ──────────────────────────────────────────────
# Session State Initialisation
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
    if last_result is None:
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


def render_confidence_meter(last_result: Optional[Dict]) -> None:
    if last_result is None:
        st.info("Waiting for analysis...")
        return

    snr = last_result["snr_db"]
    conf = last_result["confidence_pct"]
    corr = last_result.get("roi_correlation", 0.0)
    pq = last_result.get("peak_quality", 0.0)
    ss = last_result.get("signal_strength", 0.0)
    verdict = last_result["verdict"]

    if verdict == "SYNTHETIC":
        color = "#CC2222"
        level = "FAKE"
    elif conf < 50:
        color = "#E6A817"
        level = "LOW"
    else:
        color = "#2D7D4E"
        level = "HIGH"

    st.markdown(f"**Confidence: {conf:.0f}% ({level})**")
    st.progress(min(conf / 100.0, 1.0))
    st.markdown(
        f"<style>div.stProgress > div > div > div {{ background-color: {color}; }}</style>",
        unsafe_allow_html=True,
    )

    # Show all 4 criteria
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("SNR", f"{snr:.1f} dB", delta="Pass" if snr > 3.0 else "Fail")
    col2.metric("ROI Correlation", f"{corr:.2f}", delta="Pass" if corr > 0.4 else "Fail")
    col3.metric("Peak Quality", f"{pq:.1f}", delta="Pass" if pq > 2.0 else "Fail")
    col4.metric("Signal Strength", f"{ss:.2f}", delta="Pass" if ss > 0.15 else "Fail")


def render_verdict(last_result: Optional[Dict]) -> None:
    if last_result is None:
        st.warning("Awaiting sufficient frames for analysis...")
        return

    verdict = last_result["verdict"]
    hr = last_result["hr_bpm"]

    if verdict == "LIVE HUMAN":
        st.success(f"LIVE HUMAN | Heart Rate: {hr:.0f} BPM")
    else:
        st.error("SYNTHETIC — Potential Deepfake / Photo / Video Detected")


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
mode = st.sidebar.radio("Mode", ["Live Webcam", "Analyse Video File"])
threshold = st.sidebar.slider("SNR Threshold (dB)", 1.0, 6.0, 3.0, 0.1)

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

left_col, right_col = st.columns(2)

if mode == "Live Webcam":
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

    render_confidence_meter(st.session_state.last_result)
    render_verdict(st.session_state.last_result)

    if st.session_state.hr_history:
        st.subheader("Heart Rate History")
        render_hr_history(st.session_state.hr_history)

elif mode == "Analyse Video File":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("Analysing video..."):
            result = analyze_video(tmp_path, snr_threshold=threshold)

        st.session_state.last_result = result

        # Read frames for BVP display
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
            last_frame = None
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
                    last_frame = visualize_roi(frame, fl, h, w)
                count += 1
        cap.release()

        st.session_state.green_buffer = green_buffer

        if last_frame is not None:
            display = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
            left_col.image(display, channels="RGB", use_column_width=True)

        if len(green_buffer) >= 150:
            full_result = process_signal_buffer(
                green_buffer, webcam_fps=fps, roi_buffers=roi_buffers_local
            )
            if full_result is not None:
                st.session_state.last_result = full_result

        render_bvp_waveform(right_col, st.session_state.green_buffer)
        render_fft_spectrum(right_col, st.session_state.last_result)
        render_confidence_meter(st.session_state.last_result)
        render_verdict(st.session_state.last_result)

        os.unlink(tmp_path)

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown("*Neuro-Pulse | SRM AP University | CSE Section I*")
