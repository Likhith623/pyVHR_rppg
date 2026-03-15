#!/usr/bin/env python3
"""
=============================================================================
NEURO-PULSE COMPLETE SETUP: Dataset Download + Project Structure + AI Prompts
SRM AP University | CSE Section I
=============================================================================
HOW TO RUN:
  python neuro_pulse_dataset_setup.py              # full run
  python neuro_pulse_dataset_setup.py --skip-download   # skip download
  python neuro_pulse_dataset_setup.py --prompts-only    # prompts only
=============================================================================
"""
import os, sys, shutil, argparse, subprocess
from pathlib import Path

BASE_DIR     = Path(__file__).parent.resolve()
FF_SCRIPT    = BASE_DIR / "download_FaceForensics.py"
DOWNLOAD_DIR = BASE_DIR / "ff_downloads"
PROJECT_DIR  = BASE_DIR / "neuro_pulse"
REAL_DIR     = PROJECT_DIR / "datasets" / "real"
FAKE_DIR     = PROJECT_DIR / "datasets" / "synthetic"
OUTPUTS_DIR  = PROJECT_DIR / "outputs"
PROMPTS_DIR  = BASE_DIR / "ai_prompts"
SERVER, COMP, NVID = "EU2", "c40", "50"

def download_dataset():
    print("\n=== DOWNLOADING FACEFORENSICS++ (EU2 server, c40 compression) ===")
    if not FF_SCRIPT.exists():
        print(f"ERROR: Put download_FaceForensics.py in same folder as this file.\n  Expected: {FF_SCRIPT}")
        sys.exit(1)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for dataset, label in [("original","REAL"), ("Deepfakes","FAKE")]:
        input(f"\nPress ENTER to download {label} videos (dataset={dataset})...")
        cmd = [sys.executable, str(FF_SCRIPT), str(DOWNLOAD_DIR),
               "--dataset", dataset, "--compression", COMP,
               "--type", "videos", "--num_videos", NVID, "--server", SERVER]
        print("Running:", " ".join(str(x) for x in cmd))
        subprocess.run(cmd)
    print("\nDOWNLOAD DONE.")

def organise_files():
    print("\n=== ORGANISING FILES ===")
    for d in [REAL_DIR, FAKE_DIR, OUTPUTS_DIR,
              PROJECT_DIR/"src", PROJECT_DIR/"dashboard",
              PROJECT_DIR/"tests", PROJECT_DIR/"notebooks"]:
        d.mkdir(parents=True, exist_ok=True)
    for d in [REAL_DIR, FAKE_DIR, OUTPUTS_DIR, PROJECT_DIR/"notebooks"]:
        (d/".gitkeep").touch()
    for d in [PROJECT_DIR/"src", PROJECT_DIR/"dashboard", PROJECT_DIR/"tests"]:
        p = d/"__init__.py"
        if not p.exists(): p.write_text("# Neuro-Pulse\n")
    for src_path, dst in [
        (DOWNLOAD_DIR/"original_sequences"/"youtube"/"c40"/"videos", REAL_DIR),
        (DOWNLOAD_DIR/"manipulated_sequences"/"Deepfakes"/"c40"/"videos", FAKE_DIR)
    ]:
        count = 0
        if src_path.exists():
            for v in src_path.glob("*.mp4"):
                d = dst/v.name
                if not d.exists():
                    shutil.copy2(v, d); count += 1
            print(f"  Copied {count} videos to {dst}")
        else:
            print(f"  WARNING: Source not found: {src_path}")
            print(f"  Manually copy .mp4 files to: {dst}")
    r = len(list(REAL_DIR.glob("*.mp4")))
    f = len(list(FAKE_DIR.glob("*.mp4")))
    print(f"  Status: {r} real | {f} fake")
    print("STEP 3 DONE.")

def write_prompts():
    print("\n=== WRITING AI PROMPTS ===")
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    prompts = {}

    prompts["01_phase1_setup.txt"] = """\
I am building Neuro-Pulse, a deepfake detection system using rPPG.
Phase 1: Environment Setup. Generate these files completely:

1. requirements.txt:
   opencv-python==4.8.1, mediapipe==0.10.9, scipy==1.11.4, numpy==1.24.3,
   matplotlib==3.8.0, streamlit==1.29.0, scikit-learn, tqdm

2. setup.sh: conda create neuropulse python=3.10, activate, pip install -r requirements.txt

3. verify_env.py: import all libs, print versions, 5-frame webcam test, print PASS/FAIL

4. create_structure.py: create all neuro_pulse/ subdirs with __init__.py and .gitkeep files

5. README.md: title, abstract, install steps, 5-phase timeline table, run commands
"""

    prompts["02_phase2_roi_extractor.txt"] = """\
I am building Neuro-Pulse. Phase 2. Write: neuro_pulse/src/roi_extractor.py

TOP of file (before any imports):
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

Use MediaPipe 0.10.9 mp.solutions.face_mesh.FaceMesh:
  max_num_faces=1, min_detection_confidence=0.5,
  min_tracking_confidence=0.5, refine_landmarks=False

Constants:
  FOREHEAD_IDX    = [10, 338, 297, 332, 284, 251, 389]
  LEFT_CHEEK_IDX  = [234, 93, 132, 58, 172]
  RIGHT_CHEEK_IDX = [454, 323, 361, 288, 397]

Functions:
1. extract_roi_green(frame, face_landmarks, h, w) -> float or None
   cv2.fillConvexPoly mask per ROI, green channel (BGR index 1) mean per ROI,
   return average of 3 ROI means minus top-left 40x40 background mean.
   Return None if mask empty.

2. normalize_signal(signal_array: list) -> list
   Z-score on last min(len,256) samples. Return list.

3. visualize_roi(frame, face_landmarks, h, w) -> frame
   Draw 3 ROI polygons in green (0,255,0). Return annotated copy.

4. __main__: webcam loop, extract per frame, print stats every 30 frames, q to quit.

Full docstrings, type hints, graceful None on no face detected.
"""

    prompts["03_phase3_signal_processor.txt"] = """\
I am building Neuro-Pulse. Phase 3. Write: neuro_pulse/src/signal_processor.py

TOP (before imports):
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

Functions (implement EXACTLY):

1. butterworth_bandpass(signal, lowcut=0.7, highcut=4.0, fs=30.0, order=4)
   scipy.signal.detrend first. butter BPF. filtfilt. Return filtered array.

2. resample_signal(signal, original_fs=30.0, target_fs=256.0)
   scipy.interpolate.CubicSpline. Return resampled array at target_fs.

3. compute_psd_welch(signal, fs=256.0)
   scipy.signal.welch(signal, fs=fs, nperseg=256, noverlap=128). Return (freqs, psd).

4. compute_snr_and_hr(freqs, psd, low=0.7, high=4.0)
   mask = (freqs>=low)&(freqs<=high)
   signal_power = np.max(psd[mask])
   noise_power  = np.mean(psd[~mask])
   snr_db = 10*np.log10(signal_power/(noise_power+1e-10))
   hr_bpm = float(freqs[mask][np.argmax(psd[mask])]*60)
   Return (hr_bpm, snr_db)

5. classify_liveness(snr_db: float, threshold: float = 3.0)
   confidence_pct = float(min(100,max(0,(snr_db/10.0)*100)))
   Return ('LIVE HUMAN', confidence_pct) if snr_db > threshold else ('SYNTHETIC', confidence_pct)

6. process_signal_buffer(green_buffer: list, webcam_fps: float = 30.0)
   Return None if len<150. Pipeline: bandpass->resample->welch->snr_hr->classify.
   Return {hr_bpm, snr_db, verdict, confidence_pct, freqs:list, psd:list}

__main__: 1.2Hz sine + Gaussian noise, 300 samples. process_signal_buffer.
assert abs(hr_bpm-72)<=5. Print PASS or FAIL.
Full docstrings and type hints.
"""

    prompts["04a_phase4a_realtime_pipeline.txt"] = """\
I am building Neuro-Pulse. Phase 4A. Write: neuro_pulse/src/realtime_pipeline.py

TOP:
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

Import:
from src.roi_extractor import extract_roi_green, visualize_roi
from src.signal_processor import process_signal_buffer

1. collections.deque(maxlen=300) as rolling buffer.
2. argparse: --source default='0' (convert to int if digit), --threshold default=3.0
3. cap=cv2.VideoCapture(source). set FPS=30. Exit if not isOpened().
4. Init FaceMesh ONCE before loop (mp.solutions.face_mesh.FaceMesh), use as context manager.
5. Per frame:
   a. read frame; break if not ret
   b. convert BGR->RGB, process with face_mesh
   c. if face: extract_roi_green, append; frame=visualize_roi
   d. else: append last or 0.0; show 'No face detected' orange text
   e. if len>=150: process_signal_buffer, store as last_result
   f. overlay: if last_result show HR/SNR/verdict; else show 'Analysing (n/150)' yellow
   g. imshow, waitKey, q to quit
6. Every 150 frames: print avg FPS, warn if <20
7. cap.release()+destroyAllWindows() in finally block.
Full docstrings and error handling.
"""

    prompts["04b_phase4b_batch_analyzer.txt"] = """\
I am building Neuro-Pulse. Phase 4B. Write: neuro_pulse/src/batch_analyzer.py

TOP:
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

Import:
from src.roi_extractor import extract_roi_green
from src.signal_processor import process_signal_buffer

Init FaceMesh ONCE at module level (not inside functions):
import mediapipe as mp
_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1, min_detection_confidence=0.5,
    min_tracking_confidence=0.5, static_image_mode=False)

1. analyze_video(video_path, snr_threshold=3.0) -> dict
   cv2.VideoCapture, max 300 frames, extract_roi_green via _face_mesh.
   process_signal_buffer. Return {filename, verdict, snr_db, hr_bpm,
   confidence_pct, frames_processed, fps_estimated, face_detection_rate}.
   try/except: return verdict='ERROR' on failure.

2. analyze_directory(directory, label, snr_threshold=3.0)
   Glob .mp4 .mov .avi. analyze_video per file. Add 'ground_truth':label.
   tqdm if importable else print every 5. Return list of dicts.

3. compute_metrics(results) -> dict
   TP=verdict LIVE+truth REAL, TN=verdict SYNTHETIC+truth FAKE,
   FP=verdict LIVE+truth FAKE, FN=verdict SYNTHETIC+truth REAL.
   accuracy=(TP+TN)/total, tpr, tnr, fpr, fnr.
   Save CSV to outputs/batch_results.csv. Print table. Return dict.

4. __main__: --real_dir, --fake_dir, --threshold=3.0, --output=outputs/batch_results.csv
   Errors to outputs/errors.log
"""

    prompts["05_phase5_dashboard.txt"] = """\
I am building Neuro-Pulse. Phase 5. Write: neuro_pulse/dashboard/app.py

TOP:
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

Import:
from src.roi_extractor import extract_roi_green, visualize_roi
from src.signal_processor import process_signal_buffer
from src.batch_analyzer import analyze_video

FIRST st call: st.set_page_config(page_title='Neuro-Pulse', layout='wide', page_icon='🧠')

session_state: running=False, green_buffer=[], last_result=None, hr_history=[], frame_count=0

Sidebar: mode radio (Live Webcam/Analyse Video File), threshold slider 1.0-6.0 default 3.0, Start/Stop button.
Layout: 2 equal columns (left=camera+ROI, right=BVP+FFT). Below: meter+verdict+HR history.

LIVE MODE: open cap, init FaceMesh, loop frames: extract_roi_green, append to buffer.
Every 150 frames: process_signal_buffer, update last_result.
Every 15 frames: update all 5 components. On Stop: cap.release().

BATCH MODE: st.file_uploader, save temp, analyze_video, set last_result, render components.

COMPONENT 1 - BVP Waveform:
  pd.DataFrame({'BVP Signal': green_buffer[-180:]}) -> right_col.line_chart

COMPONENT 2 - FFT Spectrum:
  matplotlib: plot psd, shade 0.7-4Hz green, red dashed peak line,
  annotate Hz and BPM, xlim 0.5-5.0. right_col.pyplot(fig); plt.close(fig)

COMPONENT 3 - Confidence Meter:
  snr<3: red #CC2222 FAKE | 3-6: orange #E6A817 LOW | >=6: green #2D7D4E HIGH
  st.markdown bold SNR+confidence. st.progress(conf). CSS to colour bar.

COMPONENT 4 - Verdict: st.success (LIVE HUMAN + HR) or st.error (SYNTHETIC)

COMPONENT 5 - HR History: only when LIVE. append to hr_history[-60:]. st.line_chart.

Footer: st.markdown('*Neuro-Pulse | SRM AP University | CSE Section I*')
"""

    prompts["06_evaluator.txt"] = """\
I am building Neuro-Pulse. Final evaluation. Write: neuro_pulse/src/evaluator.py

TOP:
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

Import: from src.batch_analyzer import analyze_video

1. load_dataset(real_dir, fake_dir) -> list
   Glob .mp4 .mov .avi. Return [{path, ground_truth_label:'REAL'/'FAKE'}, ...]

2. run_evaluation(dataset, snr_threshold=3.0) -> dict
   analyze_video per file. TP/TN/FP/FN. accuracy, tpr, tnr, fpr, fnr.
   AUC via sklearn.metrics.roc_auc_score(labels, snr_scores) REAL=1 FAKE=0.
   Return metrics + per_file_results.

3. plot_roc_curve(per_file_results, save_path)
   sklearn.metrics.roc_curve, snr as score, REAL=1 FAKE=0.
   Mark 3.0 dB operating point as red dot. Save PNG.

4. plot_snr_distribution(per_file_results, save_path)
   Overlapping histograms alpha=0.6. Red dashed line at 3.0 dB 'Threshold'.
   Labels, legend, title. Save PNG.

5. generate_latex_table(metrics) -> str
   LaTeX table with accuracy, tpr, tnr, fpr, fnr, auc to 2 decimal places.
   Include caption 'Neuro-Pulse Detection Performance'.

6. generate_html_report(metrics, roc_path, dist_path) -> str
   Self-contained HTML, base64 images, styled metrics table.

__main__: --real_dir, --fake_dir, --threshold=3.0, --output_dir.
Run, save ROC PNG + distribution PNG + .tex + .html. Print accuracy.
"""

    prompts["07_run_demo.txt"] = """\
I am building Neuro-Pulse. Write: neuro_pulse/run_demo.py

TOP:
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

Import:
from src.roi_extractor import extract_roi_green, visualize_roi
from src.signal_processor import process_signal_buffer

Script:
1. Print 'NEURO-PULSE DEMONSTRATION — SRM AP University'
2. Open webcam 0. If fails: print error, exit.
3. Init MediaPipe FaceMesh.
4. Collect 300 frames: show cv2 window with ROI overlay + frame counter.
   Append green values. Print 'Frame X/300' every 60 frames.
5. process_signal_buffer(buffer).
6. Print:
   ==========================================
   NEURO-PULSE RESULT
   ==========================================
   Verdict:     [LIVE HUMAN or SYNTHETIC]
   Heart Rate:  XX BPM
   SNR:         X.X dB
   Confidence:  XX%
   ==========================================
7. If LIVE HUMAN: print 'SYSTEM WORKING CORRECTLY'
   If SYNTHETIC:  print 'WARNING: Improve lighting or stay still'
8. cap.release(), destroyAllWindows()

Run: python neuro_pulse/run_demo.py
"""

    for fname, content in prompts.items():
        (PROMPTS_DIR/fname).write_text(content.strip(), encoding="utf-8")
        print(f"  Written: {fname}")
    print(f"\nAll {len(prompts)} prompts in: {PROMPTS_DIR}\nSTEP 4 DONE.")

def print_next_steps():
    r = len(list(REAL_DIR.glob("*.mp4"))) if REAL_DIR.exists() else 0
    f = len(list(FAKE_DIR.glob("*.mp4"))) if FAKE_DIR.exists() else 0
    print(f"""
{"="*70}
WHAT TO DO NEXT
{"="*70}

DATASET: {r} real | {f} fake  (target: 100 real, 50 fake)

AI PROMPTS: {PROMPTS_DIR}

BUILD ORDER (one prompt at a time):
  1. 01_phase1_setup.txt         -> bash setup.sh -> python verify_env.py (must PASS)
  2. 02_phase2_roi_extractor.txt -> python src/roi_extractor.py (see face polygons)
  3. 03_phase3_signal_processor.txt -> python src/signal_processor.py (must PASS)
  4. 04a_phase4a_realtime_pipeline.txt -> python src/realtime_pipeline.py (see LIVE HUMAN)
  5. 04b_phase4b_batch_analyzer.txt -> python src/batch_analyzer.py --real_dir ... --fake_dir ...
  6. 05_phase5_dashboard.txt     -> streamlit run dashboard/app.py
  7. 06_evaluator.txt            -> python src/evaluator.py ... (generates ROC + LaTeX table)
  8. 07_run_demo.txt             -> python run_demo.py (final proof for supervisor)

RECORD YOUR OWN REAL VIDEOS:
  {r} real videos found. Need 100+. Each team member: 10 clips x 30 seconds.
  Save as .mp4, copy to: {REAL_DIR}

IF A PROMPT ERRORS:
  Paste error back to AI: "Fix this error only: [paste error]"

{"="*70}
""")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--prompts-only", action="store_true")
    args = parser.parse_args()
    print("\n" + "="*70 + "\nNEURO-PULSE SETUP — SRM AP University\n" + "="*70)
    if args.prompts_only:
        write_prompts(); print_next_steps(); return
    if not args.skip_download:
        ans = input("\nType YES to download dataset (3-5 GB, 20-60 min) or NO to skip: ").strip().upper()
        if ans == "YES":
            download_dataset()
    organise_files()
    write_prompts()
    print_next_steps()

if __name__ == "__main__":
    main()