# Neuro-Pulse: rPPG-Based Deepfake Detection System

## Abstract

Neuro-Pulse is a deepfake detection system that leverages **remote photoplethysmography (rPPG)** to distinguish real human faces from synthetic ones. Real faces exhibit subtle, periodic skin-colour variations caused by blood flow — a physiological signal that deepfake generators fail to reproduce accurately. The system extracts a Blood Volume Pulse (BVP) signal from facial regions of interest, applies spectral analysis, and classifies liveness based on signal-to-noise ratio (SNR) in the cardiac frequency band (0.7–4.0 Hz / 42–240 BPM).

## Installation

### Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Webcam (for live mode)

### Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd pyVHR_rppg/neuro_pulse

# 2. Run the setup script
chmod +x setup.sh
bash setup.sh

# 3. Activate the environment
conda activate neuropulse

# 4. Verify the installation
python verify_env.py

# 5. Create project structure (if not already present)
python create_structure.py
```

### Manual Installation

```bash
conda create -n neuropulse python=3.10 -y
conda activate neuropulse
pip install -r requirements.txt
```

## Project Structure

```
neuro_pulse/
├── src/
│   ├── roi_extractor.py       # Face ROI extraction via MediaPipe
│   ├── signal_processor.py    # BVP filtering, FFT, SNR classification
│   ├── realtime_pipeline.py   # Live webcam detection pipeline
│   └── batch_analyzer.py      # Batch video analysis with metrics
├── dashboard/
│   └── app.py                 # Streamlit web dashboard
├── models/                    # Saved models (future)
├── outputs/                   # CSV results, error logs
├── data/
│   ├── real/                  # Real face videos
│   └── fake/                  # Deepfake videos
├── tests/                     # Unit tests
├── requirements.txt
├── setup.sh
├── verify_env.py
├── create_structure.py
└── README.md
```

## Development Timeline

| Phase | Description                        | Key Deliverables                                |
|-------|------------------------------------|-------------------------------------------------|
| 1     | Environment Setup                  | `requirements.txt`, `setup.sh`, `verify_env.py` |
| 2     | ROI Extraction                     | `roi_extractor.py` — MediaPipe face mesh ROIs   |
| 3     | Signal Processing                  | `signal_processor.py` — BPF, FFT, SNR, classify |
| 4     | Pipeline Integration               | `realtime_pipeline.py`, `batch_analyzer.py`      |
| 5     | Dashboard & Visualization          | `app.py` — Streamlit interactive dashboard       |

## Run Commands

### Live Webcam Detection
```bash
cd neuro_pulse
python -m src.realtime_pipeline
```

### Live Detection with Video File
```bash
python -m src.realtime_pipeline --source path/to/video.mp4
```

### Batch Analysis
```bash
python -m src.batch_analyzer --real_dir data/real --fake_dir data/fake
```

### Signal Processor Self-Test
```bash
python -m src.signal_processor
```

### Streamlit Dashboard
```bash
cd neuro_pulse
streamlit run dashboard/app.py
```

---

*Neuro-Pulse | SRM AP University | CSE Section I*
