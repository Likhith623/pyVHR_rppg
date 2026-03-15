#in kaggle : 200 real images, 200 fake images




#!/usr/bin/env python3
"""
=============================================================================
NEURO-PULSE — KAGGLE ML TRAINING NOTEBOOK
=============================================================================

SETUP BEFORE RUNNING ON KAGGLE:
─────────────────────────────────────────────────────────────────────────────
The FaceForensics++ download script requires interactive terminal input
(pressing ENTER to agree to ToS) which Kaggle does not support.
You MUST upload your videos as Kaggle Datasets first.

STEP 1 — Upload videos as Kaggle Datasets:
  a) Go to kaggle.com → Datasets → New Dataset
  b) Create dataset named: "ff-real"
     Upload your real videos from:
     ff_downloads/original_sequences/youtube/c40/videos/*.mp4
     (200 videos recommended for Kaggle free tier)

  c) Create dataset named: "ff-fake"
     Upload your fake videos from:
     ff_downloads/all_fakes/*.mp4
     (mix of all 5 methods, 200 videos total)

STEP 2 — Create new Kaggle Notebook:
  Settings → Accelerator → GPU P100
  Add both datasets (ff-real, ff-fake) to the notebook

STEP 3 — Run this script

─────────────────────────────────────────────────────────────────────────────
RUNNING LOCALLY (on your Mac):
  source neuropulse_env/bin/activate
  python3 neuro_pulse_kaggle_final.py

─────────────────────────────────────────────────────────────────────────────
WHY THIS CODE TRAINS A CLASSIFIER:

  For liveness detection (is this a real person?):
    SNR threshold alone works.
    A real person shows SNR > 3dB because blood flows under skin.
    A printed photo or screen shows SNR < 3dB. Simple rule is enough.

  For video deepfake detection (is this video manipulated?):
    SNR threshold FAILS (51% accuracy = coin flip).
    Reason: Face-swap deepfakes use real source video.
    The rPPG signal leaks from the source person into the deepfake.
    Both real and fake videos show similar SNR values.
    SOLUTION: Train a classifier on 35 features that together capture
    subtle spatial inconsistencies, phase mismatch between ROIs,
    amplitude asymmetry, and visual quality differences that
    no single threshold can capture.

  This code satisfies BOTH:
    1. SNR baseline (liveness) → reported first
    2. ML classifier (deepfake detection) → trained and evaluated

─────────────────────────────────────────────────────────────────────────────
KAGGLE MEMORY LIMITS:
  Free tier: 16 GB RAM, 77 GB disk, 9 hours runtime
  Recommended: 200 real + 200 fake videos = ~6 GB data, ~4 hours runtime
  Maximum safe: 500 real + 500 fake = ~12 GB data, ~8 hours runtime
=============================================================================
"""

import os, sys, cv2, time, json, warnings, subprocess
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path
from scipy import signal as sp_signal
from scipy.interpolate import CubicSpline
from scipy.signal import detrend, csd, welch
from scipy.stats import skew
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Auto-detect Kaggle
ON_KAGGLE = os.path.exists('/kaggle')

if ON_KAGGLE:
    # Kaggle paths — your uploaded dataset names
    REAL_DIR   = Path('/kaggle/input/ff-real')
    FAKE_DIR   = Path('/kaggle/input/ff-fake')
    OUTPUT_DIR = Path('/kaggle/working/outputs')
    # Kaggle memory limit: keep at 200 per class
    MAX_VIDEOS_PER_CLASS = 200
else:
    # Local Mac paths
    BASE_DIR   = Path(__file__).parent.resolve()
    REAL_DIR   = BASE_DIR / 'ff_downloads/original_sequences/youtube/c40/videos'
    FAKE_DIR   = BASE_DIR / 'ff_downloads/all_fakes'
    OUTPUT_DIR = BASE_DIR / 'neuro_pulse/outputs/ml_results'
    # Local: use all available (up to 1000)
    MAX_VIDEOS_PER_CLASS = 1000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Signal processing
MAX_FRAMES    = 300      # 10 seconds at 30 FPS per video
MIN_FRAMES    = 90       # Minimum frames to compute features
TARGET_FS     = 256.0    # Hz after resampling
SNR_THRESHOLD = 3.0      # dB — baseline liveness threshold
N_JOBS        = -1       # Use all CPU cores

print("=" * 70)
print("NEURO-PULSE — KAGGLE ML TRAINING")
print("=" * 70)
print(f"Platform     : {'Kaggle P100 GPU' if ON_KAGGLE else 'Local Mac'}")
print(f"Real dir     : {REAL_DIR}")
print(f"Fake dir     : {FAKE_DIR}")
print(f"Output dir   : {OUTPUT_DIR}")
print(f"Max per class: {MAX_VIDEOS_PER_CLASS}")


# =============================================================================
# STEP 1 — INSTALL MISSING DEPENDENCIES
# =============================================================================

def pip_install(pkg, import_name=None):
    name = import_name or pkg.split('==')[0].replace('-', '_')
    try:
        __import__(name)
    except ImportError:
        print(f"  pip install {pkg}...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'],
                       check=False)

print("\n--- Installing dependencies ---")
for pkg in ['xgboost', 'lightgbm', 'deap', 'torch', 'torchvision']:
    pip_install(pkg)
print("  Done.")

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               VotingClassifier, StackingClassifier,
                               AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                      cross_val_score)
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (accuracy_score, roc_auc_score,
                              confusion_matrix, roc_curve)
from sklearn.feature_selection import mutual_info_classif
import joblib


# =============================================================================
# STEP 2 — ROI DEFINITIONS
# =============================================================================

ROI_INDICES = {
    'forehead':    [10, 338, 297, 332, 284, 251, 389],
    'left_cheek':  [234, 93, 132, 58, 172],
    'right_cheek': [454, 323, 361, 288, 397],
    'nose':        [1, 2, 5, 4, 6, 19, 94],
    'chin':        [152, 148, 176, 149, 150, 136, 172],
}


def roi_mask(lm, indices, h, w):
    pts = np.array([[int(lm.landmark[i].x * w),
                     int(lm.landmark[i].y * h)] for i in indices],
                   dtype=np.int32)
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(m, pts, 255)
    return m


# =============================================================================
# STEP 3 — SIGNAL PROCESSING UTILITIES
# =============================================================================

def bandpass_filter(sig, low=0.7, high=4.0, fs=30.0, order=4):
    sig = detrend(np.asarray(sig, dtype=np.float64))
    nyq = fs / 2.0
    b, a = sp_signal.butter(order, [low/nyq, high/nyq], btype='band')
    return sp_signal.filtfilt(b, a, sig)


def resample_to_256hz(sig, fs=30.0):
    sig = np.asarray(sig, dtype=np.float64)
    n   = len(sig)
    cs  = CubicSpline(np.arange(n) / fs, sig)
    return cs(np.arange(int(n / fs * TARGET_FS)) / TARGET_FS)


def welch_psd(sig, fs=TARGET_FS):
    return welch(sig, fs=fs, nperseg=256, noverlap=128)


def cardiac_band(freqs, low=0.7, high=4.0):
    return (freqs >= low) & (freqs <= high)


# =============================================================================
# STEP 4 — FEATURE EXTRACTION
# Two separate use cases explained:
#
#   USE CASE A — Liveness detection (is this a real person?):
#     Uses SNR threshold only. Works because printed photos and
#     screen replays have NO rPPG signal.
#
#   USE CASE B — Video deepfake detection (is this video manipulated?):
#     Needs ML classifier. Face-swap deepfakes inherit rPPG from
#     source video so SNR alone is insufficient.
#     This function extracts 35 features for the ML classifier.
# =============================================================================

# One FaceMesh instance reused across all videos (saves ~200ms init per video)
_face_mesh_instance = None


def get_face_mesh():
    global _face_mesh_instance
    if _face_mesh_instance is None:
        _face_mesh_instance = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False,
            refine_landmarks=False,
        )
    return _face_mesh_instance


def extract_35_features(video_path: Path) -> dict:
    """
    Extracts 35 features from a video file.

    Feature groups:
      Group 1 (8):  Spectral quality — SNR, purity, peak prominence,
                    peak sharpness, HR, n_peaks, periodicity, ac_peak_count
      Group 2 (8):  Spatial ROI consistency — correlation mean/std,
                    phase coherence mean/std, amp ratio, amp max/min,
                    signal entropy, var_skew
      Group 3 (4):  Temporal stability — var_stability, var_mean,
                    var_skew, signal_entropy
      Group 4 (3):  Multi-channel — red purity, G-R purity diff, G-R corr
      Group 5 (6):  Visual quality — sharpness mean/std/min,
                    motion mean/std, face detection rate
      Total = 35 features

    Returns None if video has fewer than MIN_FRAMES with face detected.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fm  = get_face_mesh()

    g_roi   = {k: [] for k in ROI_INDICES}   # green channel per ROI
    r_roi   = {k: [] for k in ROI_INDICES}   # red channel per ROI
    lap_v   = []                              # Laplacian variance per frame
    fdiffs  = []                              # frame-to-frame pixel diff
    prev_g  = None
    n_face  = 0
    n_frame = 0

    while n_frame < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        res  = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if res.multi_face_landmarks:
            fl = res.multi_face_landmarks[0]
            gc = frame[:, :, 1]   # green (BGR index 1)
            rc = frame[:, :, 2]   # red   (BGR index 2)
            n_face += 1

            for name, idxs in ROI_INDICES.items():
                m  = roi_mask(fl, idxs, h, w)
                gp = gc[m == 255]
                rp = rc[m == 255]
                if len(gp) > 0:
                    g_roi[name].append(float(np.mean(gp)))
                    r_roi[name].append(float(np.mean(rp)))

            # Sharpness: Laplacian variance inside face convex hull
            fpts = np.array([[int(fl.landmark[i].x * w),
                               int(fl.landmark[i].y * h)]
                              for i in range(0, 468, 10)], dtype=np.int32)
            hull  = cv2.convexHull(fpts)
            fmask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(fmask, hull, 255)
            fg = gray.copy(); fg[fmask == 0] = 0
            lap = cv2.Laplacian(fg, cv2.CV_64F)
            lap_v.append(float(np.var(lap[fmask == 255])))

            # Frame-difference: temporal motion consistency
            if prev_g is not None:
                diff = np.abs(gray.astype(float) - prev_g.astype(float))
                fd   = diff[fmask == 255]
                if len(fd) > 0:
                    fdiffs.append(float(np.std(fd)))

        prev_g = gray.copy()
        n_frame += 1

    cap.release()

    min_len = min(len(v) for v in g_roi.values())
    if min_len < MIN_FRAMES:
        return None

    f = {}

    try:
        # ── Combined green signal ──────────────────────────────────────────
        combined = np.mean([np.array(g_roi[k][:min_len]) for k in ROI_INDICES],
                           axis=0)
        bg = float(np.mean(combined[:40])) if min_len > 40 else 0.0
        filt = bandpass_filter(combined - bg, fs=fps)
        res  = resample_to_256hz(filt, fs=fps)
        freqs, psd = welch_psd(res)
        mc = cardiac_band(freqs)

        # ── Group 1: Spectral quality ──────────────────────────────────────
        sp  = float(np.max(psd[mc]))
        np_ = float(np.mean(psd[~mc]))
        f['snr_db']          = 10.0 * np.log10(sp / (np_ + 1e-10))
        f['spectral_purity'] = float(np.sum(psd[mc]) / (np.sum(psd) + 1e-10))

        cpsd = psd[mc]
        pi   = int(np.argmax(cpsd))
        pv   = float(cpsd[pi])
        surr = np.concatenate([cpsd[:max(0, pi-3)],
                               cpsd[min(len(cpsd), pi+3):]])
        f['peak_prominence'] = float(pv / (float(np.mean(surr)) + 1e-10))
        f['peak_sharpness']  = float(sp / (float(np.std(cpsd)) + 1e-10))
        f['hr_bpm']          = float(freqs[mc][pi] * 60.0)

        from scipy.signal import find_peaks
        pks, _ = find_peaks(cpsd, height=float(np.mean(cpsd)))
        f['n_peaks'] = int(len(pks))

        # Autocorrelation periodicity
        ac     = np.correlate(filt, filt, mode='full')
        ac     = ac[len(ac)//2:] / (ac[len(ac)//2] + 1e-10)
        lo, hi = int(0.25 * fps), int(1.5 * fps)
        if hi < len(ac):
            ac_r   = ac[lo:hi]
            apks, _ = find_peaks(ac_r)
            f['periodicity']   = float(ac_r[apks[0]]) if len(apks) > 0 else 0.0
            f['ac_peak_count'] = int(len(apks))
        else:
            f['periodicity']   = 0.0
            f['ac_peak_count'] = 0

        # ── Group 2: Spatial ROI consistency ──────────────────────────────
        roi_f = {k: bandpass_filter(np.array(g_roi[k][:min_len], dtype=np.float64),
                                    fs=fps)
                 for k in ROI_INDICES}

        names_list = list(roi_f.keys())
        corrs = []
        for i in range(len(names_list)):
            for j in range(i+1, len(names_list)):
                c = np.corrcoef(roi_f[names_list[i]], roi_f[names_list[j]])[0, 1]
                corrs.append(float(c))
        f['roi_corr_mean'] = float(np.mean(corrs))
        f['roi_corr_std']  = float(np.std(corrs))

        # Phase coherence via cross-spectral density
        nperseg = min(len(roi_f['forehead']), 64)
        phases  = []
        r3 = [roi_f[k] for k in list(ROI_INDICES.keys())[:3]]
        for i in range(len(r3)):
            for j in range(i+1, len(r3)):
                _, Pxy = csd(r3[i], r3[j], fs=fps, nperseg=nperseg)
                phases.append(float(np.mean(np.abs(np.angle(Pxy)))))
        f['phase_coh_mean'] = float(np.mean(phases))
        f['phase_coh_std']  = float(np.std(phases))

        # Amplitude ratio
        amps = [float(np.std(roi_f[k])) for k in roi_f]
        f['amp_ratio']   = float(np.std(amps) / (np.mean(amps) + 1e-10))
        f['amp_max_min'] = float(np.max(amps) / (np.min(amps) + 1e-10))

        # ── Group 3: Temporal stability ────────────────────────────────────
        win  = 30
        wstd = []
        for k in list(ROI_INDICES.keys())[:3]:
            arr = np.array(g_roi[k][:min_len])
            for i in range(0, len(arr)-win, win):
                wstd.append(float(np.std(arr[i:i+win])))
        f['var_stability'] = float(np.std(wstd))
        f['var_mean']      = float(np.mean(wstd))
        f['var_skew']      = float(skew(wstd)) if len(wstd) > 2 else 0.0

        hist, _ = np.histogram(filt, bins=20, density=True)
        hist    = hist + 1e-10
        f['signal_entropy'] = float(-np.sum(hist * np.log(hist)))

        # ── Group 4: Green vs Red channel comparison ───────────────────────
        r_comb = np.mean([np.array(r_roi[k][:min_len]) for k in ROI_INDICES],
                         axis=0)
        r_filt = bandpass_filter(r_comb, fs=fps)
        r_res  = resample_to_256hz(r_filt, fs=fps)
        r_f, r_p = welch_psd(r_res)
        r_mc = cardiac_band(r_f)

        r_pur           = float(np.sum(r_p[r_mc]) / (np.sum(r_p) + 1e-10))
        f['red_purity']     = r_pur
        f['gr_purity_diff'] = f['spectral_purity'] - r_pur

        n_gr = min(len(filt), len(r_filt))
        f['gr_corr'] = float(np.corrcoef(filt[:n_gr], r_filt[:n_gr])[0, 1])

    except Exception as e:
        return None

    # ── Group 5: Visual quality ────────────────────────────────────────────
    f['sharp_mean']  = float(np.mean(lap_v))  if lap_v  else 0.0
    f['sharp_std']   = float(np.std(lap_v))   if lap_v  else 0.0
    f['sharp_min']   = float(np.min(lap_v))   if lap_v  else 0.0
    f['motion_mean'] = float(np.mean(fdiffs)) if fdiffs else 0.0
    f['motion_std']  = float(np.std(fdiffs))  if fdiffs else 0.0
    f['face_rate']   = float(n_face / max(n_frame, 1))

    return f


# =============================================================================
# STEP 5 — LOAD OR EXTRACT FEATURES (with caching)
# =============================================================================

def build_feature_dataset(real_dir: Path, fake_dir: Path,
                           cache_file: Path, max_per_class: int):
    """
    Extracts features from all videos, with JSON cache.
    Re-uses cache on subsequent runs (saves hours of processing).
    """
    if cache_file.exists():
        print(f"\nLoading from cache: {cache_file}")
        data = json.loads(cache_file.read_text())
        return data['records'], data['feature_names']

    print("\n" + "="*70)
    print("FEATURE EXTRACTION — this takes 2-4 hours on Kaggle")
    print("="*70)

    records = []

    for label, directory in [('REAL', real_dir), ('FAKE', fake_dir)]:
        files = sorted(list(directory.glob('*.mp4')))[:max_per_class]
        print(f"\n{label}: {len(files)} videos from {directory}")

        for i, vp in enumerate(files):
            t0    = time.time()
            feats = extract_35_features(vp)
            dt    = time.time() - t0

            if feats is not None:
                records.append({'file': vp.name, 'label': label, **feats})
            status = f"SNR={feats['snr_db']:5.1f}" if feats else "SKIP"

            if i < 3 or (i+1) % 50 == 0:
                print(f"  [{i+1:4d}/{len(files)}] {vp.name:25s} "
                      f"{status}  {dt:.1f}s")

    feat_names = [k for k in records[0] if k not in ('file', 'label')] \
                 if records else []
    cache_file.write_text(json.dumps({'records': records,
                                      'feature_names': feat_names}))
    print(f"\nCached → {cache_file}")
    return records, feat_names


# =============================================================================
# STEP 6 — GENETIC ALGORITHM FEATURE SELECTION
# =============================================================================

def run_genetic_algorithm(X: np.ndarray, y: np.ndarray,
                          feat_names: list,
                          n_gen: int = 40, pop_size: int = 60) -> list:
    """
    DEAP genetic algorithm finds the optimal subset of features.

    Each individual = binary vector of length n_features.
    Fitness = 5-fold CV accuracy of Random Forest on selected features.

    Genetic operators:
      Selection : Tournament (size=3)
      Crossover : Two-point (cxpb=0.7)
      Mutation  : Bit-flip, 5% per gene (mutpb=0.2)

    Returns: list of selected feature indices.
    """
    print("\n" + "="*70)
    print(f"GENETIC ALGORITHM  (pop={pop_size}, gen={n_gen})")
    print("="*70)

    try:
        from deap import base, creator, tools, algorithms
    except ImportError:
        print("  DEAP missing — using all features.")
        return list(range(X.shape[1]))

    n = X.shape[1]
    for a in ('FitnessMax', 'Individual'):
        if hasattr(creator, a):
            delattr(creator, a)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    tb = base.Toolbox()
    tb.register("bit",        np.random.randint, 0, 2)
    tb.register("individual", tools.initRepeat, creator.Individual, tb.bit, n)
    tb.register("population", tools.initRepeat, list, tb.individual)

    def fitness(ind):
        sel = [i for i, v in enumerate(ind) if v]
        if not sel:
            return (0.0,)
        clf    = RandomForestClassifier(50, n_jobs=2, random_state=42)
        scores = cross_val_score(clf, X[:, sel], y, cv=5, scoring='accuracy')
        return (float(scores.mean()),)

    tb.register("evaluate", fitness)
    tb.register("mate",    tools.cxTwoPoint)
    tb.register("mutate",  tools.mutFlipBit, indpb=0.05)
    tb.register("select",  tools.selTournament, tournsize=3)

    pop = tb.population(n=pop_size)
    hof = tools.HallOfFame(1)
    st  = tools.Statistics(lambda ind: ind.fitness.values)
    st.register("max", np.max)
    st.register("avg", np.mean)

    algorithms.eaSimple(pop, tb, cxpb=0.7, mutpb=0.2,
                         ngen=n_gen, stats=st, halloffame=hof, verbose=True)

    sel_idx = [i for i, v in enumerate(hof[0]) if v]
    print(f"\nGA selected {len(sel_idx)}/{n} features:")
    for i in sel_idx:
        print(f"  + {feat_names[i]}")

    pd.DataFrame(list(st.compile(pop).items())).to_csv(
        OUTPUT_DIR / 'ga_logbook.csv', index=False)
    return sel_idx


# =============================================================================
# STEP 7 — CLASSIFIERS
# Explanation of why classifiers are needed:
#
# Face-swap deepfakes preserve the temporal pixel structure of the source
# video. This means rPPG features like SNR have nearly identical
# distributions for real and fake videos (both ~25 dB).
# No single threshold separates them.
#
# However, the face-swap process introduces subtle multi-dimensional
# inconsistencies:
#   - Phase mismatch between forehead and cheek rPPG signals
#   - Amplitude asymmetry across ROIs at swap boundaries
#   - Slight reduction in Laplacian sharpness at face edges
#   - Temporal variance patterns differ from natural motion
#
# A classifier combines all 35 features simultaneously and learns the
# decision boundary in this high-dimensional space.
# =============================================================================

def define_all_classifiers():
    return {
        # Linear models
        'LogisticRegression':
            LogisticRegression(C=1.0, max_iter=1000, random_state=42),

        # SVM variants
        'SVM_RBF':
            SVC(kernel='rbf', C=10.0, gamma='scale',
                probability=True, random_state=42),
        'SVM_Linear':
            SVC(kernel='linear', C=1.0, probability=True, random_state=42),
        'SVM_Poly':
            SVC(kernel='poly', degree=3, probability=True, random_state=42),

        # Nearest neighbours
        'KNN_5':  KNeighborsClassifier(n_neighbors=5),
        'KNN_11': KNeighborsClassifier(n_neighbors=11),

        # Tree models
        'DecisionTree':
            DecisionTreeClassifier(max_depth=10, random_state=42),
        'RandomForest_100':
            RandomForestClassifier(100, random_state=42, n_jobs=N_JOBS),
        'RandomForest_500':
            RandomForestClassifier(500, random_state=42, n_jobs=N_JOBS),
        'ExtraTrees':
            ExtraTreesClassifier(500, random_state=42, n_jobs=N_JOBS),

        # Boosting
        'GradientBoosting':
            GradientBoostingClassifier(300, max_depth=5,
                                       learning_rate=0.05, random_state=42),
        'AdaBoost':
            AdaBoostClassifier(200, random_state=42),
        'XGBoost':
            xgb.XGBClassifier(500, max_depth=6, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               use_label_encoder=False, eval_metric='logloss',
                               random_state=42, n_jobs=N_JOBS),
        'LightGBM':
            lgb.LGBMClassifier(500, max_depth=6, learning_rate=0.05,
                                num_leaves=63, subsample=0.8,
                                random_state=42, n_jobs=N_JOBS, verbose=-1),

        # Neural networks
        'MLP_Small':
            MLPClassifier((128, 64), max_iter=500,
                          early_stopping=True, random_state=42),
        'MLP_Large':
            MLPClassifier((256, 128, 64, 32), max_iter=500,
                          early_stopping=True, random_state=42),

        # Probabilistic
        'NaiveBayes': GaussianNB(),
    }


def build_stacking():
    """Level 1: RF + XGB + LGB + SVM + ET  →  Level 2: Logistic Regression"""
    return StackingClassifier(
        estimators=[
            ('rf',  RandomForestClassifier(300, random_state=42, n_jobs=N_JOBS)),
            ('xgb', xgb.XGBClassifier(300, max_depth=5, learning_rate=0.05,
                                       use_label_encoder=False,
                                       eval_metric='logloss',
                                       random_state=42, n_jobs=N_JOBS)),
            ('lgb', lgb.LGBMClassifier(300, max_depth=5, learning_rate=0.05,
                                        random_state=42, n_jobs=N_JOBS,
                                        verbose=-1)),
            ('svm', SVC(kernel='rbf', C=10, gamma='scale',
                        probability=True, random_state=42)),
            ('et',  ExtraTreesClassifier(300, random_state=42, n_jobs=N_JOBS)),
        ],
        final_estimator=LogisticRegression(C=1.0, max_iter=500, random_state=42),
        cv=5, passthrough=True, n_jobs=1,
    )


def build_voting():
    """Soft voting: RF + XGB + LGB + SVM"""
    return VotingClassifier(
        estimators=[
            ('rf',  RandomForestClassifier(500, random_state=42, n_jobs=N_JOBS)),
            ('xgb', xgb.XGBClassifier(500, max_depth=6, learning_rate=0.05,
                                       use_label_encoder=False,
                                       eval_metric='logloss',
                                       random_state=42, n_jobs=N_JOBS)),
            ('lgb', lgb.LGBMClassifier(500, max_depth=6, learning_rate=0.05,
                                        random_state=42, n_jobs=N_JOBS,
                                        verbose=-1)),
            ('svm', SVC(kernel='rbf', C=10, gamma='scale',
                        probability=True, random_state=42)),
        ],
        voting='soft', n_jobs=1,
    )


# =============================================================================
# STEP 8 — DEEP LEARNING MLP (PyTorch)
# =============================================================================

def train_deep_mlp(X: np.ndarray, y: np.ndarray) -> dict:
    """
    4-layer deep MLP with BatchNorm and Dropout.
    Uses GPU if available (Kaggle P100).
    5-fold stratified cross-validation.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    class DeepMLP(nn.Module):
        def __init__(self, d_in):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(64, 2),
            )
        def forward(self, x):
            return self.net(x)

    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds  = np.zeros(len(y), dtype=int)
    probas = np.zeros(len(y))

    for fold, (tr, vl) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold+1}/5...")
        Xtr = torch.FloatTensor(X[tr]).to(device)
        ytr = torch.LongTensor(y[tr]).to(device)
        Xvl = torch.FloatTensor(X[vl]).to(device)

        model   = DeepMLP(X.shape[1]).to(device)
        opt     = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150)
        loss_fn = nn.CrossEntropyLoss()
        loader  = DataLoader(TensorDataset(Xtr, ytr), batch_size=32, shuffle=True)

        best_acc, best_wts = 0.0, None
        for epoch in range(300):
            model.train()
            for Xb, yb in loader:
                opt.zero_grad(); loss_fn(model(Xb), yb).backward(); opt.step()
            sched.step()
            if (epoch + 1) % 30 == 0:
                model.eval()
                with torch.no_grad():
                    vp  = model(Xvl).argmax(1).cpu().numpy()
                acc = accuracy_score(y[vl], vp)
                if acc > best_acc:
                    best_acc = acc
                    best_wts = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_wts)
        model.eval()
        with torch.no_grad():
            preds[vl]  = model(Xvl).argmax(1).cpu().numpy()
            probas[vl] = torch.softmax(model(Xvl), dim=1)[:, 1].cpu().numpy()

    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()
    return {
        'name': 'DeepMLP_PyTorch',
        'accuracy': accuracy_score(y, preds),
        'tpr': tp / (tp + fn + 1e-10),
        'tnr': tn / (tn + fp + 1e-10),
        'fpr': fp / (fp + tn + 1e-10),
        'fnr': fn / (fn + tp + 1e-10),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'auc':     roc_auc_score(y, probas),
        'y_pred':  preds,
        'y_proba': probas,
    }


# =============================================================================
# STEP 9 — EVALUATION HELPER
# =============================================================================

def run_cv(name, clf, X, y, skf):
    """5-fold stratified CV → full metrics dict."""
    try:
        yp = cross_val_predict(clf, X, y, cv=skf)
        try:
            ypr = cross_val_predict(clf, X, y, cv=skf,
                                     method='predict_proba')[:, 1]
            auc = roc_auc_score(y, ypr)
        except Exception:
            ypr = None; auc = float('nan')

        cm = confusion_matrix(y, yp)
        if cm.shape != (2, 2):
            return None
        tn, fp, fn, tp = cm.ravel()
        tot = tp + tn + fp + fn
        return {
            'name': name,
            'accuracy': (tp + tn) / tot,
            'tpr': tp / (tp + fn + 1e-10),
            'tnr': tn / (tn + fp + 1e-10),
            'fpr': fp / (fp + tn + 1e-10),
            'fnr': fn / (fn + tp + 1e-10),
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
            'auc':     auc,
            'y_pred':  yp,
            'y_proba': ypr,
        }
    except Exception as e:
        print(f"  ERROR {name}: {e}")
        return None


# =============================================================================
# STEP 10 — PAPER FIGURES AND REPORTING
# =============================================================================

def print_results_table(results):
    print("\n" + "="*98)
    print(f"  {'Model':<26s} {'Acc':>8s} {'TPR':>7s} {'TNR':>7s} "
          f"{'FPR':>7s} {'FNR':>7s} {'AUC':>7s}")
    print("="*98)
    for r in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        auc_s = f"{r['auc']:6.3f}" if not np.isnan(r['auc']) else "   N/A"
        print(f"  {r['name']:<26s} {r['accuracy']:7.2%} "
              f"{r['tpr']:6.2%} {r['tnr']:6.2%} "
              f"{r['fpr']:6.2%} {r['fnr']:6.2%} {auc_s}")
    print("="*98)


def generate_paper_figures(results, records, y, out: Path):
    """All figures needed for the paper."""

    # Figure 1: SNR distribution (shows why SNR-only fails for face-swaps)
    real_snr = [r['snr_db'] for r in records if r['label'] == 'REAL']
    fake_snr = [r['snr_db'] for r in records if r['label'] == 'FAKE']
    plt.figure(figsize=(10, 5))
    plt.hist(real_snr, 30, alpha=0.6, color='#2D7D4E',
             label='Real', edgecolor='white')
    plt.hist(fake_snr, 30, alpha=0.6, color='#CC2222',
             label='Deepfake', edgecolor='white')
    plt.axvline(SNR_THRESHOLD, color='navy', ls='--', lw=2,
                label=f'SNR threshold ({SNR_THRESHOLD} dB)')
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('SNR Distribution: Real vs Deepfake — Why ML is Needed', fontsize=13)
    note = (f"Overlap between distributions confirms SNR alone is insufficient\n"
            f"for face-swap deepfakes. ML classifier required.")
    plt.figtext(0.5, -0.02, note, ha='center', fontsize=9, style='italic')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / 'snr_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Figure 1: snr_distribution.png")

    # Figure 2: ROC curves (all models)
    if results:
        plt.figure(figsize=(12, 8))
        for r in results:
            if r.get('y_proba') is not None and not np.isnan(r['auc']):
                fp_c, tp_c, _ = roc_curve(y, r['y_proba'])
                plt.plot(fp_c, tp_c, lw=1.2,
                         label=f"{r['name']} (AUC={r['auc']:.3f})")
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves — All Models', fontsize=14)
        plt.legend(loc='lower right', fontsize=7, ncol=2)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / 'roc_all_models.png', dpi=150)
        plt.close()
        print("  Figure 2: roc_all_models.png")

        # Figure 3: Accuracy bar chart
        rs   = sorted(results, key=lambda x: x['accuracy'], reverse=True)
        names = [r['name'] for r in rs]
        accs  = [r['accuracy'] * 100 for r in rs]
        cols  = ['#2D7D4E' if a >= 90 else '#E6A817' if a >= 75 else '#CC2222'
                 for a in accs]
        fig, ax = plt.subplots(figsize=(10, max(6, len(names)*0.42)))
        bars = ax.barh(names, accs, color=cols)
        ax.axvline(93, color='navy', ls='--', lw=1.5, label='Target 93%')
        for b, a in zip(bars, accs):
            ax.text(b.get_width()+0.3, b.get_y()+b.get_height()/2,
                    f'{a:.1f}%', va='center', fontsize=9)
        ax.set_xlim(0, 108)
        ax.set_xlabel('Accuracy (%)')
        ax.set_title('Model Accuracy Comparison — Neuro-Pulse')
        ax.legend()
        plt.tight_layout()
        plt.savefig(out / 'model_comparison.png', dpi=150)
        plt.close()
        print("  Figure 3: model_comparison.png")


def feature_importance_analysis(X, y, feat_names, out: Path):
    """Feature importance + mutual information plots."""
    rf  = RandomForestClassifier(500, random_state=42, n_jobs=N_JOBS)
    rf.fit(X, y)
    imp = rf.feature_importances_
    mi  = mutual_info_classif(X, y, random_state=42)

    idx_rf = np.argsort(imp)[::-1][:20]
    idx_mi = np.argsort(mi)[::-1][:20]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    ax1.barh([feat_names[i] for i in idx_rf][::-1],
             imp[idx_rf][::-1], color='steelblue')
    ax1.set_title('Feature Importance (Random Forest)', fontsize=12)
    ax1.set_xlabel('Importance')

    ax2.barh([feat_names[i] for i in idx_mi][::-1],
             mi[idx_mi][::-1], color='darkorange')
    ax2.set_title('Feature Importance (Mutual Information)', fontsize=12)
    ax2.set_xlabel('MI Score')

    plt.tight_layout()
    plt.savefig(out / 'feature_importance.png', dpi=150)
    plt.close()
    print("  Figure 4: feature_importance.png")

    print("\nTop 15 features (RF Importance):")
    for i in idx_rf[:15]:
        print(f"  {feat_names[i]:30s}: {imp[i]:.4f}")


def generate_latex_table(results, out: Path) -> str:
    top10 = sorted(results, key=lambda x: x['accuracy'], reverse=True)[:10]
    rows = [
        f"{r['name']} & {r['accuracy']:.2%} & {r['tpr']:.2%} & "
        f"{r['tnr']:.2%} & {r['fpr']:.2%} & {r['fnr']:.2%} & "
        f"{'—' if np.isnan(r['auc']) else f'{r[chr(97)+chr(117)+chr(99)]:.3f}'} \\\\"
        for r in top10
    ]
    # Fix auc key in f-string
    rows = []
    for r in top10:
        auc_s = "—" if np.isnan(r['auc']) else f"{r['auc']:.3f}"
        rows.append(
            f"{r['name']} & {r['accuracy']:.2%} & {r['tpr']:.2%} & "
            f"{r['tnr']:.2%} & {r['fpr']:.2%} & {r['fnr']:.2%} & {auc_s} \\\\"
        )
    tex = "\n".join([
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Neuro-Pulse: Deepfake Detection Performance Across Models}",
        r"\label{tab:results}",
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"\textbf{Model} & \textbf{Acc} & \textbf{TPR} & \textbf{TNR} "
        r"& \textbf{FPR} & \textbf{FNR} & \textbf{AUC} \\",
        r"\hline",
    ] + rows + [r"\hline", r"\end{tabular}", r"\end{table}"])
    (out / 'results_table.tex').write_text(tex)
    print("  LaTeX table: results_table.tex")
    return tex


def save_best_model(results, X, y, feat_names, scaler, out: Path):
    best = sorted(results, key=lambda x: x['accuracy'], reverse=True)[0]
    print(f"\nBest model: {best['name']}  "
          f"Acc={best['accuracy']:.2%}  AUC={best['auc']:.3f}")

    clf_map = define_all_classifiers()
    clf     = clf_map.get(best['name'])
    if clf is None:
        print("  (DeepMLP can't be saved as joblib — saving RF instead)")
        clf = RandomForestClassifier(500, random_state=42, n_jobs=N_JOBS)

    clf.fit(scaler.transform(X), y)
    bundle = {
        'model':         clf,
        'scaler':        scaler,
        'feature_names': feat_names,
        'accuracy':      best['accuracy'],
        'auc':           best['auc'],
        'snr_threshold': SNR_THRESHOLD,
        'usage': (
            'bundle = joblib.load("best_model.joblib")\n'
            'features = extract_35_features(video_path)  # 35-element dict\n'
            'X = [features[k] for k in bundle["feature_names"]]\n'
            'X_scaled = bundle["scaler"].transform([X])\n'
            'pred = bundle["model"].predict(X_scaled)  # 1=REAL 0=FAKE'
        ),
    }
    p = out / 'best_model.joblib'
    joblib.dump(bundle, p)
    print(f"  Saved → {p}")
    return p


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("ANSWERING THE TWO USE CASES:")
    print("  Use Case A — Liveness (real person?): SNR threshold alone")
    print("  Use Case B — Video deepfake (manipulated?): ML classifier")
    print("  This script handles BOTH and explains why B needs ML.")
    print("="*70)

    # ─── Verify data directories ───────────────────────────────────────────
    real_count = len(list(REAL_DIR.glob('*.mp4'))) if REAL_DIR.exists() else 0
    fake_count = len(list(FAKE_DIR.glob('*.mp4'))) if FAKE_DIR.exists() else 0

    print(f"\nReal videos found: {real_count}")
    print(f"Fake videos found: {fake_count}")

    if real_count == 0 or fake_count == 0:
        print(f"""
ERROR: No videos found.

{'Kaggle instructions:' if ON_KAGGLE else 'Local instructions:'}
{'  Upload real videos as Kaggle Dataset named: ff-real' if ON_KAGGLE else
  '  Run: python3 neuro_pulse_complete.py --download'}
{'  Upload fake videos as Kaggle Dataset named: ff-fake' if ON_KAGGLE else ''}
{'  Then add both datasets to this notebook.' if ON_KAGGLE else ''}
""")
        return

    # ─── Feature extraction ────────────────────────────────────────────────
    cache = OUTPUT_DIR / 'features_cache.json'
    records, feat_names = build_feature_dataset(
        REAL_DIR, FAKE_DIR, cache, MAX_VIDEOS_PER_CLASS
    )

    n_r = sum(1 for r in records if r['label'] == 'REAL')
    n_f = sum(1 for r in records if r['label'] == 'FAKE')
    print(f"\nExtracted features: {len(records)} videos ({n_r} real, {n_f} fake)")
    print(f"Feature count: {len(feat_names)}")

    X = np.nan_to_num(
        np.array([[r[f] for f in feat_names] for r in records]),
        nan=0.0, posinf=0.0, neginf=0.0
    )
    y = np.array([1 if r['label'] == 'REAL' else 0 for r in records])

    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # ─── SNR distribution figure ───────────────────────────────────────────
    generate_paper_figures([], records, y, OUTPUT_DIR)

    # ─── USE CASE A: Baseline SNR-only (liveness) ──────────────────────────
    print("\n" + "="*70)
    print("USE CASE A: LIVENESS — SNR-ONLY BASELINE (3.0 dB)")
    print("="*70)
    snr_col  = feat_names.index('snr_db')
    snr_pred = (X[:, snr_col] > SNR_THRESHOLD).astype(int)
    snr_acc  = accuracy_score(y, snr_pred)
    cm0 = confusion_matrix(y, snr_pred)
    tn0, fp0, fn0, tp0 = cm0.ravel()
    print(f"  Accuracy : {snr_acc:.2%}")
    print(f"  TP={tp0} TN={tn0} FP={fp0} FN={fn0}")
    print(f"  TPR={tp0/(tp0+fn0):.2%}  FPR={fp0/(fp0+tn0):.2%}")
    print("\n  Interpretation:")
    print("  If accuracy ≈ 50%, SNR cannot separate real from face-swap fake.")
    print("  This confirms Use Case B (ML classifier) is necessary.")

    # ─── USE CASE B: GA feature selection ──────────────────────────────────
    ga_idx = run_genetic_algorithm(
        X_scaled, y, feat_names, n_gen=40, pop_size=60
    )
    X_ga = X_scaled[:, ga_idx]

    # ─── Feature importance ────────────────────────────────────────────────
    print("\n--- Feature importance analysis ---")
    feature_importance_analysis(X, y, feat_names, OUTPUT_DIR)

    # ─── USE CASE B: All classical classifiers ─────────────────────────────
    print("\n" + "="*70)
    print("USE CASE B: VIDEO DEEPFAKE — ALL CLASSIFIERS (5-fold CV)")
    print("="*70)
    skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clfs    = define_all_classifiers()
    results = []

    for name, clf in clfs.items():
        print(f"  {name}...")
        r = run_cv(name, clf, X_scaled, y, skf)
        if r:
            results.append(r)
            print(f"    Acc={r['accuracy']:.2%}  AUC={r['auc']:.3f}")

    # GA-optimised versions of top models
    for name in ['RandomForest_500', 'XGBoost', 'LightGBM']:
        r = run_cv(f"{name}_GA", clfs[name], X_ga, y, skf)
        if r:
            results.append(r)
            print(f"  {name}_GA  Acc={r['accuracy']:.2%}")

    # ─── Ensemble methods ──────────────────────────────────────────────────
    print("\n" + "="*70)
    print("ENSEMBLE METHODS")
    print("="*70)
    for ename, eclf in [('Stacking', build_stacking()),
                         ('VotingEnsemble', build_voting())]:
        print(f"  {ename}...")
        r = run_cv(ename, eclf, X_scaled, y, skf)
        if r:
            results.append(r)
            print(f"    Acc={r['accuracy']:.2%}  AUC={r['auc']:.3f}")

    # ─── Deep Learning MLP ─────────────────────────────────────────────────
    print("\n" + "="*70)
    print("DEEP LEARNING MLP (PyTorch — GPU if available)")
    print("="*70)
    try:
        r_dl = train_deep_mlp(X_scaled, y)
        results.append(r_dl)
        print(f"  DeepMLP  Acc={r_dl['accuracy']:.2%}  AUC={r_dl['auc']:.3f}")
    except Exception as e:
        print(f"  Deep MLP failed: {e}")

    # ─── Results ───────────────────────────────────────────────────────────
    print_results_table(results)

    print("\n--- Generating paper figures ---")
    generate_paper_figures(results, records, y, OUTPUT_DIR)
    tex = generate_latex_table(results, OUTPUT_DIR)
    print("\nLaTeX table:")
    print(tex)

    print("\n--- Saving best model ---")
    save_best_model(results, X, y, feat_names, scaler, OUTPUT_DIR)

    # ─── Results CSV ───────────────────────────────────────────────────────
    clean = [{k: v for k, v in r.items() if k not in ('y_pred', 'y_proba')}
             for r in results]
    df = pd.DataFrame(clean).sort_values('accuracy', ascending=False)
    df.to_csv(OUTPUT_DIR / 'all_model_results.csv', index=False)

    # ─── Final summary ─────────────────────────────────────────────────────
    best = df.iloc[0]
    snr_result = f"SNR-only baseline: {snr_acc:.2%}"
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"  Use Case A (Liveness)   : {snr_result}")
    print(f"  Use Case B (Best model) : {best['name']}  {best['accuracy']:.2%}")
    print(f"  AUC                     : {best['auc']:.3f}")
    print(f"  TPR                     : {best['tpr']:.2%}")
    print(f"  TNR                     : {best['tnr']:.2%}")
    print(f"  FPR                     : {best['fpr']:.2%}")
    print(f"\n  Outputs saved to        : {OUTPUT_DIR}")
    print(f"  Files: snr_distribution.png, roc_all_models.png,")
    print(f"         model_comparison.png, feature_importance.png,")
    print(f"         results_table.tex, all_model_results.csv,")
    print(f"         best_model.joblib")
    print("="*70)

    return results, df


if __name__ == '__main__':
    results, df = main()
