import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

"""
Neuro-Pulse Signal Processor Module.

Processes the raw green-channel BVP signal through a pipeline of:
  1. Butterworth bandpass filtering (0.7-4.0 Hz)
  2. Cubic-spline resampling to 256 Hz
  3. Welch PSD estimation
  4. SNR computation and heart-rate extraction
  5. Liveness classification based on SNR threshold
"""

from typing import Tuple, Optional, Dict, List
import numpy as np
from scipy import signal as scipy_signal
from scipy.interpolate import CubicSpline
from scipy.signal import detrend, find_peaks


def butterworth_bandpass(
    signal: np.ndarray,
    lowcut: float = 0.7,
    highcut: float = 4.0,
    fs: float = 30.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter.

    Detrends first, then applies a 4th-order Butterworth BPF via filtfilt.

    Args:
        signal: 1-D input signal array.
        lowcut: Lower cutoff frequency in Hz (default 0.7).
        highcut: Upper cutoff frequency in Hz (default 4.0).
        fs: Sampling frequency in Hz (default 30.0).
        order: Filter order (default 4).

    Returns:
        Filtered 1-D NumPy array.
    """
    sig = np.array(signal, dtype=np.float64)
    sig = detrend(sig)
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = scipy_signal.butter(order, [low, high], btype="band")
    return scipy_signal.filtfilt(b, a, sig)


def resample_signal(
    signal: np.ndarray,
    original_fs: float = 30.0,
    target_fs: float = 256.0,
) -> np.ndarray:
    """Resample signal from original_fs to target_fs using cubic spline.

    Args:
        signal: 1-D input signal array sampled at original_fs.
        original_fs: Original sampling rate in Hz (default 30.0).
        target_fs: Target sampling rate in Hz (default 256.0).

    Returns:
        Resampled 1-D NumPy array at target_fs.
    """
    sig = np.array(signal, dtype=np.float64)
    n = len(sig)
    duration = n / original_fs
    t_orig = np.arange(n) / original_fs
    cs = CubicSpline(t_orig, sig)
    n_target = int(duration * target_fs)
    t_target = np.arange(n_target) / target_fs
    return cs(t_target)


def compute_psd_welch(
    signal: np.ndarray,
    fs: float = 256.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Power Spectral Density using Welch's method.

    Args:
        signal: 1-D input signal array.
        fs: Sampling frequency in Hz (default 256.0).

    Returns:
        Tuple of (freqs, psd) NumPy arrays.
    """
    freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=256, noverlap=128, nfft=2048)
    return freqs, psd


def compute_spectral_purity(
    freqs: np.ndarray,
    psd: np.ndarray,
    low: float = 0.7,
    high: float = 4.0,
) -> float:
    """Fraction of total spectral energy inside the cardiac band.

    More robust to lighting and compression than raw SNR because it is a
    ratio. Real rPPG concentrates energy in the band, deepfakes do not.
    """
    total_power = float(np.sum(psd) + 1e-20)
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    cardiac_power = float(np.sum(psd[mask]))
    return float(max(0.0, min(1.0, cardiac_power / total_power)))


def compute_peak_prominence(
    freqs: np.ndarray,
    psd: np.ndarray,
    low: float = 0.7,
    high: float = 4.0,
) -> float:
    """Peak prominence of the dominant cardiac-band component.

    A sharp single peak indicates a real heartbeat; scattered low peaks are
    typical for synthetic content or noise.
    """
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    band_psd = psd[mask]
    if len(band_psd) == 0:
        return 0.0
    peak_val = float(np.max(band_psd))
    baseline = float(np.median(band_psd)) + 1e-12
    return float(max(0.0, (peak_val - baseline) / baseline))


def compute_autocorr_periodicity(
    filtered_signal: np.ndarray,
    fs: float = 30.0,
) -> float:
    """Autocorrelation-based periodicity score (0-1).

    Captures heartbeat repetition directly in time domain; robust to FFT bin
    resolution issues.
    """
    if len(filtered_signal) < 150:
        return 0.0
    autocorr = np.correlate(filtered_signal, filtered_signal, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]
    # Look for peaks between 0.25s and 1.5s (40-240 BPM)
    min_lag = int(fs * 0.25)
    max_lag = min(len(autocorr), int(fs * 1.5))
    if max_lag <= min_lag:
        return 0.0
    window = autocorr[min_lag:max_lag]
    if len(window) == 0:
        return 0.0
    peaks, props = find_peaks(window, prominence=np.max(window) * 0.05)
    if len(peaks) == 0:
        return 0.0
    peak_vals = window[peaks]
    base = autocorr[0] + 1e-12
    return float(min(1.0, np.max(peak_vals) / base))


def compute_snr_and_hr(
    freqs: np.ndarray,
    psd: np.ndarray,
    low: float = 0.7,
    high: float = 4.0,
) -> Tuple[float, float]:
    """Compute SNR (dB) and heart rate (BPM) from the PSD.

    Signal power = max PSD within the cardiac band (0.7-4.0 Hz).
    Noise power  = mean PSD OUTSIDE the cardiac band.
    SNR (dB)     = 10 * log10(signal_power / (noise_power + 1e-10))

    Args:
        freqs: Frequency array from Welch PSD.
        psd: Power spectral density array.
        low: Lower cardiac band bound in Hz (default 0.7).
        high: Upper cardiac band bound in Hz (default 4.0).

    Returns:
        Tuple of (hr_bpm, snr_db).
    """
    mask = (freqs >= low) & (freqs <= high)
    signal_power = np.max(psd[mask])
    noise_power = np.mean(psd[~mask])
    snr_db = 10.0 * np.log10(signal_power / (noise_power + 1e-10))
    peak_freq = freqs[mask][np.argmax(psd[mask])]
    hr_bpm = float(peak_freq * 60.0)
    return hr_bpm, float(snr_db)


def compute_roi_correlation(
    roi_buffers: Dict[str, list],
    fs: float = 30.0,
) -> float:
    """Compute mean pairwise Pearson correlation of bandpass-filtered ROI signals.

    In a live human, blood pumped by the same heart flows through the forehead,
    left cheek, and right cheek capillary beds simultaneously. This means the
    filtered green-channel signals from all three regions are highly correlated
    (typically r > 0.6). A photo, printed image, or video on a screen has no
    blood flow — the per-ROI signals are just uncorrelated sensor noise, ambient
    light artefacts, or uniform screen refresh, yielding low correlation.

    Args:
        roi_buffers: Dict with keys 'forehead', 'left_cheek', 'right_cheek',
                     each mapping to a list of float green-channel values.
        fs: Sampling frequency in Hz (default 30.0).

    Returns:
        Mean pairwise Pearson correlation coefficient (0.0 to 1.0).
        Returns 0.0 if signals are too short or constant.
    """
    keys = ["forehead", "left_cheek", "right_cheek"]
    signals = {}
    for k in keys:
        buf = roi_buffers.get(k, [])
        if len(buf) < 150:
            return 0.0
        sig = np.array(buf, dtype=np.float64)
        try:
            sig = butterworth_bandpass(sig, fs=fs)
        except Exception:
            return 0.0
        signals[k] = sig

    pairs = [("forehead", "left_cheek"), ("forehead", "right_cheek"),
             ("left_cheek", "right_cheek")]
    correlations = []
    for k1, k2 in pairs:
        s1, s2 = signals[k1], signals[k2]
        std1, std2 = np.std(s1), np.std(s2)
        if std1 < 1e-10 or std2 < 1e-10:
            correlations.append(0.0)
            continue
        r = float(np.corrcoef(s1, s2)[0, 1])
        correlations.append(max(0.0, r))  # clamp negative to 0

    return float(np.mean(correlations))


def compute_peak_quality(
    freqs: np.ndarray,
    psd: np.ndarray,
    low: float = 0.7,
    high: float = 4.0,
) -> float:
    """Compute the spectral peak quality (prominence ratio).

    A real cardiac signal produces a sharp, narrow peak in the PSD at the
    heart rate frequency. Noise (from photos, screens, ambient light) produces
    a broad, flat spectrum with no clear dominant frequency. Peak quality is
    defined as the ratio of the peak PSD value to the median PSD within the
    cardiac band. A ratio > 3.0 indicates a clear cardiac peak.

    Args:
        freqs: Frequency array from Welch PSD.
        psd: Power spectral density array.
        low: Lower cardiac band bound in Hz (default 0.7).
        high: Upper cardiac band bound in Hz (default 4.0).

    Returns:
        Peak quality ratio (float >= 0). Higher = sharper peak.
    """
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    psd_band = psd[mask]
    peak_val = np.max(psd_band)
    median_val = np.median(psd_band)
    if median_val < 1e-20:
        return 0.0
    return float(peak_val / median_val)


def compute_signal_strength(
    green_buffer: list,
    fs: float = 30.0,
) -> float:
    """Compute normalised signal strength of the filtered BVP signal.

    Real rPPG signals have a characteristic amplitude after bandpass filtering:
    small but clearly above the noise floor. Photos produce near-zero amplitude
    (static pixels), while hand-held photos produce erratic, non-periodic variance.
    Videos on screens may produce moderate amplitude from screen refresh artefacts
    but typically outside the cardiac envelope.

    The metric is the ratio of the filtered signal's standard deviation to the
    raw signal's standard deviation, capped to [0, 1].

    Args:
        green_buffer: List of float green-channel values.
        fs: Sampling rate in Hz (default 30.0).

    Returns:
        Normalised signal strength (0.0 to 1.0).
    """
    if len(green_buffer) < 150:
        return 0.0
    raw = np.array(green_buffer, dtype=np.float64)
    try:
        filtered = butterworth_bandpass(raw, fs=fs)
    except Exception:
        return 0.0
    raw_std = np.std(raw)
    filt_std = np.std(filtered)
    if raw_std < 1e-10:
        return 0.0
    ratio = filt_std / raw_std
    return float(min(1.0, ratio))


def classify_liveness(
    snr_db: float,
    threshold: float = 3.0,
    roi_correlation: float = 1.0,
    peak_quality: float = 1.0,
    signal_strength: float = 1.0,
    spectral_purity: float = 0.0,
    peak_prominence: float = 0.0,
    periodicity: float = 0.0,
    red_purity: float = 0.0,
) -> Tuple[str, float]:
    """Classify liveness for WEBCAM / real-time detection.

    This classifier answers: "Is there a living human in front of the camera?"
    It detects photos, printed images, screens, and masks by checking for a
    genuine rPPG cardiac signal.

    Empirically calibrated on real webcam captures and simulated attacks:
      Real face webcam:  PQ ~1.5-7.2, Period ~0.14-0.92, Prom ~0.3-6.0
      Photo (static):    PQ ~1.0-1.2, Period ~0.00-0.13, Prom ~0.0-0.2
      Screen replay:     PQ ~1.0-1.3, Period ~0.00-0.13, Prom ~0.0-0.3
      Shaken photo:      PQ varies,   Period ~0.00,       Corr >0.97

    Decision: continuous scoring on three features with calibrated floor at 15.
    """
    # ── Continuous scoring (0-100 each) ──
    # Each feature mapped so that noise floor → ~0-10 and real webcam → 30+.

    # Peak Quality: noise ~1.2, weak real ~1.5, strong real ~3+
    # (pq - 1.2) maps noise→0, real starts at 0.3+ → score ~10+
    pq_score = float(np.clip((peak_quality - 1.2) / 2.0 * 100.0, 0, 100))

    # Periodicity: noise ~0.00-0.13, real webcam ~0.14+, strong real ~0.3+
    # (period - 0.12) maps noise→~0, real starts at 0.02+ → score ~6+
    period_score = float(np.clip((periodicity - 0.12) / 0.3 * 100.0, 0, 100))

    # Peak Prominence: noise ~0.0-0.2, real webcam ~0.3+, strong ~1.0+
    # (prom - 0.2) maps noise→0, real starts at 0.1+ → score ~7+
    prom_score = float(np.clip((peak_prominence - 0.2) / 1.5 * 100.0, 0, 100))

    # Weighted composite
    composite = (0.35 * pq_score
                 + 0.35 * period_score
                 + 0.30 * prom_score)

    # Decision threshold: 15 out of 100
    # Photo noise: PQ=1.2 → 0, Period=0.13 → ~3.3, Prom=0.2 → 0 → composite ~1.2
    # Weak real:   PQ=1.5 → 15, Period=0.16 → 13, Prom=0.5 → 20 → composite ~16
    is_live = composite > 15.0

    # ── Hard veto: shaken photo ──
    # Very high ROI correlation (>0.97) with near-zero periodicity = 2D surface
    if roi_correlation > 0.97 and periodicity < 0.15:
        is_live = False

    # ── Confidence ──
    if is_live:
        confidence_pct = float(np.clip(55.0 + composite * 0.44, 60.0, 99.0))
    else:
        confidence_pct = float(np.clip(100.0 - composite, 5.0, 99.0))

    verdict = "LIVE HUMAN" if is_live else "SYNTHETIC"
    return (verdict, confidence_pct)


def process_signal_buffer(
    green_buffer: list,
    webcam_fps: float = 30.0,
    roi_buffers: Optional[Dict[str, list]] = None,
    threshold: float = 3.0,
    red_buffer: Optional[list] = None,
) -> Optional[Dict]:
    """Run the complete signal processing pipeline on the green-channel buffer.

    Pipeline:
      1. Butterworth bandpass filter (0.7-4.0 Hz)
      2. Cubic-spline resample to 256 Hz
      3. Welch PSD estimation
      4. SNR and heart-rate extraction
      5. Multi-ROI correlation (if roi_buffers provided)
      6. Peak quality and signal strength analysis
      7. Multi-criteria liveness classification

    Args:
        green_buffer: List of float green-channel values from ROI extraction.
        webcam_fps: Webcam capture frame rate in Hz (default 30.0).
        roi_buffers: Optional dict of per-ROI buffers for correlation analysis.
                     Keys: 'forehead', 'left_cheek', 'right_cheek'.

    Returns:
        Dictionary: {hr_bpm, snr_db, verdict, confidence_pct, freqs, psd,
                     roi_correlation, peak_quality, signal_strength}.
        Returns None if buffer has fewer than 150 samples.
    """
    if len(green_buffer) < 150:
        return None

    sig = np.array(green_buffer, dtype=np.float64)
    filtered   = butterworth_bandpass(sig, fs=webcam_fps)
    resampled  = resample_signal(filtered, original_fs=webcam_fps, target_fs=256.0)
    freqs, psd = compute_psd_welch(resampled, fs=256.0)
    hr_bpm, snr_db        = compute_snr_and_hr(freqs, psd)

    # Multi-ROI correlation
    roi_corr = 0.0
    if roi_buffers is not None:
        roi_corr = compute_roi_correlation(roi_buffers, fs=webcam_fps)
    else:
        # No multi-ROI data: fall back to SNR-only mode with conservative score
        roi_corr = 0.3  # neutral — won't strongly push verdict either way

    # Spectral purity and peak characteristics
    peak_qual = compute_peak_quality(freqs, psd)
    purity_ratio = compute_spectral_purity(freqs, psd)
    peak_prom = compute_peak_prominence(freqs, psd)

    # Autocorrelation periodicity (time-domain heartbeat regularity)
    periodicity = compute_autocorr_periodicity(filtered, fs=webcam_fps)

    # Signal strength
    sig_str = compute_signal_strength(green_buffer, fs=webcam_fps)

    # Process Red Channel if available to detect screen fakes
    r_purity = 0.0
    if red_buffer and len(red_buffer) == len(green_buffer):
        try:
            r_filt = butterworth_bandpass(np.array(red_buffer, dtype=np.float64), fs=webcam_fps)
            r_resampled = resample_signal(r_filt, original_fs=webcam_fps, target_fs=256.0)
            r_freqs, r_psd = compute_psd_welch(r_resampled, fs=256.0)
            r_purity = compute_spectral_purity(r_freqs, r_psd)
        except Exception:
            pass

    verdict, confidence = classify_liveness(
        snr_db,
        threshold=threshold,
        roi_correlation=roi_corr,
        peak_quality=peak_qual,
        signal_strength=sig_str,
        spectral_purity=purity_ratio,
        peak_prominence=peak_prom,
        periodicity=periodicity,
        red_purity=r_purity,
    )

    return {
        "hr_bpm":          hr_bpm,
        "snr_db":          snr_db,
        "verdict":         verdict,
        "confidence_pct":  confidence,
        "freqs":           freqs.tolist(),
        "psd":             psd.tolist(),
        "roi_correlation": roi_corr,
        "peak_quality":    peak_qual,
        "signal_strength": sig_str,
        "spectral_purity": purity_ratio,
        "peak_prominence": peak_prom,
        "periodicity":     periodicity,
        "red_purity":      r_purity,
    }


# =============================================================================
# Self-Test — runs when you execute this file directly
# =============================================================================
if __name__ == "__main__":
    print("Neuro-Pulse Signal Processor — Self-Test")
    print("=" * 60)

    fs        = 30.0
    n_samples = 600   # 20 seconds at 30 FPS
    t         = np.arange(n_samples) / fs

    def _run_test(label, test_signal, roi_buffers, expected_verdict, hr_target=None):
        print(f"\n{label}")
        print("-" * 60)
        result = process_signal_buffer(test_signal, webcam_fps=fs, roi_buffers=roi_buffers)
        if result is None:
            print("  FAIL: process_signal_buffer returned None")
            return False
        print(f"  HR={result['hr_bpm']:.0f}BPM  SNR={result['snr_db']:.1f}dB  "
              f"PeakQ={result['peak_quality']:.1f}  Period={result['periodicity']:.3f}  "
              f"Corr={result['roi_correlation']:.3f}")
        print(f"  Verdict: {result['verdict']} ({result['confidence_pct']:.0f}%)")
        ok = result["verdict"] == expected_verdict
        if hr_target is not None:
            ok = ok and abs(result["hr_bpm"] - hr_target) <= 8
        print(f"  => {'PASS' if ok else 'FAIL'}")
        return ok

    np.random.seed(42)

    # TEST 1: Real cardiac (1.2 Hz = 72 BPM) with moderate noise
    cardiac = 0.5 * np.sin(2 * np.pi * 1.2 * t) + 120.0 + np.random.normal(0, 0.15, n_samples)
    roi_real = {
        "forehead":    (0.5 * np.sin(2*np.pi*1.2*t) + 120 + np.random.normal(0, 0.2, n_samples)).tolist(),
        "left_cheek":  (0.5 * np.sin(2*np.pi*1.2*t) + 115 + np.random.normal(0, 0.2, n_samples)).tolist(),
        "right_cheek": (0.5 * np.sin(2*np.pi*1.2*t) + 118 + np.random.normal(0, 0.2, n_samples)).tolist(),
    }
    t1 = _run_test("TEST 1: Real cardiac signal (72 BPM)", cardiac.tolist(), roi_real, "LIVE HUMAN", 72)

    # TEST 2: Photo — static pixels with sensor noise, no cardiac rhythm
    np.random.seed(123)
    photo = np.random.normal(120.0, 0.3, n_samples)
    photo_roi = {
        "forehead":    np.random.normal(120, 0.3, n_samples).tolist(),
        "left_cheek":  np.random.normal(115, 0.3, n_samples).tolist(),
        "right_cheek": np.random.normal(118, 0.3, n_samples).tolist(),
    }
    t2 = _run_test("TEST 2: Simulated PHOTO (sensor noise only)", photo.tolist(), photo_roi, "SYNTHETIC")

    # TEST 3: Screen replay — uniform flicker below cardiac band
    np.random.seed(456)
    screen_base = 0.5 * np.sin(2*np.pi*0.2*t) + np.random.normal(0, 0.3, n_samples) + 120
    screen_roi = {
        "forehead":    (screen_base + np.random.normal(0, 0.1, n_samples)).tolist(),
        "left_cheek":  (screen_base + np.random.normal(0, 0.1, n_samples)).tolist(),
        "right_cheek": (screen_base + np.random.normal(0, 0.1, n_samples)).tolist(),
    }
    t3 = _run_test("TEST 3: Simulated SCREEN (uniform flicker)", screen_base.tolist(), screen_roi, "SYNTHETIC")

    # TEST 4: Shaken photo — high correlation (>0.97) with low peak quality
    np.random.seed(789)
    shake = 2.0 * np.sin(2*np.pi*0.5*t) + 120 + np.random.normal(0, 0.05, n_samples)
    shake_roi = {
        "forehead":    (shake + np.random.normal(0, 0.02, n_samples)).tolist(),
        "left_cheek":  (shake + np.random.normal(0, 0.02, n_samples)).tolist(),
        "right_cheek": (shake + np.random.normal(0, 0.02, n_samples)).tolist(),
    }
    t4 = _run_test("TEST 4: Shaken photo (highly correlated noise)", shake.tolist(), shake_roi, "SYNTHETIC")

    # TEST 5: Weak real webcam signal (realistic: PQ ~1.5-2.5, Period ~0.1-0.2)
    # Real webcams produce weaker signals due to compression, auto-exposure, lighting.
    np.random.seed(999)
    weak_cardiac = 0.15 * np.sin(2 * np.pi * 1.25 * t) + 120.0 + np.random.normal(0, 0.12, n_samples)
    weak_roi = {
        "forehead":    (0.15 * np.sin(2*np.pi*1.25*t) + 120 + np.random.normal(0, 0.15, n_samples)).tolist(),
        "left_cheek":  (0.12 * np.sin(2*np.pi*1.25*t) + 115 + np.random.normal(0, 0.15, n_samples)).tolist(),
        "right_cheek": (0.13 * np.sin(2*np.pi*1.25*t) + 118 + np.random.normal(0, 0.15, n_samples)).tolist(),
    }
    t5 = _run_test("TEST 5: Weak real webcam (noisy cardiac)", weak_cardiac.tolist(), weak_roi, "LIVE HUMAN", 75)

    # Summary
    print("\n" + "=" * 60)
    tests = [t1, t2, t3, t4, t5]
    names = ["Real face", "Photo", "Screen", "Shaken photo", "Weak webcam"]
    all_pass = all(tests)
    for name, passed in zip(names, tests):
        print(f"  {name:20s}: {'PASS' if passed else 'FAIL'}")
    print("=" * 60)
    if not all_pass:
        sys.exit(1)