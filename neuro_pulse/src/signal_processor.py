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
    freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=256, noverlap=128)
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
) -> Tuple[str, float]:
    """Classify liveness using multi-criteria scoring.

    Combines four physics-based metrics:
      1. SNR — cardiac band signal power vs out-of-band noise
      2. Multi-ROI correlation — correlated cardiac signals across face regions
         prove blood is flowing through all capillary beds from the same heart
      3. Peak quality — sharp narrow spectral peak = real cardiac rhythm,
         broad flat spectrum = noise / artifacts
      4. Signal strength — real rPPG has characteristic amplitude; photos and
         screens produce near-zero or abnormally high variance

    A photo/screen fails because:
      - No blood flow → ROI signals are uncorrelated noise → low correlation
      - No cardiac rhythm → no sharp spectral peak → low peak quality
      - Static pixels → near-zero signal variance → low signal strength

    Args:
        snr_db: Signal-to-noise ratio in decibels.
        threshold: SNR threshold for live classification (default 3.0 dB).
        roi_correlation: Mean pairwise correlation of filtered per-ROI signals (0-1).
        peak_quality: Spectral peak prominence ratio (0+). Above 3.0 is strong.
        signal_strength: Normalised signal amplitude metric (0-1). Real rPPG: 0.3-0.9.

    Returns:
        Tuple of (verdict, confidence_pct).
        verdict is 'LIVE HUMAN' or 'SYNTHETIC'.
        confidence_pct is a float 0-100.
    """
    # --- Individual pass/fail checks (slightly conservative defaults) ---
    snr_pass = snr_db >= threshold
    purity_pass = spectral_purity >= 0.12
    corr_pass = roi_correlation >= 0.35
    peak_pass = peak_quality >= 2.0
    prom_pass = peak_prominence >= 0.4
    strength_pass = signal_strength >= 0.12
    periodic_pass = periodicity >= 0.25

    # --- Weighted liveness score (0-100) ---
    snr_score = float(min(100.0, max(0.0, (snr_db / 8.0) * 100.0)))
    corr_score = float(min(100.0, max(0.0, roi_correlation * 100.0)))
    peak_score = float(min(100.0, max(0.0, (peak_quality / 8.0) * 100.0)))
    purity_score = float(min(100.0, max(0.0, spectral_purity * 100.0)))
    prom_score = float(min(100.0, max(0.0, peak_prominence * 50.0)))
    strength_score = float(min(100.0, max(0.0, signal_strength * 100.0)))
    periodic_score = float(min(100.0, max(0.0, periodicity * 100.0)))

    composite = (
        0.18 * snr_score
        + 0.18 * purity_score
        + 0.17 * corr_score
        + 0.17 * peak_score
        + 0.10 * prom_score
        + 0.10 * strength_score
        + 0.10 * periodic_score
    )

    # HARD REQUIREMENTS (physics-based reasoning):
    # 1. Peak sharpness and correlation must both be present.
    # 2. Either (SNR AND purity) or (purity + prominence + periodicity) must hold.
    base_live = peak_pass and corr_pass and prom_pass
    combo_a = snr_pass and purity_pass
    combo_b = purity_pass and periodic_pass and prom_pass
    is_live = base_live and (combo_a or combo_b)

    confidence_pct = float(min(100.0, max(0.0, composite)))

    verdict = "LIVE HUMAN" if is_live else "SYNTHETIC"
    return (verdict, confidence_pct)


def process_signal_buffer(
    green_buffer: list,
    webcam_fps: float = 30.0,
    roi_buffers: Optional[Dict[str, list]] = None,
    threshold: float = 3.0,
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

    verdict, confidence = classify_liveness(
        snr_db,
        threshold=threshold,
        roi_correlation=roi_corr,
        peak_quality=peak_qual,
        signal_strength=sig_str,
        spectral_purity=purity_ratio,
        peak_prominence=peak_prom,
        periodicity=periodicity,
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
    }


# =============================================================================
# Self-Test — runs when you execute this file directly
# =============================================================================
if __name__ == "__main__":
    print("Neuro-Pulse Signal Processor — Self-Test")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # TEST 1: Real cardiac signal → LIVE HUMAN
    # -------------------------------------------------------------------------
    print("\nTEST 1: Simulated REAL cardiac signal (correlated ROIs)")
    print("-" * 60)

    np.random.seed(42)

    fs          = 30.0
    n_samples   = 600       # 20 seconds at 30 FPS
    target_hz   = 1.0       # 1.0 Hz = exactly 60 BPM
    target_bpm  = 60

    t            = np.arange(n_samples) / fs
    clean_signal = np.sin(2 * np.pi * target_hz * t)
    noise        = np.random.normal(0, 0.02, n_samples)
    test_signal  = (clean_signal + noise).tolist()

    # Simulate correlated ROI signals (same cardiac rhythm + small noise)
    roi_buffers = {
        "forehead":    (clean_signal + np.random.normal(0, 0.03, n_samples)).tolist(),
        "left_cheek":  (clean_signal + np.random.normal(0, 0.03, n_samples)).tolist(),
        "right_cheek": (clean_signal + np.random.normal(0, 0.03, n_samples)).tolist(),
    }

    result = process_signal_buffer(test_signal, webcam_fps=fs, roi_buffers=roi_buffers)

    if result is None:
        print("FAIL: process_signal_buffer returned None")
        sys.exit(1)

    print(f"  Detected HR      : {result['hr_bpm']:.1f} BPM  (expected {target_bpm})")
    print(f"  SNR              : {result['snr_db']:.2f} dB")
    print(f"  ROI Correlation  : {result['roi_correlation']:.3f}")
    print(f"  Peak Quality     : {result['peak_quality']:.2f}")
    print(f"  Signal Strength  : {result['signal_strength']:.3f}")
    print(f"  Verdict          : {result['verdict']}")
    print(f"  Confidence       : {result['confidence_pct']:.1f}%")

    hr_ok      = abs(result['hr_bpm'] - target_bpm) <= 5
    verdict_ok = result['verdict'] == "LIVE HUMAN"
    test1_pass = hr_ok and verdict_ok

    if test1_pass:
        print("  => TEST 1: PASS")
    else:
        print("  => TEST 1: FAIL")
        if not hr_ok:
            print(f"    HR {result['hr_bpm']:.1f} not within 5 BPM of {target_bpm}")
        if not verdict_ok:
            print(f"    Verdict '{result['verdict']}' should be 'LIVE HUMAN'")

    # -------------------------------------------------------------------------
    # TEST 2: Photo / static image → SYNTHETIC
    # Simulates what happens when a photo is held up to the camera:
    #   - No blood flow → each ROI is just uncorrelated sensor noise
    #   - No cardiac rhythm → no sharp spectral peak
    #   - Very low signal amplitude after bandpass filtering
    # -------------------------------------------------------------------------
    print(f"\nTEST 2: Simulated PHOTO signal (uncorrelated noise)")
    print("-" * 60)

    np.random.seed(123)

    # Photo: each ROI is just independent low-amplitude noise (no shared signal)
    photo_signal = np.random.normal(0, 0.5, n_samples).tolist()
    photo_roi = {
        "forehead":    np.random.normal(120, 0.3, n_samples).tolist(),
        "left_cheek":  np.random.normal(115, 0.3, n_samples).tolist(),
        "right_cheek": np.random.normal(118, 0.3, n_samples).tolist(),
    }

    result2 = process_signal_buffer(photo_signal, webcam_fps=fs, roi_buffers=photo_roi)

    if result2 is None:
        print("FAIL: process_signal_buffer returned None")
        sys.exit(1)

    print(f"  SNR              : {result2['snr_db']:.2f} dB")
    print(f"  ROI Correlation  : {result2['roi_correlation']:.3f}")
    print(f"  Peak Quality     : {result2['peak_quality']:.2f}")
    print(f"  Signal Strength  : {result2['signal_strength']:.3f}")
    print(f"  Verdict          : {result2['verdict']}")
    print(f"  Confidence       : {result2['confidence_pct']:.1f}%")

    test2_pass = result2['verdict'] == "SYNTHETIC"
    if test2_pass:
        print("  => TEST 2: PASS")
    else:
        print("  => TEST 2: FAIL — photo was classified as LIVE HUMAN!")

    # -------------------------------------------------------------------------
    # TEST 3: Screen replay → SYNTHETIC
    # Simulates a video on a screen: uniform illumination change across all ROIs
    # (screen brightness changes affect all pixels equally) but no real cardiac
    # frequency content after bandpass filtering.
    # -------------------------------------------------------------------------
    print(f"\nTEST 3: Simulated SCREEN signal (uniform flicker, no cardiac)")
    print("-" * 60)

    np.random.seed(456)

    # Screen: slow ambient flicker (0.2 Hz, below cardiac band) + noise
    screen_base = 0.5 * np.sin(2 * np.pi * 0.2 * t) + np.random.normal(0, 0.3, n_samples)
    screen_signal = screen_base.tolist()
    screen_roi = {
        "forehead":    (screen_base + np.random.normal(0, 0.1, n_samples)).tolist(),
        "left_cheek":  (screen_base + np.random.normal(0, 0.1, n_samples)).tolist(),
        "right_cheek": (screen_base + np.random.normal(0, 0.1, n_samples)).tolist(),
    }

    result3 = process_signal_buffer(screen_signal, webcam_fps=fs, roi_buffers=screen_roi)

    if result3 is None:
        print("FAIL: process_signal_buffer returned None")
        sys.exit(1)

    print(f"  SNR              : {result3['snr_db']:.2f} dB")
    print(f"  ROI Correlation  : {result3['roi_correlation']:.3f}")
    print(f"  Peak Quality     : {result3['peak_quality']:.2f}")
    print(f"  Signal Strength  : {result3['signal_strength']:.3f}")
    print(f"  Verdict          : {result3['verdict']}")
    print(f"  Confidence       : {result3['confidence_pct']:.1f}%")

    test3_pass = result3['verdict'] == "SYNTHETIC"
    if test3_pass:
        print("  => TEST 3: PASS")
    else:
        print("  => TEST 3: FAIL — screen was classified as LIVE HUMAN!")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    all_pass = test1_pass and test2_pass and test3_pass
    if all_pass:
        print("  RESULT: ALL 3 TESTS PASSED")
        print("    Test 1 (Real face)  → LIVE HUMAN   OK")
        print("    Test 2 (Photo)      → SYNTHETIC    OK")
        print("    Test 3 (Screen)     → SYNTHETIC    OK")
    else:
        print("  RESULT: SOME TESTS FAILED")
        print(f"    Test 1 (Real face)  : {'PASS' if test1_pass else 'FAIL'}")
        print(f"    Test 2 (Photo)      : {'PASS' if test2_pass else 'FAIL'}")
        print(f"    Test 3 (Screen)     : {'PASS' if test3_pass else 'FAIL'}")
        sys.exit(1)
    print("=" * 60)