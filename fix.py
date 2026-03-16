import os
import re

path = '/Users/likhith./pyVHR_rppg/neuro_pulse/src/signal_processor.py'
with open(path, 'r') as f:
    orig = f.read()

new_func = """def classify_liveness(
    snr_db: float,
    threshold: float = 3.0,
    roi_correlation: float = 1.0,
    peak_quality: float = 1.0,
    signal_strength: float = 1.0,
    spectral_purity: float = 0.0,
    peak_prominence: float = 0.0,
    periodicity: float = 0.0,
) -> Tuple[str, float]:
    \"\"\"Classify liveness using multi-criteria scoring.\"\"\"
    score = 0.0
    
    score += (min(1.0, spectral_purity / 0.5) * 40.0)
    score += min(30.0, max(0.0, (snr_db / 5.0) * 30.0))
    score += (min(1.0, max(0.0, roi_correlation)) * 15.0)
    score += (min(1.0, max(0.0, periodicity / 0.2)) * 15.0)

    is_live = False
    
    if score >= 50.0:
        is_live = True
        
    if spectral_purity > 0.4 and roi_correlation > 0.3:
        is_live = True

    if periodicity < 0.15 and roi_correlation < 0.3:
        score -= 25.0
        if score < 55.0:
            is_live = False

    if is_live:
        verdict = "LIVE HUMAN"
        confidence_pct = float(min(100.0, score))
    else:
        verdict = "SYNTHETIC"
        confidence_pct = float(max(50.0, 100.0 - score))

    return verdict, confidence_pct
"""

old_func_pattern = re.compile(r"def classify_liveness\(.*?-> Tuple\[str, float\]:.*?return verdict, float\(confidence_pct\)", re.DOTALL | re.MULTILINE)

new_content = old_func_pattern.sub(new_func, orig)

with open(path, 'w') as f:
    f.write(new_content)
