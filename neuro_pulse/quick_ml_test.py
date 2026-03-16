"""Quick ML classifier test on 10 real + 10 fake FF++ videos."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import glob
from src.deepfake_detector import classify_video as ml_classify_video

real_dir = "/Users/likhith./pyVHR_rppg/ff_downloads/original_sequences/youtube/c40/videos"
fake_dir = "/Users/likhith./pyVHR_rppg/ff_downloads/manipulated_sequences/Deepfakes/c40/videos"

real_files = sorted(glob.glob(os.path.join(real_dir, "*.mp4")))[:10]
fake_files = sorted(glob.glob(os.path.join(fake_dir, "*.mp4")))[:10]

tp = tn = fp = fn = err = 0

for f in real_files:
    r = ml_classify_video(f)
    v = r["verdict"]
    m = r.get("method", "?")
    c = r.get("confidence_pct", 0)
    name = os.path.basename(f)
    if v == "REAL": tp += 1
    elif v == "ERROR": err += 1
    else: fn += 1
    print(f"  REAL {name:20s} -> {v:6s} ({c:.0f}%) [{m}]")

for f in fake_files:
    r = ml_classify_video(f)
    v = r["verdict"]
    m = r.get("method", "?")
    c = r.get("confidence_pct", 0)
    name = os.path.basename(f)
    if v == "FAKE": tn += 1
    elif v == "ERROR": err += 1
    else: fp += 1
    print(f"  FAKE {name:20s} -> {v:6s} ({c:.0f}%) [{m}]")

total = tp + tn + fp + fn
acc = (tp + tn) / total * 100 if total > 0 else 0
print(f"\nTP={tp} TN={tn} FP={fp} FN={fn} ERR={err}")
print(f"Accuracy: {acc:.1f}%")
