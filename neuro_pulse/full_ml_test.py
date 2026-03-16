"""Full ML classifier test on ALL 100 FF++ videos."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import glob
from src.deepfake_detector import classify_video as ml_classify_video

real_dir = "/Users/likhith./pyVHR_rppg/ff_downloads/original_sequences/youtube/c40/videos"
fake_dir = "/Users/likhith./pyVHR_rppg/ff_downloads/manipulated_sequences/Deepfakes/c40/videos"

real_files = sorted(glob.glob(os.path.join(real_dir, "*.mp4")))
fake_files = sorted(glob.glob(os.path.join(fake_dir, "*.mp4")))

print(f"Testing ML classifier on {len(real_files)} real + {len(fake_files)} fake videos")

tp = tn = fp = fn = err = 0
misclassified = []

for i, f in enumerate(real_files):
    r = ml_classify_video(f)
    v = r["verdict"]
    if v == "REAL": tp += 1
    elif v == "ERROR": err += 1
    else: fn += 1; misclassified.append(("REAL->FAKE", os.path.basename(f), r))
    if (i + 1) % 10 == 0:
        print(f"  Real: {i+1}/{len(real_files)}")

for i, f in enumerate(fake_files):
    r = ml_classify_video(f)
    v = r["verdict"]
    if v == "FAKE": tn += 1
    elif v == "ERROR": err += 1
    else: fp += 1; misclassified.append(("FAKE->REAL", os.path.basename(f), r))
    if (i + 1) % 10 == 0:
        print(f"  Fake: {i+1}/{len(fake_files)}")

total = tp + tn + fp + fn
acc = (tp + tn) / total * 100 if total > 0 else 0

print(f"\n{'='*50}")
print(f"ML Classifier Results (all {len(real_files)+len(fake_files)} videos)")
print(f"{'='*50}")
print(f"  TP (Real->Real): {tp}")
print(f"  TN (Fake->Fake): {tn}")
print(f"  FP (Fake->Real): {fp}")
print(f"  FN (Real->Fake): {fn}")
print(f"  Errors: {err}")
print(f"  Accuracy: {acc:.1f}%")
print(f"  TPR: {tp/(tp+fn)*100 if (tp+fn)>0 else 0:.1f}%")
print(f"  TNR: {tn/(tn+fp)*100 if (tn+fp)>0 else 0:.1f}%")

if misclassified:
    print(f"\nMisclassified:")
    for typ, name, r in misclassified:
        print(f"  {typ}: {name} ({r.get('method','')} conf={r.get('confidence_pct',0):.0f}%)")
