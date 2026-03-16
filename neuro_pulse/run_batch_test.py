"""
Full batch test: Run both ML classifier and liveness classifier on all FF++ videos.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import glob
from src.deepfake_detector import classify_video as ml_classify_video
from src.batch_analyzer import analyze_video

def main():
    real_dir = "/Users/likhith./pyVHR_rppg/ff_downloads/original_sequences/youtube/c40/videos"
    fake_dir = "/Users/likhith./pyVHR_rppg/ff_downloads/manipulated_sequences/Deepfakes/c40/videos"

    real_files = sorted(glob.glob(os.path.join(real_dir, "*.mp4")))
    fake_files = sorted(glob.glob(os.path.join(fake_dir, "*.mp4")))

    print(f"=== ML DEEPFAKE CLASSIFIER TEST ===")
    print(f"Real: {len(real_files)} videos, Fake: {len(fake_files)} videos")
    print()

    tp = tn = fp = fn = err = 0

    for i, f in enumerate(real_files):
        result = ml_classify_video(f)
        v = result["verdict"]
        if v == "ERROR":
            err += 1
        elif v == "REAL":
            tp += 1
        else:
            fn += 1
        if (i + 1) % 10 == 0:
            print(f"  Real: {i+1}/{len(real_files)} processed...")

    for i, f in enumerate(fake_files):
        result = ml_classify_video(f)
        v = result["verdict"]
        if v == "ERROR":
            err += 1
        elif v == "FAKE":
            tn += 1
        else:
            fp += 1
        if (i + 1) % 10 == 0:
            print(f"  Fake: {i+1}/{len(fake_files)} processed...")

    total = tp + tn + fp + fn
    acc = (tp + tn) / total * 100 if total > 0 else 0
    print(f"\nML Classifier Results:")
    print(f"  TP={tp} TN={tn} FP={fp} FN={fn} ERR={err}")
    print(f"  Accuracy: {acc:.1f}%")
    print(f"  TPR: {tp/(tp+fn)*100 if (tp+fn)>0 else 0:.1f}%")
    print(f"  TNR: {tn/(tn+fp)*100 if (tn+fp)>0 else 0:.1f}%")

    # Also test liveness classifier on a few real + fake
    print(f"\n\n=== LIVENESS CLASSIFIER (batch_analyzer) ===")
    tp2 = tn2 = fp2 = fn2 = err2 = 0

    for i, f in enumerate(real_files):
        result = analyze_video(f)
        v = result["verdict"]
        if v == "ERROR":
            err2 += 1
        elif v in ("LIVE HUMAN", "REAL"):
            tp2 += 1
        else:
            fn2 += 1
        if (i + 1) % 10 == 0:
            print(f"  Real: {i+1}/{len(real_files)} processed...")

    for i, f in enumerate(fake_files):
        result = analyze_video(f)
        v = result["verdict"]
        if v == "ERROR":
            err2 += 1
        elif v in ("SYNTHETIC", "FAKE"):
            tn2 += 1
        else:
            fp2 += 1
        if (i + 1) % 10 == 0:
            print(f"  Fake: {i+1}/{len(fake_files)} processed...")

    total2 = tp2 + tn2 + fp2 + fn2
    acc2 = (tp2 + tn2) / total2 * 100 if total2 > 0 else 0
    print(f"\nBatch Analyzer Results (uses ML classifier internally):")
    print(f"  TP={tp2} TN={tn2} FP={fp2} FN={fn2} ERR={err2}")
    print(f"  Accuracy: {acc2:.1f}%")

if __name__ == "__main__":
    main()
