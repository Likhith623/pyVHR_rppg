import sys, os
from src.media_classifier import analyze_video
import glob

real_dir = '../ff_downloads/original_sequences/youtube/c40/videos/'
fake_dir = '../ff_downloads/manipulated_sequences/Deepfakes/c40/videos/'

real_vids = sorted(glob.glob(os.path.join(real_dir, '*.mp4')))[:3]
fake_vids = sorted(glob.glob(os.path.join(fake_dir, '*.mp4')))[:3]

print("File | Signal Strength | Prominence | Periodicity | SNR")
for v in real_vids:
    res = analyze_video(v)
    if not res: continue
    print(f"REAL {os.path.basename(v)}: SS={res.get('signal_strength',0):.5f}, Purity={res.get('spectral_purity',0):.3f}")

for v in fake_vids:
    res = analyze_video(v)
    if not res: continue
    print(f"FAKE {os.path.basename(v)}: SS={res.get('signal_strength',0):.5f}, Purity={res.get('spectral_purity',0):.3f}")
