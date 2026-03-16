import sys, os
from src.media_classifier import analyze_video
import glob

real_dir = '../ff_downloads/original_sequences/youtube/c40/videos/'
fake_dir = '../ff_downloads/manipulated_sequences/Deepfakes/c40/videos/'

real_vids = sorted(glob.glob(os.path.join(real_dir, '*.mp4')))[:5]
fake_vids = sorted(glob.glob(os.path.join(fake_dir, '*.mp4')))[:5]

print("=== REAL VIDEOS ===")
for v in real_vids:
    res = analyze_video(v)
    if not res: continue
    print(f"{os.path.basename(v)}: SNR={res['snr_db']:.2f}, Purity={res.get('spectral_purity',0):.2f}, Per={res.get('periodicity',0):.2f}, Prom={res.get('peak_prominence',0):.2f}, ROI={res.get('roi_correlation', 0):.2f}")

print("\n=== FAKE VIDEOS ===")
for v in fake_vids:
    res = analyze_video(v)
    if not res: continue
    print(f"{os.path.basename(v)}: SNR={res['snr_db']:.2f}, Purity={res.get('spectral_purity',0):.2f}, Per={res.get('periodicity',0):.2f}, Prom={res.get('peak_prominence',0):.2f}, ROI={res.get('roi_correlation', 0):.2f}")
