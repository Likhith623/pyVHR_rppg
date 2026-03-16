import sys, os
from src.media_classifier import analyze_video
import cv2

# We'll just run it on 033.mp4 to see what its outputs are
res = analyze_video('../ff_downloads/original_sequences/youtube/c40/videos/033.mp4')
print(res)
