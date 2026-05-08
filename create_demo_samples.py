"""
Download sample audio clips with real sounds (ESC-50 dataset, CC-BY-NC).
"""

import os, urllib.request

DEST = os.path.join(os.path.dirname(__file__), "sample_audio")
os.makedirs(DEST, exist_ok=True)

BASE = "https://github.com/karoldvl/ESC-50/raw/master/audio/"

SAMPLES = [
    ("car_horn.wav", "1-17124-A-43.wav", "Car horn"),
    ("siren.wav", "1-31482-A-42.wav", "Police siren"),
    ("dog_bark.wav", "1-100032-A-0.wav", "Dog barking"),
    ("glass_break.wav", "1-20133-A-39.wav", "Glass breaking"),
    ("gunshot.wav", "1-115545-A-48.wav", "Fireworks (detected as gunshot/explosion)"),
    ("engine.wav", "1-18527-A-44.wav", "Engine running"),
]

print("Downloading sample audio clips...")
for local, remote, desc in SAMPLES:
    path = os.path.join(DEST, local)
    if not os.path.exists(path):
        print(f"  {local} ({desc})")
        urllib.request.urlretrieve(BASE + remote, path)
print(f"\nDone. {len(SAMPLES)} files in '{DEST}'")
