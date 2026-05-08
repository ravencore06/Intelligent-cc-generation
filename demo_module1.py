#!/usr/bin/env python3
"""
Sound Event Detection Demo (Module 1).

Usage:
  python demo_module1.py sample_audio/car_horn.wav
  python demo_module1.py --all
  python demo_module1.py my_video.mp4
"""

import sys, os, json

sys.path.insert(0, os.path.dirname(__file__))
from sound_event_detection import SoundEventDetector

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_audio")
THRESHOLD = 0.15

ALL_SAMPLES = [
    ("car_horn.wav", ["Vehicle horn, car horn, honking"]),
    ("siren.wav", ["Siren", "Alarm"]),
    ("dog_bark.wav", ["Dog", "Bark"]),
    ("glass_break.wav", ["Breaking"]),
    ("gunshot.wav", ["Gunshot, gunfire", "Explosion", "Fireworks"]),
    ("engine.wav", ["Vehicle", "Engine"]),
]


def detect(path, threshold=THRESHOLD):
    detector = SoundEventDetector(confidence_threshold=threshold)
    if path.lower().endswith(".wav"):
        result = detector.detect_from_file(path)
    else:
        result = detector.detect_from_video(path)
    detector.close()
    return result


def run_all():
    print("=" * 55)
    print("  Sound Event Detection Demo (Module 1)")
    print("=" * 55)

    if not os.path.isdir(SAMPLE_DIR):
        print("Run 'python create_demo_samples.py' first.")
        return

    for fname, expected in ALL_SAMPLES:
        path = os.path.join(SAMPLE_DIR, fname)
        if not os.path.exists(path):
            continue

        result = detect(path)
        detected = {e.sound_class for e in result.events}
        matched = detected & set(expected)

        print(f"\n  {fname}")
        print(f"  Duration: {result.duration_seconds:.1f}s")
        if result.events:
            for e in result.events:
                icon = " <<<" if e.sound_class in expected else ""
                print(
                    f"    [{e.start_time:.1f}s] {e.sound_class:35s} "
                    f"({e.confidence:.0%}){icon}"
                )
        else:
            print("    (no events above threshold)")
        if matched:
            print(f"    => Detected: {', '.join(sorted(matched))}")
        else:
            print(f"    => Expected: {', '.join(expected)}")

    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        return

    if sys.argv[1] == "--all":
        run_all()
        return

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    result = detect(path)
    print(f"\n  File: {os.path.basename(path)}")
    print(f"  Duration: {result.duration_seconds:.1f}s")
    print(f"  Events:")
    for e in result.events:
        print(
            f"    [{e.start_time:.1f}s - {e.end_time:.1f}s]  "
            f"{e.sound_class:35s}  ({e.confidence:.0%})"
        )

    out = os.path.splitext(path)[0] + "_events.json"
    result.to_json(out)
    print(f"\n  Results saved to: {out}")


if __name__ == "__main__":
    main()
