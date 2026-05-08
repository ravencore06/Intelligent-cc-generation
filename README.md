# Module 1: Sound Event Detection

Detects environmental sounds (car horns, gunshots, dog barks, sirens, etc.)
from audio using YAMNet + MediaPipe. Outputs events with timestamps and
confidence scores.

## Quick Start

```bash
pip install -r requirements.txt
python create_demo_samples.py   # downloads sample audio clips
python demo_module1.py --all     # run detection on all samples
python demo_module1.py my_file.wav   # or your own audio/video
```

## Example Output

```
  car_horn.wav
  Duration: 5.0s
    [0.0s] Vehicle horn, car horn, honking     (80%) <<<
    => Detected: Vehicle horn, car horn, honking

  gunshot.wav
  Duration: 5.0s
    [1.9s] Explosion                           (89%) <<<
    [1.9s] Gunshot, gunfire                    (33%) <<<
    => Detected: Explosion, Gunshot, gunfire
```

## Files

| File | What it does |
|------|-------------|
| `sound_event_detection.py` | Core SED module (SoundEventDetector class) |
| `demo_module1.py` | CLI demo runner |
| `create_demo_samples.py` | Downloads sample audio clips (ESC-50, CC-BY-NC) |
| `yamnet.tflite` | YAMNet model (auto-downloaded) |

## How it Works

1. Load audio, run YAMNet on 0.975s windows
2. Filter by confidence (default >= 15%)
3. Merge adjacent same-class events
4. Output results (console + JSON)

## Limitations

- ESC-50 samples are CC-BY-NC (not for commercial use)
- YAMNet is limited to 521 AudioSet classes
- No video modality integration yet
