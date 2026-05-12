# Intelligent Closed Caption Generation

This project implements an intelligent, multi-modal closed captioning (CC) pipeline. It extracts non-verbal audio cues, visual actions, and objects, and fuses them alongside speech transcripts to create rich, context-aware subtitles.

## Project File Structure

```text
Intelligent-cc-generation/
├── .git/
├── .ruff_cache/
├── __pycache__/
├── sample_audio/
├── README.md                            # Project documentation
├── requirements.txt                     # Python dependencies
├── create_demo_samples.py               # Script to generate demo/sample data
├── demo_pipeline.py                     # Main execution script for the pipeline
├── demo_module1.py                      # CLI demo for Module 1 (Audio)
├── demo_module2.py                      # CLI demo for Module 2 (Visual)
├── cc_decision_engine.py                # Module 3: Logic for prioritizing/merging events
├── cc_output.py                         # Formatter for generating SRT/SLS/JSON
├── sound_event_detection.py             # Audio event classification logic
├── yamnet.tflite                        # TFLite model for audio event detection
├── efficientnet_lite0.tflite            # TFLite model for object detection
├── face_detector.tflite                 # TFLite model for face detection
├── pose_landmarker.task                 # MediaPipe task for action detection
└── imagenet_labels.txt                  # Labels for visual object classification
```

## Workflow Modules

### Module 1: Sound Event Detection
Detects environmental sounds (car horns, gunshots, dog barks, sirens, etc.) from audio using YAMNet and MediaPipe. Outputs sound events with precise timestamps and confidence scores.

*   **Files**: `sound_event_detection.py`, `demo_module1.py`
*   **Usage**: `python demo_module1.py my_file.wav`

### Module 2: Visual Detection
Analyzes video frames for visual context using MediaPipe and TensorFlow Lite models. It detects objects (e.g., cars, weapons), actions/poses (e.g., punching, falling), and facial expressions.

*   **Files**: `demo_module2.py`
*   **Models used**: `efficientnet_lite0.tflite`, `face_detector.tflite`, `pose_landmarker.task`

### Module 3: Fusion & Decision Engine
Fuses the outputs from Module 1 (audio events), Module 2 (visual events), and a speech transcript (SRT) into a unified timeline. The Decision Engine resolves conflicts, prioritizes overlapping events, and formats them into cohesive captions. Finally, it outputs standard subtitle files (SRT, SLS, JSON).

*   **Files**: `cc_decision_engine.py`, `cc_output.py`, `demo_pipeline.py`

## Quick Start (End-to-End Pipeline)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the end-to-end pipeline on a video
python demo_pipeline.py "Avengers vs Ultron.mp4"

# 3. To include an existing speech transcript (SRT):
python demo_pipeline.py "Avengers vs Ultron.mp4" --speech transcript.srt

# 4. To reuse previously generated intermediate JSON files:
python demo_pipeline.py "Avengers vs Ultron.mp4" --reuse
```

## Example Output (Sample Captions)

```
[0.0s-1.0s] [audio] [Music] (41%); [Speech] (80%)
[0.5s-4.3s] [visual_action] ⚡ Fall Down
[2.9s-3.9s] [visual_action] ⚡ Punching
[4.8s-5.8s] [audio] [Music] (26%); [Speech] (80%)
[6.8s-7.8s] [audio] [Smash, crash] (20%)
```

## Limitations

- ESC-50 samples downloaded via `create_demo_samples.py` are CC-BY-NC.
- YAMNet is limited to the 521 AudioSet classes.
