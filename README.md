# Intelligent CC Generation

An automated Closed Caption (CC) generation pipeline leveraging OpenAI's Whisper model.

## Overview

This project provides a streamlined pipeline to extract audio from video files and generate precise, timestamped SubRip (`.srt`) subtitles using state-of-the-art automatic speech recognition (ASR). It includes zero-configuration FFmpeg integration for seamless cross-platform execution.

## Tech Stack

- **Python 3.10+**
- **OpenAI Whisper:** Core ASR engine
- **MoviePy:** Media processing
- **Imageio-FFmpeg:** Bundled FFmpeg binaries

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Intelligent-cc-generation.git
   cd Intelligent-cc-generation
   ```

2. **Install dependencies:**
   We recommend isolating dependencies using a virtual environment (`venv`).
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Place your source video in the project directory and execute the main pipeline:

```bash
python main.py <video_filename.mp4>
```

**Options:**
- `--model <size>`: Specify the Whisper model size (`tiny`, `base`, `small`, `medium`, `large`). Defaults to `tiny`.
  ```bash
  python main.py video.mp4 --model base
  ```

*Generated subtitles are automatically saved to the `output/` directory.*

## Project Structure

```text
Intelligent-cc-generation/
├── main.py                     # CLI orchestration and pipeline entry point
├── requirements.txt            # Python package dependencies
└── src/                        # Core application modules
    ├── audio_extractor.py      # Audio extraction logic (MoviePy)
    ├── speech_to_text.py       # Whisper ASR integration & local FFmpeg handling
    └── subtitle_generator.py   # Subtitle formatting and generation (.srt)
```



## Pipeline Modules Overview

According to the project specifications, the Intelligent CC tool consists of three main modules:
- **Module 1 – Sound Event Detection** (with confidence scores & timestamps)
- **Module 2 – Speaker/Scene Reaction Detection** (Frame extraction + visual analysis)
- **Module 3 – CC Decision Engine & SRT/SLS Output** (Combined audio + visual signals -> CC file generation)

### Chosen Demo: Module 3 (CC Decision Engine & SRT Output)

For this Pull Request, we have focused on demonstrating a working implementation of **Module 3**. 

**Our Approach:**
We built an automated pipeline leveraging **OpenAI's Whisper** model to extract audio from the video and generate precise, timestamped SubRip (`.srt`) files. Whisper serves as our core ASR (Automatic Speech Recognition) engine due to its state-of-the-art accuracy and robust handling of background noise. The pipeline is designed modularly, ensuring that future outputs from Module 1 and Module 2 can be seamlessly integrated into the final subtitle generation logic.

## Known Limitations & Next Steps

**Current Limitations:**
- **Missing Visual/Contextual Signals:** Currently, our Module 3 implementation relies solely on the audio track for transcription. It does not yet incorporate the visual signals (Module 2) or the distinct sound event tags (Module 1).
- **Processing Time:** Processing long videos can be computationally expensive depending on the chosen Whisper model size (`tiny`, `base`, `small`, etc.) and the available hardware.

**Areas to Improve Next (Integration):**
- **Integrate Module 1 (Sound Events):** Merge Sound Event Detection outputs into the `.srt` generation to include closed-captioning for important non-speech sounds (e.g., `[door slams]`, `[dog barks]`).
- **Integrate Module 2 (Visual Context):** Use visual signals (e.g., Speaker/Scene Reaction Detection) to help the decision engine differentiate between multiple speakers and provide contextual cues when someone is speaking off-camera.
- **Performance Optimization:** Explore faster Whisper implementations (like `faster-whisper` or `whisper.cpp`) to reduce processing time for larger media files.

## License

Distributed under the MIT License. See `LICENSE` for details.
