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



## License

Distributed under the MIT License. See `LICENSE` for details.
