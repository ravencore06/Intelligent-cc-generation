"""
Sound Event Detection (SED) Module - Module 1 of the Intelligent CC pipeline.

Detects environmental sounds (car horns, gunshots, dog barks, sirens, etc.)
from audio using the YAMNet model via MediaPipe Audio Classifier.

Outputs detected events with timestamps and confidence scores.
"""

import os
import json
import urllib.request
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.io import wavfile

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python import audio

YAMNET_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "audio_classifier/yamnet/float32/1/yamnet.tflite"
)
YAMNET_WINDOW_SEC = 0.975

DEFAULT_TARGET_SOUNDS = [
    "Vehicle horn, car horn, honking",
    "Gunshot, gunfire",
    "Dog",
    "Bark",
    "Siren",
    "Alarm",
    "Explosion",
    "Fireworks",
    "Glass",
    "Breaking",
    "Vehicle",
    "Engine",
]


@dataclass
class SoundEvent:
    """A single detected sound event with time bounds and confidence."""

    start_time: float
    end_time: float
    sound_class: str
    confidence: float
    class_index: int


@dataclass
class DetectionResult:
    """Collection of detected events for one audio file."""

    events: List[SoundEvent] = field(default_factory=list)
    duration_seconds: float = 0.0
    source_file: str = ""

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "duration_seconds": self.duration_seconds,
            "events": [
                {
                    "start_time": round(e.start_time, 3),
                    "end_time": round(e.end_time, 3),
                    "sound_class": e.sound_class,
                    "confidence": round(e.confidence, 3),
                    "class_index": e.class_index,
                }
                for e in self.events
            ],
        }

    def to_json(self, filepath: Optional[str] = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent)
        if filepath:
            with open(filepath, "w") as f:
                f.write(text)
        return text

    def to_srt(
        self, filepath: Optional[str] = None, min_confidence: float = 0.0
    ) -> str:
        lines = []
        counter = 1
        for e in self.events:
            if e.confidence < min_confidence:
                continue
            lines.append(str(counter))
            lines.append(f"{_fmt_srt(e.start_time)} --> {_fmt_srt(e.end_time)}")
            lines.append(f"[{e.sound_class}] ({e.confidence:.0%})")
            lines.append("")
            counter += 1
        text = "\n".join(lines)
        if filepath:
            with open(filepath, "w") as f:
                f.write(text)
        return text


class SoundEventDetector:
    """Sound Event Detection using MediaPipe's YAMNet classifier.

    Args:
        model_path: Path to YAMNet TFLite model.
        confidence_threshold: Minimum confidence (0-1) to report an event.
        target_classes: List of sound class names to filter for.
                        If None, all classes above threshold are reported.
    """

    def __init__(
        self,
        model_path: str = "yamnet.tflite",
        confidence_threshold: float = 0.20,
        target_classes: Optional[List[str]] = None,
        use_separation: bool = False,
        use_noise_reduction: bool = False,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes
        self.use_separation = use_separation
        self.use_noise_reduction = use_noise_reduction
        self._classifier: Optional[audio.AudioClassifier] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_from_file(self, audio_path: str) -> DetectionResult:
        """Run detection on a WAV audio file."""
        if not audio_path.lower().endswith(".wav"):
            raise ValueError(
                "Input must be a .wav file. Use detect_from_video() for video files."
            )
        
        import shutil
        temp_items: list[str] = []
        try:
            current_path = audio_path

            if self.use_separation:
                inst_path, temp_dir = self._separate_audio(current_path)
                temp_items.append(temp_dir)
                current_path = inst_path

            if self.use_noise_reduction:
                denoised_path = self._reduce_noise(current_path)
                temp_items.append(denoised_path)
                current_path = denoised_path

            result = self._detect(current_path)
            result.source_file = os.path.basename(audio_path)
            return result
        finally:
            for item in reversed(temp_items):
                if os.path.isdir(item):
                    shutil.rmtree(item, ignore_errors=True)
                elif os.path.isfile(item):
                    try:
                        os.remove(item)
                    except OSError:
                        pass

    def detect_from_video(self, video_path: str) -> DetectionResult:
        """Extract audio from video, run detection, clean up temp file."""
        import shutil
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
        
        temp_items: list[str] = [tmp_wav]
        try:
            self._extract_audio(video_path, tmp_wav)
            current_path = tmp_wav

            if self.use_separation:
                inst_path, temp_dir = self._separate_audio(current_path)
                temp_items.append(temp_dir)
                current_path = inst_path

            if self.use_noise_reduction:
                denoised_path = self._reduce_noise(current_path)
                temp_items.append(denoised_path)
                current_path = denoised_path

            result = self._detect(current_path)
            result.source_file = os.path.basename(video_path)
            return result
        finally:
            for item in reversed(temp_items):
                if os.path.isdir(item):
                    shutil.rmtree(item, ignore_errors=True)
                elif os.path.isfile(item):
                    try:
                        os.remove(item)
                    except OSError:
                        pass

    def close(self) -> None:
        if self._classifier is not None:
            self._classifier.close()
            self._classifier = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @staticmethod
    def _ensure_ffmpeg() -> None:
        """Dynamically find and setup ffmpeg from imageio_ffmpeg if not in PATH."""
        import shutil
        if shutil.which("ffmpeg") is not None:
            return
            
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            return
            
        if not ffmpeg_exe or not os.path.exists(ffmpeg_exe):
            return
            
        import tempfile
        temp_dir = os.path.join(tempfile.gettempdir(), "intelligent_cc_ffmpeg")
        os.makedirs(temp_dir, exist_ok=True)
        
        dest_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
        ffmpeg_dest = os.path.join(temp_dir, dest_name)
        
        if not os.path.exists(ffmpeg_dest):
            try:
                shutil.copy(ffmpeg_exe, ffmpeg_dest)
            except Exception:
                pass
                
        if os.path.exists(ffmpeg_dest):
            os.environ["PATH"] = temp_dir + os.pathsep + os.environ["PATH"]

    def _separate_audio(self, wav_path: str) -> tuple[str, str]:
        """Separate vocals and instrumental/background stems.
        Returns a tuple of (instrumental_path, temp_dir_to_clean_up)
        """
        import tempfile
        from audio_separator.separator import Separator

        self._ensure_ffmpeg()

        # Create a temp directory for outputs
        temp_dir = tempfile.mkdtemp(prefix="cc_separation_")
        
        # Initialize separator
        separator = Separator(output_dir=temp_dir)
        separator.load_model(model_filename='UVR-MDX-NET-Inst_HQ_3.onnx')
        
        print(f"      Separating vocals/speech from background audio...")
        outputs = separator.separate(wav_path)
        
        # outputs contains filenames. Find the instrumental stem
        inst_file = None
        for filename in outputs:
            if "Instrumental" in filename:
                inst_file = filename
                break
        
        if not inst_file:
            inst_file = outputs[0] if outputs else None
            
        if not inst_file:
            raise RuntimeError("Audio stem separation failed, no output files generated.")
            
        inst_path = os.path.join(temp_dir, inst_file)
        return inst_path, temp_dir

    def _reduce_noise(self, wav_path: str) -> str:
        """Apply spectral-gate noise reduction. Returns path to denoised WAV."""
        import tempfile
        import noisereduce as nr

        sr, data = wavfile.read(wav_path)

        if data.dtype == np.int16:
            float_data = data.astype(np.float32) / np.iinfo(np.int16).max
        elif data.dtype == np.int32:
            float_data = data.astype(np.float32) / np.iinfo(np.int32).max
        elif data.dtype == np.uint8:
            float_data = data.astype(np.float32) / 255.0 * 2.0 - 1.0
        else:
            float_data = data.astype(np.float32)

        if float_data.ndim > 1:
            reduced = np.stack([
                nr.reduce_noise(y=float_data[:, c], sr=sr, prop_decrease=0.8)
                for c in range(float_data.shape[1])
            ], axis=1)
        else:
            reduced = nr.reduce_noise(y=float_data, sr=sr, prop_decrease=0.8)

        reduced = np.clip(reduced, -1.0, 1.0)
        reduced_int16 = (reduced * np.iinfo(np.int16).max).astype(np.int16)

        fd, out_path = tempfile.mkstemp(suffix="_denoised.wav")
        os.close(fd)
        wavfile.write(out_path, sr, reduced_int16)
        return out_path

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _get_classifier(self) -> audio.AudioClassifier:
        if self._classifier is not None:
            return self._classifier
        if not os.path.exists(self.model_path):
            print(f"Downloading YAMNet model to {self.model_path} ...")
            urllib.request.urlretrieve(YAMNET_MODEL_URL, self.model_path)
        base_opts = python.BaseOptions(model_asset_path=self.model_path)
        opts = audio.AudioClassifierOptions(
            base_options=base_opts,
            running_mode=audio.RunningMode.AUDIO_CLIPS,
            max_results=5,
        )
        self._classifier = audio.AudioClassifier.create_from_options(opts)
        return self._classifier

    @staticmethod
    def _extract_audio(video_path: str, output_wav: str) -> None:
        from moviepy import VideoFileClip

        with VideoFileClip(video_path) as clip:
            if clip.audio is None:
                raise ValueError(f"No audio track in {video_path}")
            clip.audio.write_audiofile(
                output_wav,
                fps=16000,
                nbytes=2,
                codec="pcm_s16le",
                ffmpeg_params=["-ac", "1"],
                logger=None,
            )

    @staticmethod
    def _load_wav(path: str):
        sr, data = wavfile.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / np.iinfo(np.int16).max
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / np.iinfo(np.int32).max
        elif data.dtype == np.uint8:
            data = data.astype(np.float32) / 255.0 * 2.0 - 1.0
        return data, sr

    def _detect(self, wav_path: str) -> DetectionResult:
        classifier = self._get_classifier()
        waveform, sr = self._load_wav(wav_path)
        duration = len(waveform) / sr

        audio_clip = containers.AudioData.create_from_array(waveform, sr)
        raw_results = classifier.classify(audio_clip)

        raw_events: List[SoundEvent] = []
        for idx, frame_result in enumerate(raw_results):
            ts_ms = getattr(
                frame_result,
                "timestamp_ms",
                idx * YAMNET_WINDOW_SEC * 1000,
            )
            start = ts_ms / 1000.0

            for classification in frame_result.classifications:
                for cat in classification.categories:
                    if cat.score < self.confidence_threshold:
                        continue
                    if (
                        self.target_classes
                        and cat.category_name not in self.target_classes
                    ):
                        continue
                    raw_events.append(
                        SoundEvent(
                            start_time=start,
                            end_time=start + YAMNET_WINDOW_SEC,
                            sound_class=cat.category_name,
                            confidence=cat.score,
                            class_index=getattr(cat, "index", -1),
                        )
                    )

        merged = self._merge_events(raw_events)
        return DetectionResult(events=merged, duration_seconds=duration)

    @staticmethod
    def _merge_events(
        events: List[SoundEvent], max_gap: float = 0.5
    ) -> List[SoundEvent]:
        if not events:
            return []
        events = sorted(events, key=lambda e: (e.start_time, e.sound_class))
        merged: List[SoundEvent] = [events[0]]
        for e in events[1:]:
            prev = merged[-1]
            gap = e.start_time - prev.end_time
            if e.sound_class == prev.sound_class and gap <= max_gap:
                prev.end_time = max(prev.end_time, e.end_time)
                prev.confidence = max(prev.confidence, e.confidence)
            else:
                merged.append(e)
        return merged


def _fmt_srt(seconds: float) -> str:
    """Format seconds to HH:MM:SS,mmm for SRT."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
