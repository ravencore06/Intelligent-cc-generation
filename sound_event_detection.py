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
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes
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
        result = self._detect(audio_path)
        result.source_file = os.path.basename(audio_path)
        return result

    def detect_from_video(self, video_path: str) -> DetectionResult:
        """Extract audio from video, run detection, clean up temp file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
        try:
            self._extract_audio(video_path, tmp_wav)
            result = self._detect(tmp_wav)
            result.source_file = os.path.basename(video_path)
            return result
        finally:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)

    def close(self) -> None:
        if self._classifier is not None:
            self._classifier.close()
            self._classifier = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

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
        from moviepy.editor import VideoFileClip

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
