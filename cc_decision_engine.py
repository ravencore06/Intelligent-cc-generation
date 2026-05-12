"""
CC Decision Engine - Fuses audio events, visual events, and speech transcripts
into a unified closed caption timeline with intelligent prioritization.

Inputs (all optional):
  - Audio events: sound_event_detection.DetectionResult.to_dict() or JSON file
  - Visual events: demo_module2.analyze() output dict or JSON file
  - Speech transcript: path to existing SRT file or SRT string

Algorithm:
  1. Parse all inputs into UnifiedEvent objects with priority levels
  2. Remove non-caption events (scene changes used as metadata only)
  3. Sort by start_time, then priority (ascending)
  4. Resolve overlaps: higher priority preempts lower priority
  5. Merge adjacent same-priority events
  6. Apply duration constraints (min 1.0s, max 6.0s)
  7. Output the unified timeline
"""

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import IntEnum


# ---------------------------------------------------------------------------
# Priority levels (lower number = higher priority)
# ---------------------------------------------------------------------------


class Priority(IntEnum):
    CRITICAL = 1  # gunshots, explosions, falls
    HIGH = 2  # horns, punches, sirens, alarms
    MEDIUM = 3  # vehicles, engines, animals
    LOW = 4  # scene changes (metadata only)
    SPEECH = 5  # speech transcription


_MAX_CAPTION_DURATION = 6.0
_MIN_CAPTION_DURATION = 1.0
_COOLDOWN = 0.3
_SCENE_PRIORITY = int(Priority.LOW)
_DEFAULT_PRIORITY = int(Priority.MEDIUM)

# Sound class keywords -> priority
_SOUND_PRIORITY: Dict[str, int] = {
    "gunshot": int(Priority.CRITICAL),
    "gunfire": int(Priority.CRITICAL),
    "explosion": int(Priority.CRITICAL),
    "glass": int(Priority.CRITICAL),
    "breaking": int(Priority.CRITICAL),
    "siren": int(Priority.HIGH),
    "alarm": int(Priority.HIGH),
    "horn": int(Priority.HIGH),
    "honking": int(Priority.HIGH),
    "vehicle": int(Priority.MEDIUM),
    "engine": int(Priority.MEDIUM),
    "dog": int(Priority.MEDIUM),
    "bark": int(Priority.MEDIUM),
    "cat": int(Priority.MEDIUM),
    "person": int(Priority.MEDIUM),
    "speech": int(Priority.SPEECH),
    "music": int(Priority.SPEECH),
}

_ACTION_PRIORITY: Dict[str, int] = {
    "fall_down": int(Priority.CRITICAL),
    "punching": int(Priority.HIGH),
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class UnifiedEvent:
    """A single caption event in the unified timeline."""

    start_time: float
    end_time: float
    caption: str
    event_type: str  # audio / visual_object / visual_action / speech
    source_label: str
    confidence: float
    priority: int

    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        return {
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "caption": self.caption,
            "event_type": self.event_type,
            "source_label": self.source_label,
            "confidence": round(self.confidence, 3),
            "priority": self.priority,
        }


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def _priority_for_sound(sound_class: str) -> int:
    cls = sound_class.lower()
    for kw, pri in _SOUND_PRIORITY.items():
        if kw in cls:
            return pri
    return _DEFAULT_PRIORITY


def _priority_for_action(action: str) -> int:
    return _ACTION_PRIORITY.get(action, int(Priority.HIGH))


def _fmt_caption(event_type: str, label: str, detail: str, confidence: float) -> str:
    if event_type == "speech":
        return detail
    if event_type == "visual_action":
        return f"\u26a1 {detail}"
    if event_type == "visual_object":
        return f"\U0001f441 {detail}"
    conf_str = f"({confidence:.0%})" if event_type == "audio" else ""
    return f"[{label}] {conf_str}".strip()


def parse_audio_events(audio_data: Dict[str, Any]) -> List[UnifiedEvent]:
    events = []
    for e in audio_data.get("events", []):
        sc = e.get("sound_class", "")
        pri = _priority_for_sound(sc)
        caption = _fmt_caption("audio", sc, sc, e.get("confidence", 0))
        events.append(
            UnifiedEvent(
                start_time=e["start_time"],
                end_time=e["end_time"],
                caption=caption,
                event_type="audio",
                source_label=sc,
                confidence=e.get("confidence", 0.0),
                priority=pri,
            )
        )
    return events


def parse_audio_file(path: str) -> List[UnifiedEvent]:
    with open(path) as f:
        return parse_audio_events(json.load(f))


def parse_visual_events(visual_data: Dict[str, Any]) -> List[UnifiedEvent]:
    events = []
    for e in visual_data.get("events", []):
        t = e.get("type", "")
        time = e.get("time", 0.0)

        if t == "object":
            obj = e.get("object", "unknown")
            label = e.get("label", "")
            pri = _SOUND_PRIORITY.get(obj, int(Priority.MEDIUM))
            detail = f"{obj} ({label})" if label else obj
            events.append(
                UnifiedEvent(
                    start_time=time,
                    end_time=time + 1.5,
                    caption=_fmt_caption(
                        "visual_object", obj, detail, e.get("confidence", 0)
                    ),
                    event_type="visual_object",
                    source_label=obj,
                    confidence=e.get("confidence", 0.0),
                    priority=pri,
                )
            )

        elif t == "action":
            action = e.get("action", "")
            pri = _priority_for_action(action)
            detail = action.replace("_", " ").title()
            events.append(
                UnifiedEvent(
                    start_time=time,
                    end_time=time + 2.0,
                    caption=_fmt_caption("visual_action", action, detail, 1.0),
                    event_type="visual_action",
                    source_label=action,
                    confidence=1.0,
                    priority=pri,
                )
            )

        elif t == "scene_change":
            events.append(
                UnifiedEvent(
                    start_time=time,
                    end_time=time + 0.5,
                    caption="",
                    event_type="scene_change",
                    source_label="scene_change",
                    confidence=0.0,
                    priority=_SCENE_PRIORITY,
                )
            )

    return events


def parse_visual_file(path: str) -> List[UnifiedEvent]:
    with open(path) as f:
        return parse_visual_events(json.load(f))


_SRT_TIME_RE = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")


def _parse_srt_time(s: str) -> float:
    m = _SRT_TIME_RE.match(s.strip())
    if not m:
        return 0.0
    h, mi, sec, ms = int(m[1]), int(m[2]), int(m[3]), int(m[4])
    return h * 3600 + mi * 60 + sec + ms / 1000.0


def parse_speech_srt(srt_text: str) -> List[UnifiedEvent]:
    if not srt_text or not srt_text.strip():
        return []
    events = []
    blocks = re.split(r"\n\s*\n", srt_text.strip())
    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if len(lines) < 2:
            continue
        time_line = None
        text_lines = []
        for line in lines:
            if "-->" in line:
                time_line = line
            elif not line.isdigit():
                text_lines.append(line)
        if not time_line or not text_lines:
            continue
        parts = time_line.split("-->")
        start = _parse_srt_time(parts[0])
        end = _parse_srt_time(parts[1])
        text = " ".join(text_lines)
        events.append(
            UnifiedEvent(
                start_time=start,
                end_time=end,
                caption=text,
                event_type="speech",
                source_label="speech",
                confidence=1.0,
                priority=int(Priority.SPEECH),
            )
        )
    return events


def parse_speech_file(path: str) -> List[UnifiedEvent]:
    with open(path, encoding="utf-8") as f:
        return parse_speech_srt(f.read())


# ---------------------------------------------------------------------------
# Fusion logic
# ---------------------------------------------------------------------------


def _overlaps(a: UnifiedEvent, b: UnifiedEvent) -> bool:
    return a.start_time < b.end_time and b.start_time < a.end_time


def _merge_events(events: List[UnifiedEvent]) -> List[UnifiedEvent]:
    """Merge adjacent events that have the same caption within a small gap."""
    if not events:
        return []
    merged = [events[0]]
    for e in events[1:]:
        prev = merged[-1]
        gap = e.start_time - prev.end_time
        if (
            gap <= _COOLDOWN
            and e.caption == prev.caption
            and e.priority == prev.priority
        ):
            prev.end_time = max(prev.end_time, e.end_time)
        else:
            merged.append(e)
    return merged


def fuse(
    audio_events: Optional[List[UnifiedEvent]] = None,
    visual_events: Optional[List[UnifiedEvent]] = None,
    speech_events: Optional[List[UnifiedEvent]] = None,
) -> List[UnifiedEvent]:
    """Fuse all event sources into a single prioritized caption timeline.

    Algorithm:
      1. Collect all events, marking scene changes as metadata
      2. Sort by start_time then priority (lower number = higher priority)
      3. Walk through events; when higher-priority event overlaps a
         lower-priority one, truncate/preempt the lower-priority event
      4. Clamp durations to [MIN_CAPTION_DURATION, MAX_CAPTION_DURATION]
      5. Merge adjacent same-caption events
    """
    all_events: List[UnifiedEvent] = []
    if audio_events:
        all_events.extend(audio_events)
    if visual_events:
        all_events.extend(visual_events)
    if speech_events:
        all_events.extend(speech_events)

    if not all_events:
        return []

    # Sort: time ascending, then priority ascending (higher priority first)
    all_events.sort(key=lambda e: (e.start_time, e.priority))

    # --- conflict resolution pass ---
    resolved: List[UnifiedEvent] = []
    for e in all_events:
        if e.event_type == "scene_change":
            continue  # scene changes are metadata, not captions
        if not e.caption:
            continue

        # Check overlaps with already-resolved events
        inserted = False
        for i, existing in enumerate(resolved):
            if not _overlaps(e, existing):
                continue
            # Same source label: extend end_time
            if (
                e.source_label == existing.source_label
                and e.caption == existing.caption
            ):
                existing.end_time = max(existing.end_time, e.end_time)
                inserted = True
                break
            # Different: higher priority wins
            if e.priority < existing.priority:
                # New event is higher priority — truncate existing
                if e.start_time <= existing.start_time:
                    # New event completely covers the old
                    resolved[i] = e
                else:
                    # Truncate existing, insert new
                    existing.end_time = e.start_time
                    resolved.insert(i + 1, e)
                inserted = True
                break
            elif e.priority > existing.priority:
                # Existing event is higher priority — skip or truncate new
                if existing.end_time >= e.end_time:
                    inserted = True  # fully covered, discard
                    break
                else:
                    e.start_time = existing.end_time
                    # fall through to try next or append
            else:
                # Same priority: merge captions if overlapping
                existing.end_time = max(existing.end_time, e.end_time)
                if e.caption not in existing.caption:
                    existing.caption = f"{existing.caption}; {e.caption}"
                existing.confidence = max(existing.confidence, e.confidence)
                inserted = True
                break

        if not inserted:
            resolved.append(e)

    # --- clamp durations ---
    for e in resolved:
        if e.duration() > _MAX_CAPTION_DURATION:
            e.end_time = e.start_time + _MAX_CAPTION_DURATION
        if e.duration() < _MIN_CAPTION_DURATION and e.event_type != "speech":
            e.end_time = e.start_time + _MIN_CAPTION_DURATION

    # --- merge adjacent same-caption events ---
    resolved = _merge_events(resolved)

    return resolved


# ---------------------------------------------------------------------------
# Convenience: load everything from files and fuse
# ---------------------------------------------------------------------------


def fuse_from_files(
    audio_json: Optional[str] = None,
    visual_json: Optional[str] = None,
    speech_srt: Optional[str] = None,
) -> List[UnifiedEvent]:
    audio = parse_audio_file(audio_json) if audio_json else None
    visual = parse_visual_file(visual_json) if visual_json else None
    speech = parse_speech_file(speech_srt) if speech_srt else None
    return fuse(audio, visual, speech)


def fused_to_dicts(events: List[UnifiedEvent]) -> List[dict]:
    return [e.to_dict() for e in events]
