"""
CC Output Generators - Produces SRT and SLS caption files from a fused
UnifiedEvent timeline.

Formats:
  - SRT: Standard SubRip subtitle format with enriched captions combining
         audio, visual, and speech signals.
  - SLS: Styled caption format (SAMI-like) with CSS classes per event type,
         enabling color-coded display of audio events, visual events, and
         speech in media players that support SAMI.
"""

from typing import List, Optional
from cc_decision_engine import UnifiedEvent


# ---------------------------------------------------------------------------
# SRT output
# ---------------------------------------------------------------------------


def _fmt_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(events: List[UnifiedEvent], filepath: Optional[str] = None) -> str:
    lines: List[str] = []
    for idx, e in enumerate(events, start=1):
        lines.append(str(idx))
        lines.append(f"{_fmt_srt_time(e.start_time)} --> {_fmt_srt_time(e.end_time)}")
        lines.append(e.caption)
        lines.append("")
    text = "\n".join(lines)
    if filepath:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
    return text


# ---------------------------------------------------------------------------
# SLS (Styled Caption) output
# ---------------------------------------------------------------------------

_SLS_CSS_CLASSES = {
    "audio": "cc-audio",
    "visual_object": "cc-visual-object",
    "visual_action": "cc-visual-action",
    "speech": "cc-speech",
}

_SLS_HEADER = """<?xml version="1.0" encoding="utf-8"?>
<SLS>
<HEAD>
<TITLE>Intelligent Closed Captions</TITLE>
<STYLE TYPE="text/css">
<!--
  .cc-audio         { color: #FF6B6B; font-weight: bold; }
  .cc-visual-object { color: #4ECDC4; }
  .cc-visual-action { color: #FFE66D; font-weight: bold; }
  .cc-speech        { color: #FFFFFF; }
-->
</STYLE>
</HEAD>
<BODY>
"""

_SLS_FOOTER = """</BODY>
</SLS>
"""


def to_sls(events: List[UnifiedEvent], filepath: Optional[str] = None) -> str:
    lines: List[str] = [_SLS_HEADER]
    for e in events:
        css = _SLS_CSS_CLASSES.get(e.event_type, "cc-speech")
        start_ms = int(e.start_time * 1000)
        lines.append(
            f'  <SYNC Start={start_ms}>\n    <P CLASS="{css}">{_xml_escape(e.caption)}'
        )
    lines.append(_SLS_FOOTER)
    text = "\n".join(lines)
    if filepath:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
    return text


def _xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
