import sys
import os
import json
import argparse
from typing import List

from cc_decision_engine import (
    fuse,
    fuse_from_files,
    parse_audio_events,
    parse_visual_events,
    parse_speech_srt,
    UnifiedEvent,
    fused_to_dicts,
)
from cc_output import to_srt, to_sls


def run_pipeline(
    video_path: str,
    speech_srt_path: str = None,
    reuse: bool = False,
    separate: bool = False,
):
    base = os.path.splitext(video_path)[0]

    # -- Step 1: Sound event detection (Module 1) --
    audio_json_path = base + "_events.json"
    if reuse and os.path.exists(audio_json_path):
        print(f"[1/3] Reusing audio events from {audio_json_path}")
        audio_events = None  # loaded via fuse_from_files
    else:
        print("[1/3] Running Sound Event Detection (Module 1)...")
        from sound_event_detection import SoundEventDetector

        detector = SoundEventDetector(confidence_threshold=0.15, use_separation=separate)
        result = detector.detect_from_video(video_path)
        result.to_json(audio_json_path)
        audio_events = parse_audio_events(result.to_dict())
        detector.close()
        print(f"      Found {len(audio_events)} audio events")

    # -- Step 2: Visual detection (Module 2) --
    visual_json_path = base + "_visual.json"
    if reuse and os.path.exists(visual_json_path):
        print(f"[2/3] Reusing visual events from {visual_json_path}")
        visual_events = None
    else:
        print("[2/3] Running Visual Detection (Module 2)...")
        from demo_module2 import analyze

        visual_result = analyze(video_path, interval=0.5)
        visual_events = parse_visual_events(visual_result)
        print(f"      Found {len(visual_events)} visual events")

    # Load speech
    speech_events = None
    if speech_srt_path:
        if os.path.exists(speech_srt_path):
            print(f"[2b/3] Loading speech transcript from {speech_srt_path}")
            speech_events = parse_speech_srt(
                open(speech_srt_path, encoding="utf-8").read()
            )
            print(f"      Found {len(speech_events)} speech segments")
        else:
            print(f"  [!] Speech file not found: {speech_srt_path}")

    # -- Step 3: Fusion --
    print("[3/3] Fusing signals into unified timeline...")
    if reuse:
        fused = fuse_from_files(
            audio_json=audio_json_path if audio_events is None else None,
            visual_json=visual_json_path if visual_events is None else None,
            speech_srt=speech_srt_path,
        )
    else:
        if audio_events is None:
            from cc_decision_engine import parse_audio_file

            audio_events = parse_audio_file(audio_json_path)
        if visual_events is None:
            from cc_decision_engine import parse_visual_file

            visual_events = parse_visual_file(visual_json_path)
        fused = fuse(audio_events, visual_events, speech_events)

    print(f"      Unified timeline: {len(fused)} caption events")

    # -- Step 4: Output --
    srt_path = base + "_cc.srt"
    sls_path = base + "_cc.sls"
    json_path = base + "_cc_fused.json"

    to_srt(fused, srt_path)
    print(f"      SRT -> {srt_path}")

    to_sls(fused, sls_path)
    print(f"      SLS -> {sls_path}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(fused_to_dicts(fused), f, indent=2)
    print(f"      JSON -> {json_path}")

    # -- Summary --
    _print_summary(fused, video_path)
    return fused


def _print_summary(events: List[UnifiedEvent], video_path: str):
    counts: dict = {}
    for e in events:
        counts[e.event_type] = counts.get(e.event_type, 0) + 1

    print(f"\n{'=' * 50}")
    print(f"Pipeline complete: {os.path.basename(video_path)}")
    print(f"{'=' * 50}")
    for etype, cnt in sorted(counts.items()):
        print(f"  {etype:20s}: {cnt}")
    print(f"{'=' * 50}")
    print("\nSample captions (first 10):")
    for e in events[:10]:
        try:
            sys.stdout.write(
                f"  [{e.start_time:.1f}s-{e.end_time:.1f}s] [{e.event_type}] {e.caption}\n"
            )
        except UnicodeEncodeError:
            safe = e.caption.encode("ascii", errors="replace").decode("ascii")
            print(f"  [{e.start_time:.1f}s-{e.end_time:.1f}s] [{e.event_type}] {safe}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intelligent CC Generation Pipeline")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--speech", help="Path to SRT speech transcript", default=None)
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse existing _events.json and _visual.json instead of re-running detection",
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Separate vocals/dialogue from background audio before running classification",
    )
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"File not found: {args.video}")
        sys.exit(1)

    run_pipeline(args.video, speech_srt_path=args.speech, reuse=args.reuse, separate=args.separate)
