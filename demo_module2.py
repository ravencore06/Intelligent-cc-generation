import sys, os, json, urllib.request

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

CLASSIFIER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "image_classifier/efficientnet_lite0/float32/1/"
    "efficientnet_lite0.tflite"
)
LABELS_URL = "https://raw.githubusercontent.com/google-coral/test_data/master/imagenet_labels.txt"
MODEL_PATH = "efficientnet_lite0.tflite"
LABELS_PATH = "imagenet_labels.txt"

TARGETS = {
    "helicopter": list(range(650, 670)),
    "airplane": [404, 895],
    "vehicle": [654, 468, 511, 627, 661, 581, 609, 864, 817, 656],
    "person": [708],
    "dog": list(range(151, 300)),
    "cat": list(range(281, 300)),
    "motorcycle": [661, 670],
    "bicycle": [444, 671],
}


def _download(url, path):
    if not os.path.exists(path):
        print(f"  Downloading {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path)


def _detect_actions(landmarks, h, w):
    actions = []
    if not landmarks:
        return actions

    def y(lm):
        return (1 - lm.y) * h

    nose_y = y(landmarks[0])
    hip_y = (y(landmarks[23]) + y(landmarks[24])) / 2
    ankle_y = (y(landmarks[27]) + y(landmarks[28])) / 2
    height = abs(nose_y - ankle_y)
    width = abs(nose_y - hip_y)
    if height > 30 and width < height * 0.4:
        actions.append("fall_down")
    mx = (landmarks[11].x + landmarks[12].x) / 2 * w
    wrist_dist = abs(landmarks[15].x * w - mx) + abs(landmarks[16].x * w - mx)
    if wrist_dist > w * 0.7:
        actions.append("punching")
    return actions


def analyze(video_path, interval=0.5):
    _download(CLASSIFIER_URL, MODEL_PATH)
    _download(LABELS_URL, LABELS_PATH)

    with open(LABELS_PATH) as f:
        all_labels = [l.strip() for l in f.readlines()]

    # MediaPipe Image Classifier
    base_opts = python.BaseOptions(model_asset_path=MODEL_PATH)
    img_opts = vision.ImageClassifierOptions(base_options=base_opts, max_results=5)
    classifier = vision.ImageClassifier.create_from_options(img_opts)

    # MediaPipe Pose
    pose_path = "pose_landmarker.task"
    if not os.path.exists(pose_path):
        print("  Downloading pose model...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_heavy/float16/1/"
            "pose_landmarker_heavy.task",
            pose_path,
        )
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=pose_path),
        min_pose_detection_confidence=0.4,
    )
    pose_det = vision.PoseLandmarker.create_from_options(pose_opts)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    skip = int(fps * interval)
    events, prev_hist, frame_idx = [], None, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip != 0:
            frame_idx += 1
            continue

        t = round(frame_idx / fps, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Object classification
        cls_result = classifier.classify(mp_img)
        if cls_result.classifications:
            for cat in cls_result.classifications[0].categories:
                if cat.score > 0.3 and cat.index < len(all_labels):
                    label = all_labels[cat.index]
                    for name, ids in TARGETS.items():
                        if cat.index in ids:
                            events.append(
                                {
                                    "time": t,
                                    "type": "object",
                                    "object": name,
                                    "label": label,
                                    "confidence": round(cat.score, 3),
                                }
                            )

        # Scene change
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)
        hist = hist.flatten()
        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
            if diff > 40:
                events.append(
                    {"time": t, "type": "scene_change", "score": round(diff, 1)}
                )
        prev_hist = hist

        # Pose actions
        pose_res = pose_det.detect(mp_img)
        for lm in (
            pose_res.pose_landmarks if hasattr(pose_res, "pose_landmarks") else []
        ):
            for action in _detect_actions(lm, h, w):
                events.append({"time": t, "type": "action", "action": action})

        frame_idx += 1

    cap.release()
    classifier.close()
    pose_det.close()

    return {
        "video": os.path.basename(video_path),
        "duration": round(frame_idx / fps, 1),
        "total_events": len(events),
        "events": events,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python demo_module2.py video.mp4")
        return
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    print(f"Analyzing: {path}")
    result = analyze(path)
    print(f"\nDuration: {result['duration']}s | Events: {result['total_events']}")
    for e in result["events"][:20]:
        if e["type"] == "object":
            print(
                f"  [{e['time']}s] {e['object']} ({e['label']}) {e['confidence']:.0%}"
            )
        elif e["type"] == "scene_change":
            print(f"  [{e['time']}s] scene change")
        elif e["type"] == "action":
            print(f"  [{e['time']}s] action: {e['action']}")
    if len(result["events"]) > 20:
        print(f"  ... and {len(result['events']) - 20} more")
    out = os.path.splitext(path)[0] + "_visual.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
