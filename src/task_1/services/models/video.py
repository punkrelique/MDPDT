import os
import tempfile

import cv2
from PIL import Image
from transformers import pipeline

from local import pipeline_device, require_model

_model = None

_THRESHOLD = float(os.getenv("VIDEO_DETECTION_THRESHOLD", "0.7"))
_MAX_FRAMES = int(os.getenv("VIDEO_MAX_FRAMES", "16"))
_SAMPLE_FPS = float(os.getenv("VIDEO_SAMPLE_FPS", "1"))


def load() -> None:
    global _model
    source = require_model("VIDEO_MODEL", "hustvl/yolos-tiny", "Video object detection model")
    print(f"Loading video object detection model from {source}")
    _model = pipeline("object-detection", model=source, device=pipeline_device())
    print("Video object detection model loaded")


def unload() -> None:
    global _model
    _model = None


def _sample_frames(video_path: str) -> list[tuple[float, Image.Image]]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("Could not read video file")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(fps / _SAMPLE_FPS))
    frames = []
    index = 0
    while len(frames) < _MAX_FRAMES:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((index / fps, Image.fromarray(rgb)))
        index += 1
    capture.release()
    if not frames:
        raise ValueError("No frames could be extracted from video")
    return frames


def detect_objects(video_bytes: bytes) -> dict:
    if _model is None:
        raise RuntimeError("Video object detection model is not loaded")

    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as fh:
            fh.write(video_bytes)
        frames = _sample_frames(tmp_path)
    finally:
        os.remove(tmp_path)

    counts: dict[str, int] = {}
    per_frame = []
    for timestamp, image in frames:
        detections = [d for d in _model(image) if d["score"] >= _THRESHOLD]
        for d in detections:
            counts[d["label"]] = counts.get(d["label"], 0) + 1
        per_frame.append(
            {
                "timestamp": round(timestamp, 2),
                "objects": [
                    {
                        "label": d["label"],
                        "score": round(float(d["score"]), 3),
                        "box": d["box"],
                    }
                    for d in detections
                ],
            }
        )

    return {
        "frames_analyzed": len(frames),
        "unique_objects": sorted(counts, key=counts.get, reverse=True),
        "object_counts": counts,
        "frames": per_frame,
    }
