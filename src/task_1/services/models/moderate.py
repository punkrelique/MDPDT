import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from PIL import Image

from local import pipeline_device, require_model

_text = None
_image = None

_LABELS = [
    "обычное безопасное содержание",
    "сексуальный контент 18+",
    "жестокое насилие",
    "суицид или самоповреждение",
]
_SAFE = "обычное безопасное содержание"
_THRESHOLD = float(os.getenv("SENSITIVE_TEXT_THRESHOLD", "0.55"))
_NSFW_THRESHOLD = float(os.getenv("NSFW_THRESHOLD", "0.5"))


def load() -> None:
    global _text, _image
    text_path = require_model(
        "SENSITIVE_TEXT_MODEL",
        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        "Sensitive text model",
    )
    image_path = require_model("NSFW_MODEL", "Falconsai/nsfw_image_detection", "NSFW image model")
    print(f"Loading sensitive-text model from {text_path}")
    tokenizer = AutoTokenizer.from_pretrained(text_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(text_path, local_files_only=True)
    _text = pipeline(
        "zero-shot-classification",
        model=model,
        tokenizer=tokenizer,
        device=pipeline_device(),
    )
    print(f"Loading NSFW image model from {image_path}")
    _image = pipeline(
        "image-classification",
        model=image_path,
        device=pipeline_device(),
    )
    print("Moderation models loaded")


def unload() -> None:
    global _text, _image
    _text = None
    _image = None


def is_sensitive_text(text: str) -> bool:
    if _text is None:
        raise RuntimeError("Text moderation is not loaded")
    result = _text(text, candidate_labels=_LABELS, hypothesis_template="Этот текст про {}.")
    label = result["labels"][0]
    score = float(result["scores"][0])
    return label != _SAFE and score >= _THRESHOLD


def is_sensitive_image(image: Image.Image) -> bool:
    if _image is None:
        raise RuntimeError("Image moderation is not loaded")
    scores = {row["label"].lower(): float(row["score"]) for row in _image(image)}
    return scores.get("nsfw", 0.0) >= _NSFW_THRESHOLD
