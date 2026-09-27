import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from local import require_model

THRESHOLD = float(os.getenv("TOXICITY_THRESHOLD", "0.5"))

tokenizer = None
model = None


def load() -> None:
    global tokenizer, model
    source = require_model("TOXICITY_MODEL", "cointegrated/rubert-tiny-toxicity", "Toxicity model")
    print(f"Loading toxicity filter from {source}")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(source, local_files_only=True)
    model.eval()
    print("Toxicity filter loaded")


def unload() -> None:
    global tokenizer, model
    tokenizer = None
    model = None


def score(text: str) -> float:
    """Probability that the text is toxic or dangerous, in [0, 1]."""
    if model is None or tokenizer is None:
        raise RuntimeError("Toxicity filter is not loaded")
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        proba = torch.sigmoid(model(**inputs).logits)[0]
    return float(1 - proba[0] * (1 - proba[-1]))


def is_toxic(text: str) -> bool:
    return score(text) >= THRESHOLD
