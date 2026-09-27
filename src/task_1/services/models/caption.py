import torch
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor

from local import device, require_model

_model = None
_processor = None
_tokenizer = None


def load() -> None:
    global _model, _processor, _tokenizer
    source = require_model("CAPTION_MODEL", "tuman/vit-rugpt2-image-captioning", "Caption model")
    print(f"Loading caption model from {source}")
    _processor = ViTImageProcessor.from_pretrained(source, local_files_only=True)
    _tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    _model = VisionEncoderDecoderModel.from_pretrained(source, local_files_only=True)
    _model.to(device())
    _model.eval()
    print("Caption model loaded")


def unload() -> None:
    global _model, _processor, _tokenizer
    _model = None
    _processor = None
    _tokenizer = None


def caption(image: Image.Image) -> str:
    if _model is None or _processor is None or _tokenizer is None:
        raise RuntimeError("Caption model is not loaded")
    rgb = image.convert("RGB")
    pixels = _processor(images=rgb, return_tensors="pt").pixel_values.to(device())
    with torch.no_grad():
        output_ids = _model.generate(pixels, max_length=32, num_beams=4)
    text = _tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text.strip()
