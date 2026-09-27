import base64

from PIL import Image

import services.models.caption as captioner
import services.models.llm as llm
import services.models.moderate as moderate
import services.models.tts as tts
import services.models.video as video_detector


class SensitiveContentError(ValueError):
    pass


def ask(image: Image.Image, question: str) -> dict:
    if moderate.is_sensitive_text(question):
        raise SensitiveContentError("Question rejected as sensitive")
    
    if moderate.is_sensitive_image(image):
        raise SensitiveContentError("Image rejected as sensitive")
    
    image_caption = captioner.caption(image)
    if moderate.is_sensitive_text(image_caption):
        raise SensitiveContentError("Caption rejected as sensitive")
    
    text = llm.answer(question, image_caption)

    return {"caption": image_caption, "answer": text}


def describe_video(video_bytes: bytes, voice: str = "default") -> dict:
    detection = video_detector.detect_objects(video_bytes)
    description = llm.describe_objects(detection["object_counts"])
    audio = tts.synthesize(description, voice=voice)
    return {
        "objects": detection,
        "description": description,
        "audio_base64": base64.b64encode(audio).decode("ascii"),
        "audio_format": "wav",
    }
