import io
import os

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from local import device, require_model

DEFAULT_VOICE = os.getenv("TTS_DEFAULT_VOICE", "Vivian")

model = None


def load() -> None:
    global model
    source = require_model("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "TTS model")
    dtype = torch.bfloat16 if device().startswith("cuda") else torch.float32
    print(f"Loading TTS from {source} (device={device()})")
    model = Qwen3TTSModel.from_pretrained(
        source,
        device_map=device(),
        dtype=dtype,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    print("TTS loaded")


def unload() -> None:
    global model
    model = None


def voices() -> dict:
    if model is None:
        return {"voices": ["default"], "language": "Russian"}

    speakers = list(model.get_supported_speakers())
    return {
        "voices": ["default", *[s for s in speakers if s.lower() != "default"]],
        "language": "Russian",
    }


def synthesize(text: str, voice: str = "default") -> bytes:
    if model is None:
        raise RuntimeError("Model is not loaded")

    speaker = DEFAULT_VOICE if voice.lower() == "default" else voice
    wavs, sample_rate = model.generate_custom_voice(
        text=text,
        speaker=speaker,
        language="Russian",
    )

    if not wavs:
        raise RuntimeError("Model returned no audio")
    
    buf = io.BytesIO()
    sf.write(buf, wavs[0], sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()
