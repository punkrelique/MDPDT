import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from local import device, require_model

_model = None
_tokenizer = None

_SYSTEM = (
    "Ты отвечаешь на вопросы по изображению. "
    "Тебе даны только текстовая подпись к фото и вопрос пользователя. "
    "Отвечай коротко, по-русски и только на основе подписи. "
    "Не выдумывай объекты, которых нет в подписи. "
    "Если подписи недостаточно, скажи, что по подписи этого не видно."
)

_VIDEO_SYSTEM = (
    "Тебе дан список объектов, обнаруженных детектором на видео, на английском "
    "языке, с количеством кадров, на которых каждый объект встретился. "
    "Напиши одно связное предложение на русском языке о том, что происходит на видео. "
    "Не выдумывай объекты, которых нет в списке. "
    "Если список пуст, скажи, что на видео не обнаружено объектов."
)


def load() -> None:
    global _model, _tokenizer
    source = require_model("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct", "LLM")
    print(f"Loading LLM from {source}")
    _tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    dtype = torch.bfloat16 if device().startswith("cuda") else torch.float32
    _model = AutoModelForCausalLM.from_pretrained(
        source,
        local_files_only=True,
        torch_dtype=dtype,
        device_map=device(),
    )
    _model.eval()
    print("LLM loaded")


def unload() -> None:
    global _model, _tokenizer
    _model = None
    _tokenizer = None


def _generate(system: str, user: str, max_new_tokens: int) -> str:
    if _model is None or _tokenizer is None:
        raise RuntimeError("LLM is not loaded")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    with torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = out[0, inputs["input_ids"].shape[1] :]
    return _tokenizer.decode(generated, skip_special_tokens=True).strip()


def answer(question: str, image_caption: str) -> str:
    user = f"Подпись к изображению: {image_caption}\nВопрос: {question}"
    return _generate(_SYSTEM, user, max_new_tokens=128)


def describe_objects(object_counts: dict[str, int]) -> str:
    if object_counts:
        listing = ", ".join(f"{label} ({count})" for label, count in object_counts.items())
    else:
        listing = "объекты не обнаружены"
    user = f"Обнаруженные объекты: {listing}"
    return _generate(_VIDEO_SYSTEM, user, max_new_tokens=96)
