import re

_RUSSIAN_LETTER = re.compile(r"[А-Яа-яЁё]")


def is_russian(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    return bool(letters) and all(_RUSSIAN_LETTER.fullmatch(ch) for ch in letters)
