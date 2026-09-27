import os

import torch


def model_path(repo: str) -> str:
    return os.path.join("models", os.path.basename(repo))


def require_model(env_key: str, default_repo: str, name: str) -> str:
    repo = os.getenv(env_key, default_repo)
    path = model_path(repo)
    if not os.path.isfile(os.path.join(path, "config.json")):
        raise RuntimeError(f"{name} not found at {path}. Run 'make download' first.")
    return path


def device() -> str:
    return os.getenv("DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")


def pipeline_device():
    name = device()
    if name == "cpu":
        return -1
    if name.startswith("cuda"):
        parts = name.split(":")
        return int(parts[1]) if len(parts) == 2 else 0
    return name
