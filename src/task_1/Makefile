VENV := .venv
PYTHON ?= $(shell if [ -x /usr/bin/python3 ]; then echo /usr/bin/python3; else command -v python3; fi)
PY := $(VENV)/bin/python
PIP := $(PY) -m pip

HOST ?= 0.0.0.0
PORT ?= 8000

-include .env
export

SENSITIVE_TEXT_MODEL ?= MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
NSFW_MODEL ?= Falconsai/nsfw_image_detection
CAPTION_MODEL ?= tuman/vit-rugpt2-image-captioning
LLM_MODEL ?= Qwen/Qwen2.5-3B-Instruct
VIDEO_MODEL ?= hustvl/yolos-tiny
TTS_MODEL ?= Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
TOXICITY_MODEL ?= cointegrated/rubert-tiny-toxicity

MODELS := $(SENSITIVE_TEXT_MODEL) $(NSFW_MODEL) $(CAPTION_MODEL) $(LLM_MODEL) $(VIDEO_MODEL) $(TTS_MODEL) $(TOXICITY_MODEL)

.PHONY: help env install download run clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

env:
	$(PYTHON) -m venv $(VENV)

install: env
	$(PIP) install -r requirements.txt

download:
	@test -x "$(PY)" || { echo "Run 'make install' first"; exit 1; }
	@mkdir -p models
	@for repo in $(MODELS); do \
		model_name=$$(basename $$repo); \
		model_dir="models/$$model_name"; \
		mkdir -p $$model_dir; \
		$(PY) -c "from huggingface_hub import snapshot_download; snapshot_download('$$repo', local_dir='$$model_dir')"; \
	done

run:
	@test -x "$(PY)" || { echo "Run 'make install' first"; exit 1; }
	@for repo in $(MODELS); do \
		model_name=$$(basename $$repo); \
		test -f "models/$$model_name/config.json" || { echo "$$model_name missing. Run 'make download' first"; exit 1; }; \
	done
	$(PY) -m uvicorn main:app --host $(HOST) --port $(PORT)

clean:
	rm -rf $(VENV) models
