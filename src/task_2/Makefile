VENV := .venv
PYTHON ?= $(shell if [ -x /usr/bin/python3 ]; then echo /usr/bin/python3; else command -v python3; fi)
PY := $(VENV)/bin/python
PIP := $(PY) -m pip

-include .env
export

KAGGLE_DATASET ?= otto/recsys-dataset
DATA_DIR := data
MAX_SESSIONS ?= 20000
OPENBLAS_NUM_THREADS ?= 1
export OPENBLAS_NUM_THREADS

.PHONY: help env install download run clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

env: ## Create the virtualenv
	$(PYTHON) -m venv $(VENV)

install: env ## Install Python dependencies
	$(PIP) install -r requirements.txt

download: ## Download the OTTO dataset into ./data (run install first)
	@test -x "$(PY)" || { echo "Run 'make install' first"; exit 1; }
	@test -n "$$KAGGLE_USERNAME" || { echo "KAGGLE_USERNAME/KAGGLE_KEY not set. Copy .env.example to .env and fill in your Kaggle API credentials"; exit 1; }
	@mkdir -p $(DATA_DIR)
	$(PY) scripts/download_dataset.py --dataset "$(KAGGLE_DATASET)" --dest "$(DATA_DIR)"

run: ## Evaluate popularity/covisitation/ALS recall@k (run download first; set MAX_SESSIONS for a quick run)
	@test -x "$(PY)" || { echo "Run 'make install' first"; exit 1; }
	@test -n "$$(find $(DATA_DIR) -mindepth 1 ! -name '.gitkeep' ! -name '.gitignore' 2>/dev/null)" || { echo "No data in $(DATA_DIR). Run 'make download' first"; exit 1; }
	$(PY) main.py $(if $(MAX_SESSIONS),--max-sessions $(MAX_SESSIONS),) evaluate

clean: ## Remove venv and downloaded data
	rm -rf $(VENV)
	find $(DATA_DIR) -mindepth 1 ! -name '.gitkeep' ! -name '.gitignore' -delete
