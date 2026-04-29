.PHONY: install ingest app test lint clean docker-build docker-run deploy

PYTHON ?= python3.11
VIDEO  ?=
OUT    ?= data/features/$(notdir $(basename $(VIDEO))).parquet
IMAGE  ?= drone-search

VENV   := .venv
VPY    := $(VENV)/bin/python

install:
	uv venv --python $(PYTHON)
	uv pip install --python $(VPY) -e ".[dev]"

ingest:
	@if [ -z "$(VIDEO)" ]; then echo "usage: make ingest VIDEO=path/to/clip.mp4"; exit 2; fi
	$(VPY) -m drone_search ingest --video $(VIDEO) --out $(OUT)

app:
	$(VPY) -m streamlit run app/streamlit_app.py

test:
	$(VPY) -m pytest -v

lint:
	$(VPY) -m ruff check src tests app

clean:
	rm -rf .venv build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8501:8501 \
		-e GEMINI_API_KEY \
		-v $(PWD)/data:/app/data \
		$(IMAGE)

deploy:
	fly deploy
