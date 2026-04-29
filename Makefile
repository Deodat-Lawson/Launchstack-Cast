.PHONY: install ingest app test lint clean

PYTHON ?= python3.11
VIDEO  ?=
OUT    ?= data/features/$(notdir $(basename $(VIDEO))).parquet

install:
	uv venv --python $(PYTHON)
	uv pip install -e ".[dev]"

ingest:
	@if [ -z "$(VIDEO)" ]; then echo "usage: make ingest VIDEO=path/to/clip.mp4"; exit 2; fi
	python -m drone_search ingest --video $(VIDEO) --out $(OUT)

app:
	streamlit run app/streamlit_app.py

test:
	pytest -v

lint:
	ruff check src tests app

clean:
	rm -rf .venv build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
