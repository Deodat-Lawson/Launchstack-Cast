FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DRONE_SEARCH_DATA_DIR=/app/data \
    DRONE_SEARCH_YOLO_MODEL=yolov8n.pt

# OpenCV + ffmpeg runtime; build essentials for any wheels that need compiling.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# uv for fast resolved installs.
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy only the dep manifest first so the install layer caches across code edits.
COPY pyproject.toml ./
COPY src ./src
RUN uv pip install --system -e ".[dev]"

# Pre-download model weights so the first request doesn't pay the cost.
RUN python -c "from drone_search.embed import _warmup; _warmup()" \
 && python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Now copy the rest of the app.
COPY app ./app
COPY tests ./tests
COPY README.md LICENSE ./

RUN mkdir -p /app/data/uploads /app/data/features

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
