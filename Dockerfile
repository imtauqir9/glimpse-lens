# One image, four runnables (command picks which):
#   API   (default CMD)                    — presign + register + search + UI :8000
#   Worker (python -m src.worker)          — Prefect flow worker (user ingest)
#   CLIP  (uvicorn src.clip_service:app)   — one warm model behind a URL :8001
#   Seed  (python -m src.seed)             — one-shot sample gate, then exits
FROM python:3.11-slim

# ffmpeg = frame sampling. nodejs = the JavaScript runtime yt-dlp needs to
# extract YouTube formats (without it, EVERY YouTube video fails with "This
# video is not available"). Both matter only to the worker but cost little here.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# CPU-only torch first: the default Linux wheel drags in ~6GB of CUDA libs
# that CLIP-on-CPU never uses.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY ui/ ui/

EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
