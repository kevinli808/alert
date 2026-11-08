FROM python:3.11-slim

# ffmpeg lets the API accept WebM/MP4 from MediaRecorder; libsndfile is bundled with soundfile wheels.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt requirements-local.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-local.txt

COPY app ./app
# Bake YAMNet into the image so cold starts don't download it.
ENV TFHUB_CACHE_DIR=/srv/tfhub_cache TF_CPP_MIN_LOG_LEVEL=2
RUN python -c "import tensorflow_hub as hub; hub.load('https://tfhub.dev/google/yamnet/1')"

RUN useradd --create-home appuser && chown -R appuser /srv
USER appuser

EXPOSE 8000
# One worker: the model, rate limiter and event cooldown live in process memory.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers"]
