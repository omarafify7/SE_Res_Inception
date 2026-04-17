FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System deps for Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev libpng-dev curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Python deps
COPY inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY model.py .
COPY inference/__init__.py inference/
COPY inference/main.py inference/

# Checkpoints are mounted via docker-compose volume, not copied

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "inference.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
