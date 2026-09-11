# Hugging Face Spaces (Docker SDK) — Automated Fake News Detection System
# https://huggingface.co/docs/hub/spaces-sdks-docker
FROM python:3.11-slim

WORKDIR /app

# Build tools for some pip wheels; curl for health/debug if needed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch keeps the image smaller than the default CUDA build.
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-deploy.txt \
    && python -m spacy download en_core_web_sm

# Preserve repo layout: main.py resolves templates/static/results from the repo root.
COPY ["Automated Fake News Detection/app", "Automated Fake News Detection/app"]
COPY templates templates
COPY static static
COPY results results

ENV PYTHONPATH="/app/Automated Fake News Detection"
# SVM is now the default model for optimal performance and fast LIME explanations

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
