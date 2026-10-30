# --- Stage 1: build the React app (served by the API at /app) ---
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- Stage 2: the API ---
FROM python:3.12-slim

# Set environment variables to optimize Python & memory
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MALLOC_ARENA_MAX=2

WORKDIR /app

# Install system dependencies needed for compiling certain python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies first (leverage Docker cache)
COPY requirements.txt .
COPY requirements/ requirements/
# The extra index provides the CPU-only torch build (the default Linux wheel bundles CUDA libraries)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy project files
COPY . .
COPY --from=frontend /frontend/dist /app/frontend/dist

# Install the project as a package
RUN pip install .

# Optional transformer tier (the deciding local model). Leave empty on hosts with less than ~2 GB RAM;
# e.g. TRANSFORMER_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest
ARG TRANSFORMER_MODEL=""
RUN if [ -n "$TRANSFORMER_MODEL" ]; then python -c "from huggingface_hub import snapshot_download as s; \
s('$TRANSFORMER_MODEL', local_dir='artifacts/model_trainer/model_DistilBERT', \
allow_patterns=['*.json', '*.txt', 'pytorch_model.bin', '*.safetensors'])"; fi

# Expose API port
EXPOSE 8000

# Run uvicorn with optimized workers & threads
# Apply database migrations, then serve. One uvicorn worker on purpose: the rate limiter and caches live in
# process (see docs/DEPLOYMENT.md before scaling out).
# PORT is set by hosts such as Render; 8000 otherwise
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --limit-concurrency 50"]
