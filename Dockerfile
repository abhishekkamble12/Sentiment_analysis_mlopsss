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

# Install the project as a package
RUN pip install .

# Expose API port
EXPOSE 8000

# Run uvicorn with optimized workers & threads
# Apply database migrations, then serve. One uvicorn worker on purpose: the rate limiter and caches live in
# process (see docs/DEPLOYMENT.md before scaling out).
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1 --limit-concurrency 50"]
