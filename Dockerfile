FROM python:3.10-slim

# Set memory-optimized environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install only ESSENTIAL system dependencies - REDUCED SET
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpython3-dev \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy and install requirements with memory optimization
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=1000 -r requirements.txt \
    && pip cache purge

# Copy application code
COPY . /app

# Create necessary directories with minimal permissions
RUN mkdir -p /app/uploads /app/chroma_db /app/templates \
    && chmod 755 /app

# ADDED: Remove unnecessary files to save space and memory
RUN find /usr/local/lib/python3.10/site-packages -name "*.pyc" -delete \
    && find /usr/local/lib/python3.10/site-packages -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Expose port
EXPOSE 8000

# CHANGED: Use memory-optimized startup with request limits
CMD ["gunicorn", "app:app", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120", "--max-requests", "100", "--max-requests-jitter", "10"]
