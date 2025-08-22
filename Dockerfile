FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for unstructured and chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpython3-dev \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Update pip to avoid installation issues
RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE $PORT

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]
