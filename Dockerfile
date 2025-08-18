# Dockerfile for RAG Web UI (FastAPI + ChromaDB + Gemini + React WebUI)
FROM python:3.11-slim

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies (for PDF, OCR, build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Build WebUI if package.json exists (optional, non-fatal)
RUN if [ -f src/webui/package.json ]; then \
      cd src/webui && npm ci || npm install && npm run build || true && cd /app ; \
    fi

# Set environment variables for ChromaDB and uploads
ENV CHROMA_PATH=/app/data/chroma \
    UPLOAD_DIR=/app/data/uploads

EXPOSE 8000

# Run FastAPI app with Uvicorn
CMD ["sh", "-c", "uvicorn src.api.server:app --host 0.0.0.0 --port 8000"]


