FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	POETRY_VIRTUALENVS_CREATE=false \
	PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# If a web UI exists, build it (Vite/Node expected). This is optional; skip on failure.
RUN if [ -f webui/package.json ]; then \
      apt-get update && apt-get install -y --no-install-recommends nodejs npm && \
      cd webui && npm ci || npm install && npm run build || true && cd /app ; \
    fi

ENV CHROMA_PATH=/app/data/chroma \
	UPLOAD_DIR=/app/data/uploads

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api.server:app --host 0.0.0.0 --port 8000"]


