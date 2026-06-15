# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Create data directory and download vendor JS/CSS for offline PWA
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    mkdir -p /app/data/chromadb /app/web/js/vendor /app/web/css/vendor && \
    curl -sS "https://cdn.jsdelivr.net/npm/marked@12/marked.min.js"                                               -o /app/web/js/vendor/marked.min.js && \
    curl -sS "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"                        -o /app/web/js/vendor/highlight.min.js && \
    curl -sS "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css"              -o /app/web/css/vendor/github-dark.min.css && \
    rm -rf /var/lib/apt/lists/*

# Pre-download the embedding model so the first request doesn't time out.
# HF_HOME is pinned to /app/.cache so it's accessible by appuser later.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/app/.cache/huggingface

EXPOSE 8080

CMD ["python", "main.py"]
