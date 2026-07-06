# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

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

# Install curl for vendor asset download
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Download vendor JS/CSS — cached as long as pip packages don't change
RUN mkdir -p /app/web/js/vendor /app/web/css/vendor && \
    curl -sSfL "https://cdn.jsdelivr.net/npm/marked@12/marked.min.js"                                               -o /app/web/js/vendor/marked.min.js && \
    curl -sSfL "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"                        -o /app/web/js/vendor/highlight.min.js && \
    curl -sSfL "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css"              -o /app/web/css/vendor/github-dark.min.css

# Pre-download the embedding model — cached as long as pip packages don't change.
# Must be BEFORE COPY . . so code edits don't invalidate this ~400 MB layer.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# Copy application source (only invalidates layers below this line)
COPY . .

RUN mkdir -p /app/data/chromadb

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/app/.cache/huggingface

EXPOSE 8080

CMD ["python", "main.py"]
