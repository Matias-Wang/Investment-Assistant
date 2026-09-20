# ==========================================
# Dockerfile for LLM Investment Assistant
# Multi-stage build for security and minimal image size
# ==========================================

FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ------------------------------------------
# Final runtime image (without gcc/build tools)
# ------------------------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    APP_ENV=production

WORKDIR /app

# 複製已安裝的 Python 套件
COPY --from=builder /install /usr/local

# 建立非 root 執行使用者與資料目錄
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data && \
    chown -R appuser:appuser /app

# 複製應用程式原始碼
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8080

CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
