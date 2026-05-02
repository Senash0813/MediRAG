FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8002 \
    OLLAMA_HOST=http://127.0.0.1:11434 \
    OLLAMA_GENERATOR_MODEL=phi:2.7b \
    OLLAMA_JUDGE_MODEL=phi:2.7b \
    KB_FILE=data/kb.json.json \
    EMBEDDINGS_DIR=storage/embeddings \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash curl ca-certificates zstd \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --default-timeout=300 --retries 10 \
        --index-url https://download.pytorch.org/whl/cpu torch==2.3.1+cpu \
    && pip install --no-cache-dir --default-timeout=300 --retries 10 -r requirements.txt

COPY app ./app
COPY storage/embeddings/kb_faiss.index ./storage/embeddings/kb_faiss.index
COPY storage/embeddings/kb_metadata.json ./storage/embeddings/kb_metadata.json
COPY storage/embeddings/scope_faiss.index ./storage/embeddings/scope_faiss.index
COPY storage/embeddings/scope_metadata.json ./storage/embeddings/scope_metadata.json
COPY docker-entrypoint.sh ./docker-entrypoint.sh

RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 8002

ENTRYPOINT ["/app/docker-entrypoint.sh"]