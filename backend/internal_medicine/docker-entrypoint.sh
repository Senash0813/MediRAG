#!/usr/bin/env bash
set -euo pipefail

OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
PORT="${PORT:-8002}"
OLLAMA_GENERATOR_MODEL="${OLLAMA_GENERATOR_MODEL:-phi:2.7b}"
OLLAMA_JUDGE_MODEL="${OLLAMA_JUDGE_MODEL:-${OLLAMA_GENERATOR_MODEL}}"

mkdir -p "${HF_HOME:-/root/.cache/huggingface}"

ollama serve >/tmp/ollama.log 2>&1 &
OLLAMA_PID=$!

cleanup() {
  kill "${OLLAMA_PID}" >/dev/null 2>&1 || true
  wait "${OLLAMA_PID}" >/dev/null 2>&1 || true
}

trap cleanup EXIT

echo "Waiting for Ollama to start..."
for _ in $(seq 1 120); do
  if curl -fsS "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -fsS "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
  echo "Ollama failed to start. Log output:"
  cat /tmp/ollama.log
  exit 1
fi

echo "Pre-downloading Hugging Face and SentenceTransformers models..."
python -m app.scripts.warm_models

echo "Pulling Ollama generator model: ${OLLAMA_GENERATOR_MODEL}"
ollama pull "${OLLAMA_GENERATOR_MODEL}"

if [ -n "${OLLAMA_JUDGE_MODEL}" ] && [ "${OLLAMA_JUDGE_MODEL}" != "${OLLAMA_GENERATOR_MODEL}" ]; then
  echo "Pulling Ollama judge model: ${OLLAMA_JUDGE_MODEL}"
  ollama pull "${OLLAMA_JUDGE_MODEL}"
fi

echo "Starting API on port ${PORT}"
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT}"