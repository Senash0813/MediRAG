#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -f ".env.ollama" ]; then
  set -a
  source .env.ollama
  set +a
fi

export MEDIRAG_LLM_PROVIDER="${MEDIRAG_LLM_PROVIDER:-ollama}"
export PORT="${PORT:-8000}"

if [ -z "${S2_API_KEY:-}" ] || [ "${S2_API_KEY}" = "your_s2_api_key_here" ]; then
  echo "[ERROR] S2_API_KEY must be set for startup (Semantic Scholar key required)."
  exit 1
fi

echo "Starting API with Ollama provider..."
echo "MEDIRAG_LLM_PROVIDER=${MEDIRAG_LLM_PROVIDER}"
echo "MEDIRAG_OLLAMA_BASE_URL=${MEDIRAG_OLLAMA_BASE_URL:-}"
echo "MEDIRAG_OLLAMA_MODEL=${MEDIRAG_OLLAMA_MODEL:-}"
echo "PORT=${PORT}"

exec uvicorn main:app --host 0.0.0.0 --port "${PORT}"
