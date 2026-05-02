#!/bin/bash
set -e

echo "=========================================="
echo "🚀 MediRAG Cardiology - Unified Container"
echo "=========================================="

# Start Ollama in the background
echo "[1/4] Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!
sleep 3

# Wait for Ollama to be ready
echo "[2/4] Waiting for Ollama to be ready..."
for i in {1..30}; do
    if curl -f http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Ollama failed to start after 30 seconds"
        kill $OLLAMA_PID || true
        exit 1
    fi
    echo "  Attempt $i/30..."
    sleep 1
done

# Pull the quantized Phi model
echo "[3/4] Pulling phi:2.7b model (this may take 2-5 minutes on first run)..."
ollama pull phi:2.7b || {
    echo "❌ Failed to pull phi:2.7b"
    kill $OLLAMA_PID || true
    exit 1
}
echo "✅ phi:2.7b model ready"

# Start FastAPI
echo "[4/4] Starting FastAPI server..."
/opt/venv/bin/uvicorn api:app --host 0.0.0.0 --port 8001 &
FASTAPI_PID=$!

echo "=========================================="
echo "✅ MediRAG Cardiology is READY"
echo "   Ollama:  http://localhost:11434"
echo "   FastAPI: http://localhost:8001"
echo "=========================================="

# Keep both processes running
wait $OLLAMA_PID $FASTAPI_PID
