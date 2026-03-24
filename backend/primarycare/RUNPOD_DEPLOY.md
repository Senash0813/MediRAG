# RunPod Deployment Guide (MediRAG Backend)

## 1) Build and push image
Build locally (or in CI) and push to Docker Hub/GHCR:

```bash
docker build -t <registry>/<image>:latest .
docker push <registry>/<image>:latest
```

## 2) Create RunPod Pod
Use the pushed image and set these environment variables:

- `PORT=8000`
- `MEDIRAG_LLM_PROVIDER=ollama`
- `MEDIRAG_OLLAMA_BASE_URL=http://<ollama-host>:11434`
- `MEDIRAG_OLLAMA_MODEL=qwen2.5:3b`
- `S2_API_KEY=<your_semantic_scholar_key>`

Optional if you want to override defaults:

- `MEDIRAG_PASSAGES_PATH=/app/resources/selected_specialties.jsonl`
- `MEDIRAG_FAISS_INDEX_PATH=/app/resources/faiss/index.faiss`
- `MEDIRAG_EMBEDDINGS_PATH=/app/resources/faiss/embeddings.npy`
- `MEDIRAG_S2_CACHE_PATH=/app/resources/caches/semantic_scholar_cache.json`

## 3) Networking
Expose container port `8000` in RunPod. The container starts with:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

## 4) Health check
After startup, verify endpoint:

- `GET /docs`
- `POST /query4`

## 5) Ollama placement options

### Option A: Ollama in same pod
Set `MEDIRAG_OLLAMA_BASE_URL=http://127.0.0.1:11434` and run Ollama in another process/container in that pod.

### Option B: External Ollama endpoint
Set `MEDIRAG_OLLAMA_BASE_URL` to reachable internal/private URL.

## 6) Important note
`S2_API_KEY` is required by current app startup logic. If missing, app exits by design.
