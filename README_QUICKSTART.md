# MediRAG Quickstart

This guide gets the full MediRAG stack running locally with minimal setup.

## 1) Prerequisites

- Python 3.10+
- Node.js 18+ and npm
- MongoDB running (local or remote)
- Ollama installed and running (`ollama serve`)

Optional but recommended:
- CUDA GPU for `backend/neurology`
- Semantic Scholar API key for `backend/primarycare`

## 2) Repository Layout (What Runs Where)

- Frontend: `frontend` (Next.js, port `3000`)
- Neurology backend: `backend/neurology` (port `8000`)
- Cardiology backend: `backend/cardiology` (port `8001`)
- Internal medicine backend: `backend/internal_medicine` (port `8002`)
- Primary care backend: `backend/primarycare` (port `8003`)

## 3) Frontend Setup

```bash
cd frontend
npm install
```

Create `frontend/.env` with:

```env
MONGODB_URI=mongodb://localhost:27017
NEXTAUTH_SECRET=replace_with_strong_secret
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
```

Run:

```bash
npm run dev
```

Frontend will be at: `http://localhost:3000`

## 4) Backend Setup (One-Time Per Service)

Open separate terminals for each backend.

### A) Neurology (`8000`)

```bash
cd backend/neurology
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Notes:
- Requires local model/index artifacts under `data/` and `model_weights/`.
- Set `HF_TOKEN` if required by your model access.

### B) Cardiology (`8001`)

```bash
cd backend/cardiology
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8001 --reload
```

If FAISS index is missing:

```bash
python build_index.py
```

### C) Internal Medicine (`8002`)

```bash
cd backend/internal_medicine
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

Minimum `.env` check:
- `KB_FILE`
- `OLLAMA_HOST`
- `OLLAMA_GENERATOR_MODEL`

### D) Primary Care (`8003`)

```bash
cd backend/primarycare
pip install -r requirements.txt
python main.py
```

Required environment:

```env
S2_API_KEY=your_semantic_scholar_key
MEDIRAG_LLM_PROVIDER=ollama
MEDIRAG_OLLAMA_BASE_URL=http://127.0.0.1:11434
MEDIRAG_OLLAMA_MODEL=qwen2.5:3b
```

Notes:
- Primary care expects resource artifacts under `backend/primarycare/resources`.
- Current code references `medirag.data.*` modules; ensure they exist in your local working version.

## 5) Ollama Models (Typical)

Pull required local models before running:

```bash
ollama pull phi:2.7b
ollama pull qwen2.5:3b
```

Keep Ollama server running:

```bash
ollama serve
```

## 6) Health Checks

Once running, verify:

- Neurology: `http://127.0.0.1:8000/docs`
- Cardiology: `http://127.0.0.1:8001/health`
- Internal medicine: `http://127.0.0.1:8002/health`
- Primary care: `http://127.0.0.1:8003/docs`
- Frontend: `http://127.0.0.1:3000`

## 7) Known Integration Notes

- Cluster 3 frontend route currently points to `8002/rag/answer-verified`, while internal medicine API exposes `/rag` and `/query3`.
- If cluster 3 fails from UI, call `/query3` directly or align the route.
- Large datasets, model weights, and indices are mostly gitignored and must be prepared locally.
