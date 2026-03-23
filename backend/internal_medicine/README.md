# Internal Medicine RAG (FastAPI)

This converts the notebook pipeline into a FastAPI backend with:
- KB loader (JSON/JSONL)
- SentenceTransformers embeddings + FAISS index persisted to `storage/embeddings/`
- Domain/scope gate using a separate passage-level FAISS index
- RAG answering using **Ollama** (generator model set to `phi` by default)
- Post-answer verification layer (NER risk routing, semantic evidence matching, NLI + judge fallback, regeneration, final answer reconstruction)

## 1) Prereqs

- Python 3.10+ recommended
- Ollama installed and running

Pull the generator model (name must match your local Ollama tag):

```bash
ollama pull phi
```

## 2) Install

```bash
pip install -r requirements.txt
```

## 3) Configure

Copy `.env.example` to `.env` and set `KB_FILE` to your dataset.

Indexes/metadata are stored in `storage/embeddings/`:
- `kb_faiss.index`, `kb_metadata.json`
- `scope_faiss.index`, `scope_metadata.json`

## 4) Run

```bash
uvicorn app.main:app --port 8002 --reload
Open docs at:
- http://127.0.0.1:8000/docs

## 5) API

- `GET /health`
- `POST /rag` – domain-gated RAG only
- `POST /query3` – RAG + verification/regeneration
- `POST /reindex` – rebuild indices (optionally force)

Example request:

```json
{
  "query": "What is HIV?",
  "top_k": 5,
  "temperature": 0.0,
  "verify": true
}
```

## Optional: SciSpacy models

If you want SciSpacy sentence splitting + BC5CDR NER (as in the notebook), install the optional dependencies and models listed at the bottom of `requirements.txt`.
