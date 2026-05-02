# Internal Medicine RAG (FastAPI)

This converts the notebook pipeline into a FastAPI backend with:
- KB loader (JSON/JSONL)
- SentenceTransformers embeddings + FAISS index persisted to `storage/embeddings/`
- Domain/scope gate using a separate passage-level FAISS index
- RAG answering using **Ollama** (generator model set to `phi` by default)
- Post-answer verification layer (NER risk routing, semantic evidence matching, NLI + judge fallback, regeneration, final answer reconstruction)

## 1) Prereqs

- Python 3.10+ recommended for local development
- Ollama installed and running for local development

Pull the generator model locally if you are not using Docker:

```bash
ollama pull phi:2.7b
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

## 4) Run locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
Open docs at:
- http://127.0.0.1:8002/docs

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

## 5.1) Ablation study

You can run a stage ablation sweep from the project root with a JSON or JSONL file of examples. Each example should contain at least `query`, and can optionally include `reference_answer` for simple text metrics.

```bash
python -m app.scripts.ablation_study --input data/ablation_examples.json --output-dir ablation_results
```

If you are on Windows PowerShell, you can use the wrapper instead:

```powershell
.\run_ablation.ps1
```

The script writes:
- `summary.json` with one aggregate record per variant
- `all_rows.csv` with per-query results across variants
- One folder per variant with its own `rows.csv` and `summary.json`

Useful variants include `baseline`, `no_domain_gate`, `rag_only`, `no_ner`, `no_nli`, `strict_retrieval`, and `loose_retrieval`.

## Optional: SciSpacy models

If you want SciSpacy sentence splitting + BC5CDR NER (as in the notebook), install the optional dependencies and models listed at the bottom of `requirements.txt`.

## 6) Docker / Runpod

This backend now includes a single-container Docker setup in `Dockerfile`.

Build the image from `backend/internal_medicine`:

```bash
docker build -t internal-medicine-rag .
```

Run it locally:

```bash
docker run --rm -p 8002:8002 internal-medicine-rag
```

The container:
- starts Ollama inside the same image
- pulls `OLLAMA_GENERATOR_MODEL` and `OLLAMA_JUDGE_MODEL` at startup
- warms the Hugging Face / SentenceTransformers models at startup
- serves the FastAPI app on port `8002`
- includes `storage/embeddings/` in the image
