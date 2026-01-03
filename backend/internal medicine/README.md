# MediRAG backend (FastAPI)

This folder hosts a FastAPI backend that ports the logic from `pp1.ipynb` into importable Python modules and exposes Postman-testable APIs.

## 1) Setup

### Recommended (Windows): use Python 3.11

On Windows, `spacy`/`scispacy` dependencies (notably `thinc`) often require native wheels. If you're on Python 3.12 you may get build errors like **"Microsoft Visual C++ 14.0 or greater is required"**.

The most reliable path is:

- Install Python **3.11.x** (or use Conda with `python=3.11`)
- Recreate your venv
- Then install dependencies

Example (Conda)::

- `conda create -n medირag python=3.11 -y`
- `conda activate medirag`
- `pip install -r requirements.txt`

### If you must use Python 3.12

Install **Microsoft C++ Build Tools** (MSVC 14+):

- https://visualstudio.microsoft.com/visual-cpp-build-tools/

Then retry:

- `python -m pip install --upgrade pip`
- `pip install -r requirements.txt`

Create and activate a Python env, then install dependencies:

- `pip install -r requirements.txt`

Notes:
- Model downloads happen on first run (HuggingFace).
- `faiss-cpu` can be tricky on native Windows depending on your Python version; if pip fails, use Conda or WSL.
- `scispacy==0.5.4` requires `spacy>=3.7,<3.8`, and the included SciSpacy models require `spacy>=3.7.4` (this repo pins `spacy==3.7.4`).

## 2) Run

From this `backend/` directory:

- `uvicorn app.main:app --host 0.0.0.0 --port 8002`

## 3) Environment variables (optional)

- `MEDIRAG_KB_PATH` (default: `backend/miriad_balanced_300.json`)
- `MEDIRAG_INDEX_PATH` (default: `backend/data/kb_faiss.index`)
- `MEDIRAG_META_PATH` (default: `backend/data/kb_metadata.json`)
- `MEDIRAG_DEVICE` (default: `0`; set to `-1` for CPU)
- `MEDIRAG_GENERATOR_PROVIDER` (default: `ollama`; set to `hf` to use Hugging Face)
- `MEDIRAG_OLLAMA_URL` (default: `http://localhost:11434`)
- `MEDIRAG_OLLAMA_MODEL` (default: `phi:2.7b`; set to your exact `ollama list` model name)
- `MEDIRAG_OLLAMA_TIMEOUT_S` (default: `60`)
- `MEDIRAG_TOP_K` (default: `5`)
- `MEDIRAG_SIM_THRESHOLD` (default: `0.7`)
- `MEDIRAG_CORS_ORIGINS` (default: `http://localhost:3000`)

## 4) Postman

### Health
- `GET http://localhost:8000/health`

### RAG answer
- `POST http://localhost:8000/rag/answer`
- JSON body:
  ```json
  {"query":"How does iloprost-aerosol affect the hemodynamics of patients with IPAH?","top_k":5,"gen_max_length":256,"temperature":0.0}
  ```

Response:
- JSON includes only:
  - `final_answer`

### Verified answer (RAG + hallucination checks)
- `POST http://localhost:8000/rag/answer-verified`
- JSON body: same as above

Response includes `final_answer` plus debug fields (`retrieved`, `prompt`, `nli_results`).
