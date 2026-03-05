# MediRAG backend

This folder contains the refactored Python backend package extracted from `test_ready.ipynb`.

## Requirements

- Python 3.10+ recommended
- LM Studio running locally with an OpenAI-compatible server enabled
- Semantic Scholar API key (for metadata verification)

## Setup (Windows cmd.exe)

Create and activate a venv, then install dependencies:

```bat
cd /d "d:\research support\miriad\rag_pp1\MediRAG\backend"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Set required environment variables:

```bat


(Optional) override data/index locations:

```bat
set MEDIRAG_JSONL_PATH=%CD%\sampled_2500_per_specialty.jsonl
set MEDIRAG_CACHE_DIR=%CD%\.cache
```

## Run a quick smoke test

```bat
python run_smoke.py
```

## Run the FastAPI server

```bat
uvicorn run_api:app --host 127.0.0.1 --port 8003 --reload
```

Then POST a JSON body to `http://127.0.0.1:8000/query`:

```json
{ "query": "What is HIV?", "top_k": 5 }
```

## Notes

- The retrieval/scoring/prompt logic is preserved from the notebook; model access is routed through LM Studio's local API.
- Semantic Scholar calls are rate-limited (1 request/sec) exactly like the notebook.
