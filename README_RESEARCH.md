# MediRAG Research Project Documentation

## 1) Project Overview

**MediRAG** is a multi-pipeline medical question-answering research system built to reduce hallucinations in Retrieval-Augmented Generation (RAG).  
Instead of using one generic medical RAG pipeline, the project separates workloads into **four domain-specific pipelines** and integrates them through a unified frontend.

The system is organized as:
- Four specialized backend services (Neurology, Cardiology, Internal Medicine, Primary Care & Mental Health)
- One Next.js frontend for user interaction, cluster selection, and conversation history
- Data preparation logic for specialty-based dataset extraction
- Demo and deployment assets for local and cloud execution

## 2) Research Motivation and Objectives

### Problem
Generic RAG systems can still hallucinate in high-stakes domains like medicine.

### Research Direction
This project investigates whether hallucination mitigation improves when pipelines are tailored to medical subdomains.

### Core Objectives
1. Build **domain-specialized** medical RAG pipelines.
2. Compare retrieval and verification strategies across specialties.
3. Improve answer trustworthiness using:
   - domain/scope gating
   - hybrid retrieval and reranking
   - answer verification/regeneration
   - evidence-aware prompting

## 3) End-to-End Research Workflow (Start to Finish)

1. **Data acquisition and filtering**
   - Source dataset: `miriad/miriad-4.4M` (streamed from HuggingFace in notebooks/scripts)
   - Specialty filtering done in `data division logic/cluster_division.ipynb`

2. **Discipline-specific backend construction**
   - `backend/cardiology`
   - `backend/neurology`
   - `backend/internal_medicine`
   - `backend/primarycare`

3. **Index/model artifact preparation**
   - FAISS indices
   - embedding caches
   - local model weights / adapters
   - optional external enrichment caches (Semantic Scholar)

4. **API exposure per pipeline**
   - each pipeline served via FastAPI on separate ports

5. **Frontend integration**
   - cluster-based routing from one chat interface to four backends

6. **Evaluation and demo layers**
   - stage-wise outputs in cardiology/internal medicine
   - detailed retrieval demo in primary care
   - conversation persistence and auth in frontend

## 4) Repository Structure (Folder-by-Folder)

### Root
- `README.md`  
  Existing high-level project description.
- `backend/`  
  All medical pipeline services.
- `frontend/`  
  Next.js web application.
- `data division logic/`  
  Dataset extraction/filtering notebooks/scripts.
- `media/`  
  Media assets (includes `system.gif` used in root docs).
- `.gitignore`  
  Excludes local data, model weights, vector indices, caches, and env-sensitive artifacts.

### `backend/`
Contains four discipline-oriented backends plus environment folders:

#### `backend/cardiology/`
Files:
- `api.py` (FastAPI endpoints)
- `main.py` (CLI pipeline runner)
- `embedder.py` (InBEDDER + projection)
- `hyde.py` (hypothetical answer generation + embedding)
- `fusion.py` (embedding fusion)
- `retriever.py` (FAISS loading/retrieval)
- `build_index.py` (index construction from cardiology JSON)
- `requirements.txt`
- `test.txt` (example run command)
- `models_download/` (download scripts)
- `notebooks/` (HyDE data prep, finetuning, full/improved pipelines)

#### `backend/neurology/`
Files/folders:
- `main.py` (FastAPI `/query`)
- `config.py` (paths/models/tuning)
- `retrieval/` (`faiss_retriever.py`, `bm25_retriever.py`, `hybrid_rrf.py`)
- `models/` (query rewrite, domain checker, gatekeeper, blueprint, reranker, model managers)
- `prompts/` (system prompts)
- `data/` (local indices; usually not committed)
- `model_weights/` (LoRAs/rewriter weights; usually not committed)
- `Dockerfile`, `check_gpu.py`, `requirements.txt`

#### `backend/internal_medicine/`
Files:
- `app/` package
  - `main.py` (FastAPI app init/lifespan)
  - `api/routes.py` (health, rag, verification, reindex, stage evaluation)
  - `core/config.py` (Pydantic settings)
  - `services/` (`rag.py`, `verification.py`, `domain_gate.py`, etc.)
  - `schemas.py`, `state.py`
- `README.md`
- `STAGE_EVALUATION_API.md`
- `internal_medicine.ipynb`
- `demo.html`
- `requirements.txt`
- `.env.example`

#### `backend/primarycare/`
Files:
- `main.py` (FastAPI app exposing `/query4`, `/retrieve`, `/retrieve/detailed`, `/batch-query`)
- `medirag/` package
  - `api/schemas.py`
  - `config/settings.py`
  - `domain/` (`scope_gate.py`, `scoring.py`, `models.py`)
  - `llm/` (`client.py`, `instructor.py`, `answer.py`, `prompts.py`)
  - `pipeline/orchestrator.py`
  - `retrieval/` (`retriever.py`, `verifier.py`, enrichment clients)
- deployment/docs:
  - `OLLAMA_SETUP.md`
  - `RUNPOD_DEPLOY.md`
  - `start_ollama.sh`
  - `run_ollama.cmd`
  - `Dockerfile`
  - `demo-v3/animated_demo_v3.html`
- `requirements.txt`

> Note: code references `medirag.data.*` modules (`faiss_index`, `loaders`) that are not present in this checkout.

### `frontend/`
Next.js 16 + React 19 application.

#### Core areas
- `src/app/chat/page.tsx`  
  Main chat UI and backend routing by selected cluster.
- `src/app/landing/`  
  Public landing/marketing content and project narrative.
- `src/app/api/`  
  API routes for auth and conversations.
- `src/components/`  
  UI components (`ChatArea`, `Sidebar`, `Auth`, etc.).
- `src/lib/`  
  Auth, MongoDB, conversations, utility logic.
- `src/middleware.ts`  
  Auth middleware for selected routes.
- `package.json`  
  Project scripts/dependencies.

#### Frontend-to-backend mapping
- Cluster 1 -> `http://127.0.0.1:8000/query`
- Cluster 2 -> `http://127.0.0.1:8001/query2`
- Cluster 3 -> `http://127.0.0.1:8002/rag/answer-verified` (currently mismatched with internal medicine routes)
- Cluster 4 -> `http://127.0.0.1:8003/query4`

### `data division logic/`
- `cluster_division.ipynb`
  - Streams full `miriad/miriad-4.4M` to `miriad_4.4M.jsonl`
  - Filters by specialty list (example: `Neurology`, `Neurosurgery`)
  - Produces filtered files like `miriad_neurology.jsonl` and `miriad_neurology.json`
- `specialties.py`
  - Streams dataset and prints unique specialty values

### `media/`
- `system.gif`  
  Visual flow asset referenced in root documentation.

## 5) Data and Artifacts

### Primary dataset family
- Source: `miriad/miriad-4.4M`
- Common derived artifacts:
  - full JSONL dump (`miriad_4.4M.jsonl`)
  - specialty-specific filtered files (e.g., neurology)
  - JSON/JSONL cluster datasets for backend indexing

### Backend-specific artifact expectations

#### Neurology
- `backend/neurology/data/faiss.index`
- `backend/neurology/data/bm25.pkl`
- `backend/neurology/data/metadata.pkl`
- `backend/neurology/model_weights/...`

#### Cardiology
- input dataset expected by `build_index.py`:
  - `../../data/cardiology/miriad_cardiology.json`
- generated index:
  - `backend/cardiology/vectorstore/faiss_index`
- local models:
  - `backend/cardiology/models/instructor-large`
  - `backend/cardiology/models/hyde-sciFive-cardiology-generator`
  - `backend/cardiology/models/inbedder-roberta-large`
  - `backend/cardiology/models/projector.pt`

#### Internal Medicine
- env-provided KB path:
  - `KB_FILE` (default `data/kb.json`)
- generated embeddings/index metadata in:
  - `storage/embeddings/kb_faiss.index`
  - `storage/embeddings/kb_metadata.json`
  - `storage/embeddings/scope_faiss.index`
  - `storage/embeddings/scope_metadata.json`

#### Primary Care
Expected resource defaults:
- `resources/selected_specialties.jsonl`
- `resources/centroids/specialty_centroids.pkl`
- `resources/faiss/index.faiss`
- `resources/faiss/embeddings.npy`
- `resources/caches/semantic_scholar_cache.json`

## 6) Pipeline Designs (Technical Summary)

### Pipeline 1: Neurosciences (`backend/neurology`)
**Goal:** Neurology/neurosurgery-focused retrieval with multi-stage filtering and reranking.

Flow:
1. Intent/domain check (`DomainChecker`) for chitchat/out-of-scope
2. Query rewrite (Flan-T5 local dir)
3. Dual retrieval:
   - dense (FAISS via BioBERT embeddings)
   - lexical (BM25)
4. RRF fusion
5. SLM gatekeeper filters chunks
6. SLM blueprint generator creates answer blueprint
7. Instruction reranker reorders evidence
8. Phi-2 rephraser generates final response

Key endpoint: `POST /query` on port 8000

### Pipeline 2: Cardiovascular (`backend/cardiology`)
**Goal:** HyDE + instruction-aware retrieval for cardiology QA.

Flow:
1. Domain gate using FAISS similarity (text and vector gate variants)
2. InBEDDER query embedding + projection head
3. HyDE hypothetical doc generation (sciFive generator)
4. Embedding fusion (`alpha` blend)
5. FAISS retrieval from cardiology index
6. Final answer generation via Ollama (`phi:2.7b`)

Key endpoints:
- `POST /query2`
- `POST /query_stages`
- `GET /health`
on port 8001

### Pipeline 3: Internal Medicine (`backend/internal_medicine`)
**Goal:** Retrieval plus answer verification/regeneration for safer outputs.

Flow:
1. Optional domain/scope gate
2. RAG answer generation using retrieved evidence + Ollama
3. Risk routing (sentence/entity risk scoring)
4. Verification:
   - NLI
   - optional judge
   - sentence-level checks
5. Regeneration/reconstruction
6. Transparency/disclaimer layer

Key endpoints:
- `POST /rag`
- `POST /query3`
- `POST /evaluate-stages`
- `POST /reindex`
- `GET /health`
on port 8002

### Pipeline 4: Primary Care & Mental Health (`backend/primarycare`)
**Goal:** Scope-aware retrieval with authority/evidence validation and structured answer generation.

Flow:
1. Specialty scope gate via centroid similarity
2. Dense retrieval of top candidates
3. Phase-1 validation:
   - passage semantic score
   - query-title answerability similarity
4. Phase-2 validation:
   - Semantic Scholar enrichment
   - citation/evidence/risk scoring
   - final ranking
5. Verification level computation
6. Instructor LLM creates answer plan/constraints
7. Answer LLM generates grounded sections (`direct_answer`, `evidence_summary`, `limitations`)

Key endpoints:
- `POST /query4`
- `POST /retrieve`
- `POST /retrieve/detailed`
- `POST /batch-query`
on port 8003 (local script) / configurable in deployment

## 7) API Summary by Service

| Service | Port | Main Endpoint(s) | Purpose |
|---|---:|---|---|
| Neurology | 8000 | `/query` | Hybrid retrieval + SLM gating/blueprint + rerank |
| Cardiology | 8001 | `/query2`, `/query_stages` | HyDE + fusion + cardiology-domain answer |
| Internal Medicine | 8002 | `/rag`, `/query3`, `/evaluate-stages` | RAG + verification/regeneration |
| Primary Care | 8003 (local) | `/query4`, `/retrieve`, `/retrieve/detailed` | Scope gate + evidence authority scoring + structured response |

## 8) Frontend Architecture and User Flow

1. User selects one of four clusters in chat UI.
2. Frontend sends the query payload format expected by that backend.
3. Backend response is normalized for display.
4. If authenticated, conversation is stored in MongoDB.
5. Conversation history can be reopened and replayed.

Auth/data:
- NextAuth with Google + credentials
- MongoDB collections for users and conversations
- API routes for conversation CRUD and message append

## 9) Local Setup and Run Guide (Practical)

### Prerequisites
- Python 3.10+
- Node.js + npm
- MongoDB instance
- Ollama running for services that use it
- (Optional) GPU/CUDA for neurology-heavy stack
- Semantic Scholar API key for primary care pipeline

### Frontend
```bash
cd frontend
npm install
npm run dev
```
App runs on `http://localhost:3000`.

### Neurology backend
```bash
cd backend/neurology
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```
Requires local indices + model weights in expected paths.

### Cardiology backend
```bash
cd backend/cardiology
pip install -r requirements.txt
# Optional (if index not built):
python build_index.py
uvicorn api:app --host 0.0.0.0 --port 8001 --reload
```

### Internal Medicine backend
```bash
cd backend/internal_medicine
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --port 8002 --reload
```

### Primary Care backend
```bash
cd backend/primarycare
pip install -r requirements.txt
python main.py
```
Also requires:
- `S2_API_KEY` (or `MEDIRAG_S2_API_KEY`)
- resources directory artifacts
- missing `medirag.data` package issue resolved in codebase/environment

## 10) Environment Variables (Important)

### Frontend
- `MONGODB_URI`
- `NEXTAUTH_SECRET`
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`

### Neurology
- `HF_TOKEN` (for HuggingFace access where required)

### Cardiology
- `CARDIOLOGY_DOMAIN_MAX_DISTANCE`
- `CARDIOLOGY_DOMAIN_MAX_DISTANCE_TEXT`
- `CARDIOLOGY_OOD_USE_OLLAMA`

### Internal Medicine
- `KB_FILE`
- `EMBEDDINGS_DIR`
- `EMBEDDING_MODEL_NAME`
- `OLLAMA_HOST`
- `OLLAMA_GENERATOR_MODEL`
- `OLLAMA_JUDGE_MODEL`
- `JUDGE_BACKEND`
- `ENABLE_DOMAIN_GATE`, `ENABLE_NER`, `ENABLE_NLI`
- threshold/tuning vars from `app/core/config.py`

### Primary Care
- `S2_API_KEY` (required by startup logic)
- `MEDIRAG_LLM_PROVIDER` (`lmstudio` or `ollama`)
- `MEDIRAG_OLLAMA_BASE_URL`
- `MEDIRAG_OLLAMA_MODEL`
- `MEDIRAG_PASSAGES_PATH`
- `MEDIRAG_CENTROIDS_PATH`
- `MEDIRAG_FAISS_INDEX_PATH`
- `MEDIRAG_EMBEDDINGS_PATH`
- `MEDIRAG_S2_CACHE_PATH`

## 11) Evaluation and Demonstration Assets

- Cardiology staged endpoint: `/query_stages`
- Internal medicine stage-wise endpoint: `/evaluate-stages`
- Primary care detailed retrieval/demo endpoint: `/retrieve/detailed`
- UI demo assets:
  - `backend/internal_medicine/demo.html`
  - `backend/primarycare/demo-v3/animated_demo_v3.html`

Tests/scripts present:
- `backend/internal_medicine/test_stages_api.py`
- `backend/primarycare/test_detailed_demo.py`
- `backend/cardiology/test.txt` (run command + sample query)

## 12) Known Issues and Gaps (Current Repository State)

1. **Cluster 3 frontend endpoint mismatch**
   - Frontend calls `/rag/answer-verified`
   - Internal medicine backend exposes `/rag`, `/query3`, `/evaluate-stages`
   - This likely causes cluster-3 request failures unless adapted.

2. **Primary care missing package modules**
   - Imports expect `medirag.data.faiss_index` and `medirag.data.loaders`
   - Those modules are not present in this checkout.

3. **Large critical artifacts are excluded from git**
   - data files, vector indexes, model weights, caches are mostly local/ignored.
   - fresh clone is not fully runnable without reconstructing resources.

4. **Port references in docs are inconsistent in places**
   - internal medicine docs mention 8000 in examples while runtime command uses 8002.

## 13) Research Contributions Snapshot

- Demonstrates a **modular multi-backend medical RAG architecture**
- Implements **distinct hallucination-control strategies** per domain
- Combines:
  - domain/scope refusal paths
  - dense + lexical fusion retrieval
  - SLM-assisted filtering/planning
  - evidence/authority scoring from metadata
  - post-generation claim verification/regeneration
- Provides practical pathway for comparing heterogeneous RAG safety methods in one product interface

## 14) Suggested Next Improvements

1. Unify endpoint contracts and payload schemas across all 4 backends.
2. Add one repo-level orchestration script (`docker-compose` or unified launcher).
3. Add explicit data preparation docs per pipeline with expected file schemas.
4. Restore/commit non-sensitive scaffold files required for primary care data module.
5. Add benchmark/evaluation report templates for reproducible research results.

## 15) Credits

Developed as a final-year Data Science research project focused on medically grounded, trustworthy QA through specialized RAG architecture.
