# MediRAG Cardiology Pipeline — Complete Documentation

## 📚 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Pipeline Components](#core-pipeline-components)
4. [Data Flow](#data-flow)
5. [API Endpoints](#api-endpoints)
6. [Configuration & Deployment](#configuration--deployment)
7. [Notebooks](#notebooks)
8. [Models & Dependencies](#models--dependencies)
9. [Usage Examples](#usage-examples)
10. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

**MediRAG Cardiology** is an advanced Retrieval-Augmented Generation (RAG) system specifically trained for cardiology Q&A. It combines multiple state-of-the-art embedding techniques, intelligent domain filtering, and generative AI to provide accurate, context-aware cardiology answers.

### Key Features

- **Multi-Embedding Fusion**: Combines InBEDDER (domain-specific) and HyDE (generative) embeddings for superior retrieval
- **Domain Gating**: Ensures queries remain within cardiology scope before retrieval
- **Intelligent Caching**: FAISS vectorstore for fast similarity search
- **Ollama Integration**: Supports quantized Phi model for efficient inference
- **Dual Backends**: Fallback to HuggingFace Transformers if Ollama unavailable
- **Staged Pipeline Visibility**: Debug endpoint shows step-by-step execution

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER QUERY (Frontend)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Server (api.py)                         │
│         - Request parsing & validation                       │
│         - Orchestrates pipeline components                   │
│         - Returns formatted responses                        │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
  ┌──────────────┐ ┌──────────────┐ ┌────────────┐
  │ Intent       │ │ Domain Gate  │ │   Main     │
  │ Detection    │ │ (FAISS)      │ │ Pipeline   │
  │              │ │              │ │            │
  │ • Greeting   │ │ • Text Gate  │ │ 1. Embed   │
  │ • Goodbye    │ │ • Vector Gate│ │ 2. HyDE    │
  │ • Help       │ │              │ │ 3. Fuse    │
  │ • Identity   │ │ Threshold:   │ │ 4. Retrieve│
  │              │ │ 0.22 (default)│ │ 5. Answer  │
  └──────────────┘ └──────────────┘ └────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
  ┌──────────────┐ ┌──────────────┐ ┌────────────────┐
  │ Embedder     │ │ HyDE         │ │ Retriever      │
  │ (embedder.py)│ │ (hyde.py)    │ │ (retriever.py) │
  │              │ │              │ │                │
  │ • InBEDDER   │ │ • Query →    │ │ • FAISS Index  │
  │ • Projection │ │   Phi Gen    │ │ • Top-k Search │
  │   Head       │ │ • Hypothetical│ │ • Scoring      │
  │              │ │   Answers    │ │                │
  └──────────────┘ │ • Instructor │ └────────────────┘
                   │   Embedding  │
                   └──────────────┘
                         │
                         ▼
                   ┌──────────────┐
                   │ Fusion       │
                   │ (fusion.py)  │
                   │              │
                   │ • Blend Emb  │
                   │ • L2 Norm    │
                   │ • Alpha: 0.5 │
                   └──────────────┘
                         │
                         ▼
                   ┌──────────────┐
                   │ LLM Answer   │
                   │ Generation   │
                   │              │
                   │ • Phi Model  │
                   │ • Context    │
                   │ • Formatting │
                   └──────────────┘
                         │
                         ▼
              ┌────────────────────────┐
              │ Formatted Response     │
              │ - Answer               │
              │ - Retrieved Docs       │
              │ - Debug Info (opt)     │
              └────────────────────────┘
```

---

## 🔄 Core Pipeline Components

### 1. **api.py** — FastAPI Server & Request Orchestration

**Purpose**: Exposes REST endpoints and orchestrates the entire RAG pipeline.

**Key Classes**:

```python
class QueryRequest(BaseModel):
    query: str                          # User's cardiology question
    k: Optional[int] = 5                # Number of documents to retrieve
    alpha: Optional[float] = 0.5        # Embedding fusion weight
    domain_max_distance: Optional[float] = None        # Projected vector gate threshold
    domain_max_distance_text: Optional[float] = None   # Text similarity gate threshold
    debug: Optional[bool] = False       # Return debug information

class QueryResponse(BaseModel):
    answer: str                         # Generated cardiology answer
    retrieved_docs: List[str]           # Context documents (optional)
    domain_in_domain: Optional[bool]    # Domain gate result (debug)
    domain_best_score: Optional[float]  # Best similarity score (debug)
    # ... other debug fields
```

**Startup Flow**:
1. Load FAISS vectorstore (cached in memory)
2. Download/load Phi model (HuggingFace or Ollama)
3. Mark `MODEL_READY` and `VECTORSTORE_READY` as True
4. Ready to accept requests

**Request Flow** (`/query2` endpoint):

```python
@app.post("/query2", response_model=QueryResponse)
def run_query(req: QueryRequest):
    # 1. Check if backend ready
    if not MODEL_READY or not VECTORSTORE_READY:
        raise HTTPException(status_code=503, ...)
    
    # 2. Detect general intent (greeting, goodbye, help, identity)
    intent = detect_general_intent(req.query)
    if intent is not None:
        return QueryResponse(answer=generate_general_intent_answer(intent), ...)
    
    # 3. Load vectorstore (cached)
    vectorstore = load_vectorstore()
    
    # 4. Domain gate (text-based, using native FAISS embeddings)
    in_domain_text, best_score_text = _domain_gate_text(vectorstore, req.query, ...)
    
    # 5. Domain gate (vector-based, using projected embeddings)
    proj_emb = embed_and_project(req.query)
    in_domain_vec, best_score = _domain_gate(vectorstore, proj_emb, ...)
    
    # 6. If out of domain, return canned response
    if not in_domain:
        answer = generate_out_of_domain_answer(req.query)
        return QueryResponse(answer=answer, ...)
    
    # 7. Generate hypothetical documents (HyDE)
    hyde_emb, hyde_docs = generate_hypothetical_docs(req.query)
    
    # 8. Fuse embeddings: alpha * proj_emb + (1-alpha) * hyde_emb
    final_emb = fuse_embeddings(proj_emb, hyde_emb, alpha=0.5)
    
    # 9. Retrieve top-k documents from FAISS
    docs_and_scores = vectorstore.similarity_search_with_score_by_vector(final_emb, k=5)
    
    # 10. Generate final answer using Phi model
    answer = generate_final_answer(req.query, retrieved_docs)
    
    return QueryResponse(answer=answer, retrieved_docs=retrieved_docs_text)
```

**Default Configuration**:
- `k`: 5 documents
- `alpha`: 0.5 (equal weight to both embeddings)
- `domain_max_distance`: 0.22 (from env var `CARDIOLOGY_DOMAIN_MAX_DISTANCE`)
- `domain_max_distance_text`: 0.22 (from env var `CARDIOLOGY_DOMAIN_MAX_DISTANCE_TEXT`)
- `debug`: False (no debug info)

---

### 2. **embedder.py** — InBEDDER Encoder with Projection Head

**Purpose**: Encode cardiology questions into domain-specific 768-dimensional vectors.

**Components**:

```python
# Models loaded on startup
inbed_model = SentenceTransformer("models/inbedder-roberta-large")
projector = ProjectionLayer(in_dim=1024, out_dim=768)
projector.load_state_dict(torch.load("models/projector.pt"))

class ProjectionLayer(nn.Module):
    def __init__(self, in_dim=1024, out_dim=768):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
```

**Processing Pipeline**:

```
Input Query
    ↓
"Cluster this cardiology question semantically: {query}"
    ↓
InBEDDER Encoding (1024-d)
    ↓
L2 Normalization
    ↓
Projection Head (1024 → 768)
    ↓
Output: 768-d normalized vector
```

**Key Function**:

```python
def embed_and_project(query: str) -> np.ndarray:
    """
    Encodes a cardiology question using InBEDDER,
    applies projection head, and returns a 768-d vector.
    """
    instruction = "Cluster this cardiology question semantically:"
    text = f"{instruction}\n{query}"
    
    # Step 1: InBEDDER encoding (1024-d)
    emb = inbed_model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=False
    ).astype(np.float32)
    
    # Step 2: L2 normalize
    emb = l2_normalize(emb)
    
    # Step 3: Project to 768-d space
    with torch.no_grad():
        x = torch.from_numpy(emb).unsqueeze(0).to(device)
        projected = projector(x).squeeze(0)
    
    return projected.cpu().numpy()
```

**Why InBEDDER?**
- **Domain-Specific**: Fine-tuned on biomedical/cardiology literature
- **RoBERTa Foundation**: Strong understanding of medical terminology
- **Projection Head**: Maps to FAISS index space (768-d)

---

### 3. **hyde.py** — Hypothetical Document Embeddings (HyDE)

**Purpose**: Generate hypothetical cardiology answers and encode them for better retrieval coverage.

**Models Used**:

```python
# HyDE generator (SciFive fine-tuned for cardiology)
hyde_model = AutoModelForSeq2SeqLM.from_pretrained(
    "models/hyde-sciFive-cardiology-generator"
)

# Instructor embeddings for hypothetical docs
embeddings_model = HuggingFaceEmbeddings(
    model_name="models/instructor-large"
)
```

**Processing Pipeline**:

```
User Query: "How does sevoflurane postconditioning protect the myocardium?"
    ↓
Phi-2 Prompt: "Question: {query}\nParagraph:"
    ↓
Generate 4 Hypothetical Cardiology Answers
    - Hypothesis 1: "Sevoflurane postconditioning activates PKC signaling..."
    - Hypothesis 2: "By reducing reactive oxygen species generation..."
    - Hypothesis 3: "Through KATP channel activation and mitochondrial..."
    - Hypothesis 4: "Postconditioning provides protection via..."
    ↓
Embed Each Hypothesis with Instructor (768-d)
    ↓
L2 Normalize Each Embedding
    ↓
Mean Pool → Single 768-d Vector
    ↓
Output: Fused Hypothetical Embedding
```

**Key Function**:

```python
def generate_hypothetical_docs(
    query: str,
    num_return_sequences: int = 4,
    max_new_tokens: int = 300
):
    """
    HyDE Pipeline:
    Query → Hypothetical answers → Instructor embeddings
    → mean pool → L2 normalize (768-d)
    """
    
    # Step 1: Create generation prompt
    prompt = f"Question: {query}\nParagraph:"
    
    # Step 2: Tokenize
    inputs = hyde_tokenizer(prompt, return_tensors="pt").to(device)
    
    # Step 3: Generate 4 hypothetical answers
    outputs = hyde_model.generate(
        **inputs,
        do_sample=True,
        num_return_sequences=4,
        repetition_penalty=1.2,
        temperature=0.9,
        top_p=0.92,
        top_k=50,
        max_new_tokens=300
    )
    
    # Step 4: Decode hypothetical answers
    replies = [hyde_tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
    
    # Step 5: Embed hypothetical docs with Instructor (768-d)
    embeddings = embeddings_model.embed_documents(replies)  # 4x768
    
    # Step 6: L2 normalize each embedding (row-wise)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    # Step 7: Mean pool + L2 normalize
    avg_embedding = l2_normalize(np.mean(embeddings, axis=0))  # 768-d
    
    return avg_embedding, replies
```

**Why HyDE?**
- **Generative Approach**: Creates plausible answers, not just paraphrases
- **Coverage**: Better retrieval of documents with similar meaning but different wording
- **Multi-Perspective**: 4 hypothetical answers capture different angles of the question

---

### 4. **fusion.py** — Embedding Fusion

**Purpose**: Blend InBEDDER projections and HyDE embeddings for optimal retrieval.

**Function**:

```python
def fuse_embeddings(proj_query, hyde_query, alpha=0.5):
    """
    Linear combination of two embeddings with L2 normalization.
    
    Args:
        proj_query: 768-d InBEDDER projected embedding
        hyde_query: 768-d HyDE embedding (mean-pooled)
        alpha: Weight for proj_query (default 0.5)
    
    Returns:
        768-d fused and normalized embedding
    """
    # Weighted sum
    fused = alpha * proj_query + (1 - alpha) * hyde_query
    
    # L2 normalize for FAISS compatibility
    return l2_normalize(fused)
```

**Why Fusion?**
- **Complementary Strengths**: InBEDDER captures domain semantics, HyDE covers paraphrases
- **Tunable Balance**: `alpha=0.5` equally weights both approaches (adjustable via API)
- **Normalized Space**: Ensures FAISS L2 distance calculations are consistent

**Example**:
```
If alpha = 0.5:
  fused = 0.5 * InBEDDER_vec + 0.5 * HyDE_vec

If alpha = 0.7 (more InBEDDER):
  fused = 0.7 * InBEDDER_vec + 0.3 * HyDE_vec

If alpha = 0.3 (more HyDE):
  fused = 0.3 * InBEDDER_vec + 0.7 * HyDE_vec
```

---

### 5. **retriever.py** — FAISS Vector Search

**Purpose**: Load vectorstore and retrieve top-k similar documents.

**Components**:

```python
@lru_cache(maxsize=1)
def load_vectorstore() -> FAISS:
    """
    Load FAISS index using SAME embedding model used during indexing.
    Cached in memory for fast repeated access.
    """
    embeddings_model = HuggingFaceEmbeddings(
        model_name="models/instructor-large"
    )
    
    vectorstore = FAISS.load_local(
        "vectorstore/faiss_index",
        embeddings_model,
        allow_dangerous_deserialization=True
    )
    
    return vectorstore

def retrieve(
    vectorstore: FAISS,
    query_embedding: np.ndarray,
    k: int = 5,
    return_scores: bool = False
):
    """
    Perform similarity search using PRE-COMPUTED embedding.
    
    CRITICAL: query_embedding must be in SAME vector space as index (768-d).
    No re-embedding happens here.
    """
    query_embedding = l2_normalize(query_embedding).astype(np.float32)
    
    if return_scores:
        results = vectorstore.similarity_search_with_score_by_vector(
            embedding=query_embedding.tolist(),
            k=k
        )
        return results  # List[(Document, float)]
    else:
        results = vectorstore.similarity_search_by_vector(
            embedding=query_embedding.tolist(),
            k=k
        )
        return results  # List[Document]
```

**FAISS Index Details**:
- **Format**: Flat L2 distance search
- **Dimension**: 768-d (matches Instructor embedding size)
- **Location**: `vectorstore/faiss_index`
- **Metric**: L2 distance (lower = more similar)
- **Cache**: Loaded once, reused across requests

**Example Retrieval**:
```
Query: "How does sevoflurane postconditioning protect myocardium?"

Fused Embedding (768-d) → FAISS → Top-5 Results:

1. distance=0.18 | "Sevoflurane activates KATP channels..."
2. distance=0.21 | "Preconditioning protects via PKC signaling..."
3. distance=0.24 | "Ischemic postconditioning reduces..."
4. distance=0.27 | "ROS-mediated injury in myocardium..."
5. distance=0.30 | "Mitochondrial dysfunction during..."
```

---

### 6. **main.py** — Intent Detection & Answer Generation

**Purpose**: Detect user intent and generate final answers using Phi model.

**Intent Detection**:

```python
def detect_general_intent(user_text: str) -> str | None:
    """
    Detects short, non-domain queries that don't need full RAG pipeline.
    Returns: "greeting" | "goodbye" | "identity" | "help" | None
    """
    patterns = {
        "greeting": r"^\s*(?:hi|hello|hey|thanks|thank you|how are you).*$",
        "goodbye": r"^\s*(?:bye|goodbye|see you|later).*$",
        "identity": r"^\s*(?:who are you|what are you|what is medirag).*$",
        "help": r"^\s*(?:help|what can you do|how does this work).*$"
    }
    
    # Match against patterns...
    if match_found:
        return intent_name
    return None
```

**Canned Responses**:

```python
def generate_greeting_answer() -> str:
    return (
        "Hi — I'm MediRAG Cardiology. Ask me a cardiology question..."
    )

def generate_general_intent_answer(intent: str) -> str:
    if intent == "greeting":
        return generate_greeting_answer()
    elif intent == "goodbye":
        return "Glad to help. If you have another cardiology question, just send it."
    elif intent == "identity":
        return "I'm MediRAG Cardiology — a cardiology-focused Q&A assistant..."
    elif intent == "help":
        return "Ask a cardiology question in one message..."
```

**Answer Generation**:

```python
def generate_final_answer(query: str, retrieved_docs: List[Document]) -> str:
    """
    Uses Phi model (via Ollama or HuggingFace) to generate concise answer.
    
    Workflow:
    1. Prepare context from retrieved documents
    2. Build system prompt with strict formatting rules
    3. Call Phi model
    4. Strip follow-up content/formatting issues
    5. Return clean answer
    """
    
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    prompt = f"""You are MediRAG Cardiology, a cardiology Q&A assistant.

Your task is to answer the user's question using ONLY the provided context.

STRICT OUTPUT RULES:
- Write ONLY one clear answer paragraph (maximum 3-5 sentences).
- Do NOT write follow-up questions.
- Do NOT write follow-up exercises.
- Do NOT write discussion sections.
- Do NOT write "Answer:" or any other header/prefix.

Context:
{context}

Question:
{query}

Respond with ONLY the answer paragraph (no prefix, no follow-up):
"""
    
    # Generate using Ollama or HuggingFace
    raw_answer = _ollama_generate(prompt)
    
    # Clean up any remaining formatting issues
    clean_answer = _strip_followup_content(raw_answer)
    
    return clean_answer
```

**Ollama Integration**:

```python
def _get_phi_pipeline():
    """Initialize Phi model using Ollama (if available) or HuggingFace."""
    
    # Check if Ollama is running
    _ollama_available = _check_ollama_available()
    
    if _ollama_available:
        # Use quantized Phi 2.7b from Ollama
        _use_ollama = True
        return True  # Success without loading pipeline
    else:
        # Fall back to HuggingFace Phi-2
        if torch.cuda.is_available():
            device = 0  # GPU
            dtype = torch.float16
        else:
            device = -1  # CPU
            dtype = torch.float32
        
        _phi_pipeline = hf_pipeline(
            "text-generation",
            model="microsoft/phi-2",
            dtype=dtype,
            device=device,
            trust_remote_code=True
        )
        return _phi_pipeline

def _ollama_generate(prompt: str) -> str:
    """Generate using Ollama API."""
    payload = {
        "model": "phi:2.7b",
        "prompt": prompt,
        "stream": False,
        "temperature": 0.3,
        "top_p": 0.9,
        "num_predict": 300
    }
    
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=120
    )
    
    result = response.json()
    return result.get("response", "").strip()
```

**Domain Gating**:

```python
def _domain_gate_text(vectorstore, query: str, max_distance: float) -> tuple[bool, float]:
    """
    Check if query is within cardiology domain using native FAISS embeddings.
    
    Returns: (is_in_domain, best_similarity_score)
    """
    # Get best matching document using FAISS Instructor embeddings
    res = vectorstore.similarity_search_with_score(query=query, k=1)
    
    best_score = res[0][1]  # FAISS L2 distance
    
    # If score below threshold, consider in-domain
    return (best_score <= max_distance, best_score)
```

---

## 📊 Data Flow

### Complete Request Flow

```
1. FRONTEND SENDS QUERY
   POST /query2
   {
     "query": "How is atrial fibrillation managed?",
     "k": 5,
     "alpha": 0.5,
     "debug": false
   }

2. API SERVER (api.py)
   ├─ Parse QueryRequest
   ├─ Check startup status (MODEL_READY, VECTORSTORE_READY)
   └─ Route to pipeline

3. INTENT DETECTION
   ├─ Check if greeting/goodbye/help/identity
   └─ If yes → Return canned response (EARLY EXIT)

4. DOMAIN GATE (Text-Based)
   ├─ Query: "How is atrial fibrillation managed?"
   ├─ FAISS native embedding (Instructor 768-d)
   ├─ Best similarity score: 0.18
   ├─ Threshold: 0.22 (default)
   └─ Decision: IN_DOMAIN (0.18 < 0.22)

5. EMBEDDING STAGE (InBEDDER)
   ├─ Input: "Cluster this cardiology question semantically: How is atrial fibrillation managed?"
   ├─ InBEDDER Encoding: 1024-d vector
   ├─ L2 Normalize: 1024-d
   ├─ Projection Head: 1024-d → 768-d
   └─ Output: proj_emb (768-d normalized)

6. HyDE STAGE (Hypothetical Documents)
   ├─ Prompt: "Question: How is atrial fibrillation managed?\nParagraph:"
   ├─ Phi-2 Generate 4 hypothetical answers
   │  ├─ "Atrial fibrillation is managed with rate/rhythm control..."
   │  ├─ "Initial treatment involves anticoagulation to prevent..."
   │  ├─ "Catheter ablation is used for rhythm restoration..."
   │  └─ "Pharmacologic management includes beta-blockers..."
   ├─ Embed each with Instructor (768-d each)
   ├─ L2 Normalize
   ├─ Mean Pool
   └─ Output: hyde_emb (768-d normalized)

7. FUSION STAGE
   ├─ fused = 0.5 * proj_emb + 0.5 * hyde_emb
   ├─ L2 Normalize
   └─ Output: final_emb (768-d normalized)

8. RETRIEVAL STAGE (FAISS)
   ├─ Query: final_emb (768-d)
   ├─ Search k=5 in FAISS index
   └─ Retrieved docs with scores:
      ├─ doc1: "AF rate control strategies..." (distance: 0.15)
      ├─ doc2: "Anticoagulation in AF..." (distance: 0.17)
      ├─ doc3: "Ablation procedures..." (distance: 0.19)
      ├─ doc4: "Drug-induced arrhythmias..." (distance: 0.22)
      └─ doc5: "Monitoring in AF patients..." (distance: 0.24)

9. ANSWER GENERATION (Phi Model)
   ├─ Build context from doc1-doc5
   ├─ Create system prompt with retrieved context
   ├─ Call Phi (via Ollama or HuggingFace)
   ├─ Raw output: "Answer: Atrial fibrillation management involves... Follow-up: What about..."
   ├─ Strip follow-up content
   └─ Final answer: "Atrial fibrillation management involves rate/rhythm control, anticoagulation, and potentially catheter ablation depending on patient factors and symptom severity."

10. RESPONSE SENT TO FRONTEND
    {
      "answer": "Atrial fibrillation management involves...",
      "retrieved_docs": ["AF rate control strategies...", "Anticoagulation in AF...", ...]
    }
```

---

## 🔌 API Endpoints

### `/query2` — Main Query Endpoint

**Request**:
```bash
POST /query2
Content-Type: application/json

{
  "query": "How is atrial fibrillation managed?",
  "k": 5,
  "alpha": 0.5,
  "domain_max_distance": 0.35,
  "domain_max_distance_text": 0.35,
  "debug": false
}
```

**Response**:
```json
{
  "answer": "Atrial fibrillation management involves rate/rhythm control, anticoagulation for stroke prevention, and potentially catheter ablation...",
  "retrieved_docs": [
    "AF rate control strategies using beta-blockers and calcium channel blockers...",
    "Anticoagulation in atrial fibrillation reduces stroke risk...",
    "Catheter ablation for rhythm restoration in AF..."
  ],
  "domain_in_domain": null,
  "domain_best_score": null
}
```

**With `debug=true`**:
```json
{
  "answer": "...",
  "retrieved_docs": [...],
  "domain_in_domain": true,
  "domain_best_score": 0.18,
  "domain_max_distance": 0.22,
  "domain_best_score_text": 0.18,
  "domain_max_distance_text": 0.22
}
```

### `/query_stages` — Staged Pipeline (Debug)

**Request**:
```bash
POST /query_stages
Content-Type: application/json

{
  "query": "How is atrial fibrillation managed?"
}
```

**Response** (detailed stages):
```json
{
  "query": "How is atrial fibrillation managed?",
  "stages": [
    {
      "stage_number": 1,
      "stage_name": "Input Query",
      "description": "User's input question to the medical RAG system",
      "data": {
        "query": "How is atrial fibrillation managed?",
        "query_length": 38,
        "query_word_count": 6
      }
    },
    {
      "stage_number": 2,
      "stage_name": "Domain Gate",
      "description": "Determines if the query is within cardiology domain",
      "data": {
        "is_in_domain": true,
        "best_similarity_score": 0.18,
        "threshold": 0.22,
        "decision": "IN_DOMAIN"
      }
    },
    {
      "stage_number": 3,
      "stage_name": "inBEDDER + Projection Head",
      "description": "Query embedding using InBEDDER model with projection",
      "data": {
        "embedding_dimension": 768,
        "embedding_norm": 1.0,
        "embedding_sample": [0.001, 0.002, ...],
        "model": "InBEDDER-RoBERTa-Large"
      }
    },
    {
      "stage_number": 4,
      "stage_name": "HyDE",
      "description": "Generates hypothetical cardiology answers",
      "data": {
        "num_hypothetical_answers": 4,
        "hypothetical_answers": [
          "AF management involves rate/rhythm control...",
          "Anticoagulation is critical for stroke prevention...",
          "Catheter ablation provides rhythm restoration...",
          "Pharmacologic agents include beta-blockers..."
        ]
      }
    },
    {
      "stage_number": 5,
      "stage_name": "Embedding Fusion",
      "description": "Blends InBEDDER and HyDE embeddings",
      "data": {
        "alpha": 0.5,
        "final_embedding_norm": 1.0,
        "description": "Fused 0.5 * InBEDDER + 0.5 * HyDE"
      }
    },
    {
      "stage_number": 6,
      "stage_name": "FAISS Retrieval",
      "description": "Retrieves top-k similar documents",
      "data": {
        "k_retrieved": 5,
        "documents": [
          {
            "rank": 1,
            "distance": 0.15,
            "content": "AF rate control strategies..."
          }
        ]
      }
    },
    {
      "stage_number": 7,
      "stage_name": "Answer Generation",
      "description": "Generates final answer using Phi model",
      "data": {
        "model": "microsoft/phi-2 (via Ollama or HuggingFace)",
        "temperature": 0.3,
        "max_tokens": 300
      }
    }
  ],
  "final_answer": "Atrial fibrillation management involves...",
  "is_in_domain": true
}
```

### `/health` — Health Check

```bash
GET /health

{
  "status": "ready",
  "model": "microsoft/phi-2",
  "model_ready": true,
  "vectorstore_ready": true
}
```

### `/` — Root Info

```bash
GET /

{
  "status": "running",
  "message": "MediRAG Cardiology API is running",
  "model": "microsoft/phi-2",
  "model_ready": true,
  "vectorstore_ready": true,
  "endpoints": ["/health", "/query2", "/query_stages", "/docs"]
}
```

---

## ⚙️ Configuration & Deployment

### Environment Variables

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434              # Ollama endpoint (docker-compose: http://ollama:11434)
OLLAMA_TIMEOUT_S=120                                # Request timeout

# Domain Gating
CARDIOLOGY_DOMAIN_MAX_DISTANCE=0.22                 # Projected vector gate threshold
CARDIOLOGY_DOMAIN_MAX_DISTANCE_TEXT=0.22           # Text embedding gate threshold

# Out-of-Domain Handling
CARDIOLOGY_OOD_USE_OLLAMA=0                        # Use Phi to generate OOD responses (0=canned response)
```

### Docker Deployment

**Unified Dockerfile** (includes all dependencies):
```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.10 python3-pip

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Docker Compose** (with Ollama sidecar):
```yaml
version: '3'

services:
  cardiology-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      OLLAMA_BASE_URL: http://ollama:11434
      CARDIOLOGY_DOMAIN_MAX_DISTANCE: "0.22"
    depends_on:
      - ollama

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama:/root/.ollama
    
volumes:
  ollama:
```

### Model Download

Pre-download models to avoid startup delays:

```bash
# Download InBEDDER
python models_download/download_inbedder_model.py

# Download Instructor
python models_download/download_instructor_model.py

# Download SciFive (HyDE generator)
python models_download/download_sciFive_model.py

# Pull Phi model in Ollama
ollama pull phi:2.7b
```

---

## 📔 Notebooks

### 1. `build_index_final.ipynb`

**Purpose**: Build FAISS vectorstore from raw cardiology data.

**Steps**:
1. Load cardiology JSON data (`data/cardiology/miriad_cardiology.json`)
2. Prepare documents with instruction template
3. Initialize Instructor embedding model
4. Create FAISS index from documents
5. Save index to `vectorstore/faiss_index`

**Output**: Searchable vectorstore with ~10,000 documents

---

### 2. `hyde_data_prep_final.ipynb`

**Purpose**: Prepare training data for HyDE model fine-tuning.

**Steps**:
1. Load cardiology Q&A pairs
2. Split into train/val sets
3. Create generation dataset (Question → Hypothetical Answer)
4. Format for SeqSeq training

**Output**: Tokenized dataset ready for SciFive fine-tuning

---

### 3. `hyde_finetuning_final.ipynb`

**Purpose**: Fine-tune SciFive on cardiology data for hypothesis generation.

**Steps**:
1. Load pre-trained SciFive model
2. Initialize training setup (optimizer, loss)
3. Train on cardiology Q&A pairs
4. Validate on held-out set
5. Save best checkpoint

**Output**: Cardiology-specific HyDE generator model

---

### 4. `medirag_cardiology_full_colab_pipeline_flan_t5.ipynb`

**Purpose**: End-to-end pipeline demonstration (legacy Flan-T5 variant).

**Steps**:
1. Load sample cardiology questions
2. Run through complete RAG pipeline
3. Visualize embeddings and retrieval results
4. Generate answers

---

### 5. `medirag_pipeline_improved.ipynb`

**Purpose**: Improved pipeline with ablation analysis.

**Steps**:
1. Compare embedding methods (InBEDDER vs Instructor vs BERT)
2. Evaluate HyDE vs non-HyDE
3. Measure retrieval quality (RAGAS scores)
4. Optimize fusion parameters

---

## 🧠 Models & Dependencies

### Core Models

| Model | Purpose | Dimension | Source |
|-------|---------|-----------|--------|
| **InBEDDER-RoBERTa-Large** | Domain-specific embedding | 1024 → 768 | HuggingFace (biomedical) |
| **Instructor-Large** | FAISS index embedding | 768 | HuggingFace (general + instruction) |
| **SciFive-Cardiology** | HyDE hypothesis generation | Seq2Seq | Custom fine-tuned |
| **Phi-2** | Answer generation | 2.7B params | Microsoft (via HuggingFace/Ollama) |
| **Projector Head** | InBEDDER → 768-d mapping | 1024 → 768 | Trained projection layer |

### Python Dependencies

```
# Core
fastapi
pydantic
uvicorn

# ML/Embeddings
torch
transformers
sentence-transformers
langchain-community
langchain-huggingface

# Vector Search
faiss-cpu  # or faiss-gpu

# Model Loading
huggingface-hub

# Utilities
numpy
requests
python-dotenv
```

---

## 💡 Usage Examples

### Example 1: Basic Query

```python
import requests

response = requests.post(
    "http://localhost:8000/query2",
    json={
        "query": "What are the causes of atrial fibrillation?",
        "k": 5,
        "alpha": 0.5
    }
)

answer = response.json()["answer"]
print(answer)
# Output: "Atrial fibrillation can be caused by hypertension, structural heart disease, thyrotoxicosis, excessive alcohol use, obstructive sleep apnea, acute pulmonary embolism, and other factors including genetic predisposition..."
```

### Example 2: With Debug Information

```python
response = requests.post(
    "http://localhost:8000/query2",
    json={
        "query": "How does beta-blocker work in heart failure?",
        "k": 5,
        "alpha": 0.5,
        "debug": True
    }
)

data = response.json()
print(f"Answer: {data['answer']}")
print(f"Domain Gate Decision: {data['domain_in_domain']}")
print(f"Best Similarity Score: {data['domain_best_score']:.4f}")
print(f"Retrieved {len(data['retrieved_docs'])} documents")
```

### Example 3: Staged Pipeline

```python
response = requests.post(
    "http://localhost:8000/query_stages",
    json={"query": "What is the prognosis of dilated cardiomyopathy?"}
)

stages = response.json()["stages"]
for stage in stages:
    print(f"Stage {stage['stage_number']}: {stage['stage_name']}")
    print(f"  {stage['description']}")
    print(f"  Data: {stage['data']}")
```

### Example 4: Custom Alpha (Embedding Fusion Weight)

```python
# More InBEDDER (domain-specific)
response_inbedder = requests.post(
    "http://localhost:8000/query2",
    json={
        "query": "Management of arrhythmogenic cardiomyopathy",
        "alpha": 0.8  # 80% InBEDDER, 20% HyDE
    }
)

# Balanced
response_balanced = requests.post(
    "http://localhost:8000/query2",
    json={
        "query": "Management of arrhythmogenic cardiomyopathy",
        "alpha": 0.5  # 50/50 split
    }
)

# More HyDE (paraphrase coverage)
response_hyde = requests.post(
    "http://localhost:8000/query2",
    json={
        "query": "Management of arrhythmogenic cardiomyopathy",
        "alpha": 0.2  # 20% InBEDDER, 80% HyDE
    }
)
```

---

## 🔧 Troubleshooting

### Issue: "Backend is still loading model/vectorstore"

**Cause**: Request sent before startup completion (typically 30-60 seconds).

**Solution**:
```bash
# Check health endpoint
curl http://localhost:8000/health

# Wait for:
# "status": "ready"
# "model_ready": true
# "vectorstore_ready": true

# Then retry query
```

### Issue: "❌ Ollama returned empty output"

**Cause**: Ollama service not running or Phi model not available.

**Solution**:
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# If empty, pull Phi model
ollama pull phi:2.7b

# Verify model available
ollama list | grep phi
```

### Issue: FAISS Dimension Mismatch

**Cause**: Query embedding dimension ≠ index dimension (should be 768).

**Solution**:
```python
# Verify embedding dimension
from embedder import embed_and_project
query = "Sample question"
emb = embed_and_project(query)
print(f"Embedding shape: {emb.shape}")  # Should be (768,)

# Verify FAISS index dimension
from retriever import load_vectorstore
vs = load_vectorstore()
print(f"Index dimension: {vs.index.d}")  # Should be 768
```

### Issue: Low Retrieval Quality

**Solution 1**: Adjust alpha parameter
```python
# Try more HyDE if InBEDDER too rigid
requests.post("http://localhost:8000/query2", json={
    "query": "...",
    "alpha": 0.3  # Increase HyDE contribution
})
```

**Solution 2**: Adjust domain thresholds
```python
# Relax domain gate if excluding valid queries
requests.post("http://localhost:8000/query2", json={
    "query": "...",
    "domain_max_distance": 0.25,  # Increase threshold
    "domain_max_distance_text": 0.25
})
```

**Solution 3**: Rebuild vectorstore if outdated
```bash
cd backend/cardiology
python build_index.py
```

### Issue: Slow Answer Generation

**Cause**: Using CPU-based HuggingFace Phi instead of GPU/Ollama.

**Solution**:
```bash
# Use Ollama for quantized inference (faster)
# Set up Ollama service with phi:2.7b model

# Verify GPU availability in Python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))
```

---

## 📈 Performance Metrics

### Query Latency (Typical)

| Stage | Duration (ms) |
|-------|---------------|
| Intent Detection | 1-2 |
| Domain Gate | 20-30 |
| InBEDDER Embedding | 50-100 |
| HyDE Generation | 2000-4000 |
| Embedding Fusion | 1-2 |
| FAISS Retrieval | 5-10 |
| Answer Generation | 3000-8000 |
| **Total** | **5100-12140** |

### Memory Usage

- **Vectorstore (FAISS)**: ~500 MB (10k documents × 768-d)
- **Models in Memory**:
  - InBEDDER: ~500 MB
  - Projector Head: ~5 MB
  - Instructor: ~500 MB
  - SciFive (HyDE): ~1.5 GB
  - Phi-2 (HuggingFace): ~5 GB (FP32) or ~2.5 GB (FP16)
- **Total Ollama Mode**: ~2.5 GB (quantized Phi)
- **Total HuggingFace Mode**: ~8 GB

---

## 🚀 Quick Start

```bash
# 1. Clone repo and navigate
cd backend/cardiology

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download models (optional if pre-downloaded)
python models_download/download_inbedder_model.py
python models_download/download_instructor_model.py
python models_download/download_sciFive_model.py

# 4. Build/verify FAISS index (if not already built)
python build_index.py

# 5. Start Ollama (optional but recommended for speed)
# In separate terminal:
ollama serve

# 6. In another terminal, pull Phi model
ollama pull phi:2.7b

# 7. Start API server
uvicorn api:app --host 0.0.0.0 --port 8000

# 8. Test endpoint
curl -X POST http://localhost:8000/query2 \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How is atrial fibrillation managed?",
    "k": 5,
    "alpha": 0.5
  }'
```

---

## 📝 Summary

**MediRAG Cardiology** is a sophisticated RAG system that:

1. **Embeds queries** using domain-specific InBEDDER + projection head
2. **Generates hypothetical answers** using fine-tuned SciFive (HyDE)
3. **Fuses embeddings** with tunable alpha parameter
4. **Retrieves documents** from FAISS vectorstore (768-d space)
5. **Generates final answers** using Phi-2 LLM
6. **Validates domain relevance** with dual-gate system

The architecture balances **accuracy** (domain-specific embeddings), **coverage** (HyDE paraphrases), **speed** (Ollama quantization), and **flexibility** (adjustable parameters).

