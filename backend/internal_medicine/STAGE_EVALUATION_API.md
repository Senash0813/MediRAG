# Stage-Wise Evaluation API

## Endpoint: `/evaluate-stages`

This endpoint provides a demonstration of how each stage of the RAG pipeline performs and outputs its results. It's designed for evaluation and demonstration purposes without overwhelming detail.

## Request

**Method:** `POST`

**Body:**
```json
{
  "query": "What is the treatment for hypertension?",
  "top_k": 5,
  "temperature": 0.0,
  "verify": true
}
```

## Response Structure

The response shows output from each of the 6 pipeline stages:

### Stage A: Domain/Scope Gate
- **Purpose:** Checks if the query is within the knowledge base scope
- **Output:**
  - Decision: IN_DOMAIN or OUT_OF_DOMAIN
  - Top-1 similarity score
  - Average top-K similarity
  - Cohesion score among retrieved passages

### Stage B: RAG Answer Generation
- **Purpose:** Generates an answer using retrieved documents
- **Output:**
  - Generated answer text
  - Number of retrieved documents
  - Top document similarity scores

### Stage C: Risk Routing
- **Purpose:** Identifies high-risk medical statements requiring verification
- **Output:**
  - Total sentences in answer
  - Number of high-risk sentences
  - Risk threshold used
  - Per-sentence risk scores and entity counts

### Stage D: Verification
- **Purpose:** Verifies claims against evidence using NLI and optional LLM judging
- **Output (Path D0 - Answer-Level):**
  - NLI label and entailment probability
  - Support status (SUPPORTED/UNSUPPORTED/UNCERTAIN)
  - Whether judge was called

- **Output (Path D1 - Sentence-Level):**
  - Counts of verified/regenerated/hallucinated/uncertain sentences
  - Sample verification results showing similarity, NLI, and status

### Stage E: Reconstruction
- **Purpose:** Rebuilds answer with verified/regenerated claims
- **Output:**
  - Number of sentence replacements made
  - Final answer excerpt

### Stage F: Transparency Layer
- **Purpose:** Adds disclaimers for safety and transparency
- **Output:**
  - Whether disclaimer was added
  - Complete final answer with disclaimers

## Example Usage

### Using curl:
```bash
curl -X POST "http://localhost:8000/evaluate-stages" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the side effects of aspirin?",
    "verify": true
  }'
```

### Using Python requests:
```python
import requests

response = requests.post(
    "http://localhost:8000/evaluate-stages",
    json={
        "query": "What are the contraindications for metformin?",
        "verify": True
    }
)

stages = response.json()
print(f"Stage A: {stages['stage_a_domain_gate']['decision']}")
print(f"Stage B: {stages['stage_b_rag']['generated_answer']}")
print(f"Stage C: {stages['stage_c_risk_routing']['high_risk_sentences']} high-risk sentences")
print(f"Stage D: {stages['stage_d_verification']['path']}")
print(f"Stage E: {stages['stage_e_reconstruction']['replacements_made']} replacements")
print(f"Stage F: {stages['stage_f_transparency']['disclaimer_added']}")
```

## Notes

- Set `verify: false` to skip stages C-F and only see RAG generation
- If query is out of domain (Stage A fails), subsequent stages are skipped
- The endpoint demonstrates the pipeline flow without excessive technical detail
- For full technical details, use the `/query3` endpoint with `verify: true`
