"""
Query Intent Classifier
Classifies user queries into different categories:
- CARDIOLOGY: Medical question related to cardiology
- GENERAL_GREETING: Hello, Hi, How are you, etc.
- UNRELATED: Query outside cardiology scope
"""

import numpy as np
import re
from typing import Tuple
from embedder import embed_and_project, l2_normalize


# ============================
# SETTINGS
# ============================
CARDIOLOGY_SIMILARITY_THRESHOLD = 0.65  # Threshold for cardiology classification
SEMANTIC_TOP_K = 5  # Compare against top-K nearest chunks in cardiology KB

GREETING_PATTERNS = [
    r"^(hello|hi|hey|greetings|good\s*(morning|afternoon|evening|night))",
    r"^(how\s*are\s*you|how\s*do\s*you\s*do)",
    r"^(what.*s.*up|sup|yo)",
]

# ============================
# HELPER FUNCTIONS
# ============================
def _is_greeting(query: str) -> bool:
    """Check if query matches common greeting patterns."""
    query_lower = query.lower().strip()
    for pattern in GREETING_PATTERNS:
        if re.match(pattern, query_lower):
            return True
    return False


def _distance_to_similarity(distance: float) -> float:
    """Convert FAISS L2 distance (normalized vectors) to bounded similarity [0,1]."""
    return float(max(0.0, 1.0 - min(distance, 1.0)))


# ============================
# MAIN CLASSIFIER
# ============================
def classify_query(query: str, vectorstore=None) -> Tuple[str, float, dict]:
    """
    Classify query intent.

    Returns:
        Tuple of (intent, confidence_score, metadata)
        
        intent types:
        - "CARDIOLOGY": Medical question related to cardiology
        - "GENERAL_GREETING": Hello, Hi, How are you, etc.
        - "UNRELATED": Query not semantically close to cardiology knowledge base
    """
    metadata = {
        "similarity_score": 0.0,
        "top_k_similarity": [],
        "original_query": query,
    }

    # ===================================
    # Step 1: Check for greetings
    # ===================================
    if _is_greeting(query):
        return ("GENERAL_GREETING", 1.0, metadata)

    # ===================================
    # Step 2: Semantic similarity check with cardiology index
    # ===================================
    if vectorstore is not None:
        try:
            # Embed query
            proj_emb = embed_and_project(query)
            proj_emb = l2_normalize(np.array(proj_emb))

            # Compare with cardiology KB using top-K nearest chunks
            results = vectorstore.similarity_search_with_score_by_vector(
                embedding=proj_emb.tolist(),
                k=SEMANTIC_TOP_K
            )

            if results:
                similarities = [_distance_to_similarity(score) for _, score in results]
                best_similarity = max(similarities)

                metadata["similarity_score"] = best_similarity
                metadata["top_k_similarity"] = similarities

                if best_similarity >= CARDIOLOGY_SIMILARITY_THRESHOLD:
                    return ("CARDIOLOGY", best_similarity, metadata)

                return ("UNRELATED", best_similarity, metadata)

        except Exception as e:
            print(f"⚠️  Vectorstore query failed: {e}. Marking query as unrelated.")
            return ("UNRELATED", 0.0, metadata)

    # ===================================
    # Step 3: No vectorstore available
    # ===================================
    return ("UNRELATED", 0.0, metadata)
