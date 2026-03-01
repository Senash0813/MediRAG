"""
Query Intent Classifier
Classifies user queries into different categories:
- CARDIOLOGY: Medical question related to cardiology
- GENERAL_GREETING: Hello, Hi, How are you, etc.
- GENERAL_CHAT: General knowledge questions
- OTHER_MEDICAL: Medical questions outside cardiology
- UNCLEAR: Could be cardiology but low confidence
"""

import numpy as np
import re
from typing import Tuple
from embedder import embed_and_project, l2_normalize


# ============================
# SETTINGS
# ============================
CARDIOLOGY_SIMILARITY_THRESHOLD = 0.65  # Threshold for cardiology classification
GREETING_PATTERNS = [
    r"^(hello|hi|hey|greetings|good\s*(morning|afternoon|evening|night))",
    r"^(how\s*are\s*you|how\s*do\s*you\s*do)",
    r"^(what.*s.*up|sup|yo)",
]

OTHER_MEDICAL_KEYWORDS = [
    "neurology", "neurology", "neuritis", "brain", "nervous system",
    "oncology", "cancer", "tumor", "chemotherapy",
    "pulmonology", "lung", "respiratory", "asthma", "pneumonia",
    "gastroenterology", "stomach", "intestine", "liver", "pancreas",
    "rheumatology", "arthritis", "autoimmune",
    "pediatrics", "child", "infant", "pediatric",
    "psychiatry", "mental", "depression", "anxiety",
    "dermatology", "skin", "dermatitis", "psoriasis",
    "nephrology", "kidney", "renal", "dialysis",
    "endocrinology", "diabetes", "thyroid", "hormone",
]

CARDIOLOGY_KEYWORDS = [
    "heart", "cardiac", "coronary", "artery", "ventricle", "atrium",
    "myocardial", "infarction", "mi", "stent", "bypass", "pacemaker",
    "arrhythmia", "fibrillation", "hypertension", "heart failure",
    "angina", "echocardiogram", "ekg", "ecg", "valve", "aorta",
    "cardiology", "cardiologist", "cardiovascular",
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


def _has_other_medical_keywords(query: str) -> bool:
    """Check if query mentions other medical domains."""
    query_lower = query.lower()
    for keyword in OTHER_MEDICAL_KEYWORDS:
        if keyword in query_lower:
            return True
    return False


def _has_cardiology_keywords(query: str) -> bool:
    """Check if query contains cardiology-specific keywords."""
    query_lower = query.lower()
    for keyword in CARDIOLOGY_KEYWORDS:
        if keyword in query_lower:
            return True
    return False


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
        - "GENERAL_CHAT": General knowledge questions
        - "OTHER_MEDICAL": Medical questions outside cardiology
        - "UNCLEAR": Low confidence - could be cardiology but below threshold
    """
    metadata = {
        "has_cardiology_keywords": False,
        "has_other_medical_keywords": False,
        "similarity_score": 0.0,
        "original_query": query,
    }

    # ===================================
    # Step 1: Check for greetings
    # ===================================
    if _is_greeting(query):
        return ("GENERAL_GREETING", 1.0, metadata)

    # ===================================
    # Step 2: Check for other medical domains
    # ===================================
    if _has_other_medical_keywords(query):
        metadata["has_other_medical_keywords"] = True
        return ("OTHER_MEDICAL", 0.8, metadata)

    # ===================================
    # Step 3: Semantic similarity check with cardiology index
    # ===================================
    if vectorstore is not None:
        try:
            # Embed query
            proj_emb = embed_and_project(query)
            proj_emb = l2_normalize(np.array(proj_emb))

            # Get similarity with top document
            results = vectorstore.similarity_search_with_score_by_vector(
                embedding=proj_emb.tolist(),
                k=1
            )

            if results:
                _, similarity_score = results[0]
                metadata["similarity_score"] = float(similarity_score)

                # Adjust score: FAISS returns distance, we want similarity
                # L2 distance ranges 0-2 for normalized vectors
                # Convert to similarity: 1 - score makes sense for small scores
                similarity = 1.0 - min(similarity_score, 1.0)

                if similarity > CARDIOLOGY_SIMILARITY_THRESHOLD:
                    metadata["has_cardiology_keywords"] = _has_cardiology_keywords(
                        query
                    )
                    return ("CARDIOLOGY", similarity, metadata)
                else:
                    # Below threshold - unclear
                    metadata["has_cardiology_keywords"] = _has_cardiology_keywords(
                        query
                    )
                    return ("UNCLEAR", similarity, metadata)

        except Exception as e:
            print(f"⚠️  Vectorstore query failed: {e}. Falling back to keyword matching.")

    # ===================================
    # Step 4: Fallback to keyword matching
    # ===================================
    if _has_cardiology_keywords(query):
        metadata["has_cardiology_keywords"] = True
        return ("CARDIOLOGY", 0.7, metadata)

    # ===================================
    # Default: General chat
    # ===================================
    return ("GENERAL_CHAT", 0.5, metadata)
