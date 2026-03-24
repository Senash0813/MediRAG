# models/instruction_reranker.py

from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class InstructionReranker:
    """
    Instruction-following cross-encoder re-ranker using sentence-transformers.
    Lazy-loaded - model only initialized when first used.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", top_k: int = 3):
        self.model_name = model_name
        self.top_k = top_k
        self.reranker = None
        self._initialized = False

    def _initialize(self):
        """Lazy initialization - load model when first needed."""
        if self._initialized:
            return
            
        print(f"[Instruction Reranker] Initializing model {self.model_name} using CrossEncoder...")
        try:
            self.reranker = CrossEncoder(self.model_name)
            self._initialized = True
            print("[Instruction Reranker] ✓ Model loaded successfully")
        except Exception as e:
            print(f"[Instruction Reranker] ✗ FAILED to initialize: {e}")
            raise

    def rerank(self, user_query: str, blueprint: List[str], chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Scores and ranks the chunks based on the SLM blueprint.
        """
        if not chunks:
            print("[Instruction Reranker] No chunks provided to rank. Returning empty list.")
            return []
            
        self._initialize()
        print(f"[Instruction Reranker] Ranking {len(chunks)} chunks using Blueprint guidance...")
        
        # 1. Format the Blueprint string
        blueprint_text = "\n".join(blueprint)
        
        # 2. Create the "Super-Query" by sandwiching instruction and query
        super_query = (
            f"Instruction: Prioritize chunks that satisfy the following clinical blueprint:\n"
            f"{blueprint_text}\n\n"
            f"Query: {user_query}"
        )
        
        # 3. Create pairs of [super_query, chunk_text]
        pairs = [[super_query, chunk.get('answer', chunk.get('text', ''))] for chunk in chunks]
        
        try:
            # 4. Compute scores using CrossEncoder predict
            scores = self.reranker.predict(pairs)
            
            # CrossEncoder.predict returns a single float if len(pairs) == 1
            if isinstance(scores, float):
                scores = [scores]
                
            # 5. Attach scores to the chunks
            for chunk, score in zip(chunks, scores):
                chunk['rerank_score'] = float(score)
                
            # 6. Sort chunks by score in descending order
            ranked_chunks = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
            
            top_chunks = ranked_chunks[:self.top_k]
            
            if top_chunks:
                print(f"[Instruction Reranker] ✓ Top score: {top_chunks[0]['rerank_score']:.4f}")
                
            return top_chunks
            
        except Exception as e:
            print(f"[Instruction Reranker] ✗ Ranking FAILED: {e}")
            print("[Instruction Reranker] Returning original chunks (unranked) up to top_k.")
            return chunks[:self.top_k]