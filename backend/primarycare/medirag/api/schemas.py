from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    query: str
    # Frontend expects `answer`; keep `answer_text` name internally but serialize as `answer`.
    answer_text: str = Field(serialization_alias="answer")
    instruction_obj: Dict[str, Any]
    verified_docs: List[Dict[str, Any]]
    final_prompt: str
