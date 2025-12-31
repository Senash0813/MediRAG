from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    query: str
    answer_text: str
    instruction_obj: Dict[str, Any]
    verified_docs: List[Dict[str, Any]]
    final_prompt: str
