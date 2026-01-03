from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RagAnswerRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    gen_max_length: int = Field(default=256, ge=16, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class RetrievedDoc(BaseModel):
    score: float
    id: str
    text: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class RagAnswerResponse(BaseModel):
    answer: str
    retrieved: List[RetrievedDoc]
    prompt: Optional[str] = None
    raw_generation: Optional[Any] = None


class AnswerOnlyResponse(BaseModel):
    answer: str


class FinalAnswerOnlyResponse(BaseModel):
    final_answer: str


class VerifiedAnswerResponse(BaseModel):
    answer: str
    final_answer: str
    verification_failed: bool
    retrieved: List[RetrievedDoc]
    prompt: Optional[str] = None
    nli_results: List[Dict[str, Any]] = Field(default_factory=list)
