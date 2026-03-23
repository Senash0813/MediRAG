from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, BaseModel, Field


class RetrievedDoc(BaseModel):
    score: float
    id: str
    text: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class AnswerRequest(BaseModel):
    query: str = Field(min_length=1, validation_alias=AliasChoices("query", "question"))
    top_k: Optional[int] = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    verify: bool = Field(default=True, description="Run NER/NLI/judge/regen pipeline")


class RagOnlyResponse(BaseModel):
    answer: str
    retrieved: List[RetrievedDoc]
    prompt: str
    ood: bool = False
    ood_info: Dict[str, Any] = Field(default_factory=dict)


class VerificationResponse(BaseModel):
    final_answer: str
    original_answer: str
    answer_level_result: Optional[Dict[str, Any]] = None
    sentence_level_results: List[Dict[str, Any]] = Field(default_factory=list)
    hallucination_candidates: List[Dict[str, Any]] = Field(default_factory=list)


class AnswerResponse(BaseModel):
    answer: str = Field(description="Final answer (post-verification/disclaimers)")
    original_answer: str
    retrieved: List[RetrievedDoc]
    prompt: str
    ood: bool = False
    ood_info: Dict[str, Any] = Field(default_factory=dict)
    verification: Optional[VerificationResponse] = None


class ReindexRequest(BaseModel):
    kb_file: Optional[str] = Field(default=None, description="Override settings.kb_file")
    force: bool = Field(default=False)


class HealthResponse(BaseModel):
    status: str
    kb_loaded: bool
    index_loaded: bool
    scope_index_loaded: bool
    scispacy_ready: bool = False
    ner_ready: bool = False
    nli_ready: bool = False
    judge_ready: bool = False
    kb_error: Optional[str] = None


class StageWiseResponse(BaseModel):
    query: str
    stage_a_domain_gate: Dict[str, Any] = Field(description="Domain/Scope checking results")
    stage_b_rag: Dict[str, Any] = Field(description="RAG answer generation")
    stage_c_risk_routing: Dict[str, Any] = Field(description="Entity extraction and risk scoring")
    stage_d_verification: Dict[str, Any] = Field(description="NLI verification results")
    stage_e_reconstruction: Dict[str, Any] = Field(description="Final answer reconstruction")
    stage_f_transparency: Dict[str, Any] = Field(description="Disclaimers and transparency")
