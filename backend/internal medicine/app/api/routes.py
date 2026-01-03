from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core import settings
from app.core.resources import get_resources, get_resources_error
from app.schemas.rag import FinalAnswerOnlyResponse, RagAnswerRequest, RagAnswerResponse, VerifiedAnswerResponse
from app.services.ner import build_final_results, filter_high_risk_sentences, split_sentences
from app.services.rag import answer_question
from app.services.verification import verify_and_postprocess


router = APIRouter()


@router.get("/health")
def health():
    try:
        resources = get_resources()
    except Exception:
        return {
            "ok": False,
            "error": get_resources_error(),
        }

    return {
        "ok": True,
        "kb_path": str(settings.KB_PATH),
        "index_path": str(settings.INDEX_PATH),
        "meta_path": str(settings.META_PATH),
        "docs_in_index": int(resources.index.ntotal) if hasattr(resources.index, "ntotal") else None,
        "device": resources.device,
    }


@router.post("/rag/answer", response_model=RagAnswerResponse)
def rag_answer(req: RagAnswerRequest):
    resources = get_resources()

    resp = answer_question(
        req.query,
        top_k=req.top_k,
        gen_max_length=req.gen_max_length,
        temperature=req.temperature,
        embedder=resources.embedder,
        index=resources.index,
        metadata=resources.metadata,
        rag_generator=resources.rag_generator,
    )

    return resp


@router.post("/rag/answer-verified", response_model=VerifiedAnswerResponse)
def rag_answer_verified(req: RagAnswerRequest):
    resources = get_resources()

    resp = answer_question(
        req.query,
        top_k=req.top_k,
        gen_max_length=req.gen_max_length,
        temperature=req.temperature,
        embedder=resources.embedder,
        index=resources.index,
        metadata=resources.metadata,
        rag_generator=resources.rag_generator,
    )

    rag_output = resp["answer"]

    claims = split_sentences(rag_output, nlp_sci=resources.nlp_sci)
    final_results = build_final_results(claims, nlp_bc5cdr=resources.nlp_bc5cdr, biomed_ner=resources.biomed_ner)
    hallucination_candidates = filter_high_risk_sentences(final_results)

    retrieved_texts = [r["text"] for r in resp.get("retrieved", [])]

    verification = verify_and_postprocess(
        rag_answer=rag_output,
        retrieved_texts=retrieved_texts,
        hallucination_candidates=hallucination_candidates,
        embedder=resources.embedder,
        nlp_sci=resources.nlp_sci,
        nli_pipeline=resources.nli_pipeline,
        rag_generator=resources.rag_generator,
        sim_threshold=settings.SIM_THRESHOLD,
        unverified_message=settings.UNVERIFIED_MESSAGE,
    )

    if "answer" not in resp:
        raise HTTPException(status_code=500, detail="RAG response missing 'answer'")

    return {
        "answer": verification["final_answer"],
        "final_answer": resp["answer"],
        "verification_failed": verification["verification_failed"],
        "retrieved": resp.get("retrieved", []),
        "prompt": resp.get("prompt"),
        "nli_results": verification.get("nli_results", []),
    }


@router.post("/rag/query3", response_model=FinalAnswerOnlyResponse)
def rag_query3(req: RagAnswerRequest):
    resources = get_resources()

    resp = answer_question(
        req.query,
        top_k=req.top_k,
        gen_max_length=req.gen_max_length,
        temperature=req.temperature,
        embedder=resources.embedder,
        index=resources.index,
        metadata=resources.metadata,
        rag_generator=resources.rag_generator,
    )

    if "answer" not in resp:
        raise HTTPException(status_code=500, detail="RAG response missing 'answer'")

    rag_output = resp["answer"]

    claims = split_sentences(rag_output, nlp_sci=resources.nlp_sci)
    final_results = build_final_results(claims, nlp_bc5cdr=resources.nlp_bc5cdr, biomed_ner=resources.biomed_ner)
    hallucination_candidates = filter_high_risk_sentences(final_results)

    retrieved_texts = [r["text"] for r in resp.get("retrieved", [])]

    verification = verify_and_postprocess(
        rag_answer=rag_output,
        retrieved_texts=retrieved_texts,
        hallucination_candidates=hallucination_candidates,
        embedder=resources.embedder,
        nlp_sci=resources.nlp_sci,
        nli_pipeline=resources.nli_pipeline,
        rag_generator=resources.rag_generator,
        sim_threshold=settings.SIM_THRESHOLD,
        unverified_message=settings.UNVERIFIED_MESSAGE,
    )

    return {
        "final_answer": verification["final_answer"],
    }


