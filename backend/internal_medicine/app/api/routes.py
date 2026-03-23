from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from ..core.config import get_settings
from ..schemas import (
    AnswerRequest,
    AnswerResponse,
    HealthResponse,
    RagOnlyResponse,
    ReindexRequest,
    RetrievedDoc,
    StageWiseResponse,
    VerificationResponse,
)
from ..services.domain_gate import domain_check
from ..services.rag import answer_question
from ..services.verification import VerificationConfig, VerificationModels, run_full_verification
from ..state import AppState, build_state

router = APIRouter()


def get_state(request: Request) -> AppState:
    state = getattr(request.app.state, "rag_state", None)
    if state is None:
        raise HTTPException(status_code=503, detail="App not initialized")
    return state


StateDep = Annotated[AppState, Depends(get_state)]


@router.get("/health")
def health(state: StateDep) -> HealthResponse:
    return HealthResponse(
        status="ok",
        kb_loaded=len(state.kb_docs) > 0,
        index_loaded=state.index is not None and getattr(state.index, "ntotal", 0) > 0,
        scope_index_loaded=state.scope_index is not None and getattr(state.scope_index, "ntotal", 0) > 0,
        scispacy_ready=bool(getattr(state, "scispacy_ready", False)),
        ner_ready=bool(getattr(state, "ner_ready", False)),
        nli_ready=bool(getattr(state, "nli_ready", False)),
        judge_ready=bool(getattr(state, "judge_ready", False)),
        kb_error=getattr(state, "kb_load_error", None),
    )


@router.post("/rag")
def rag_only(req: AnswerRequest, state: StateDep) -> RagOnlyResponse:
    settings = state.settings
    top_k = req.top_k or settings.top_k

    if getattr(state.index, "ntotal", 0) == 0 or len(state.metadata) == 0:
        return RagOnlyResponse(
            answer=(
                "Knowledge base is empty or not indexed yet. "
                "Set KB_FILE and call /reindex (or restart the API) to build embeddings."
            ),
            retrieved=[],
            prompt="",
            ood=False,
            ood_info={},
        )

    if settings.enable_domain_gate:
        if len(state.scope_meta) == 0 or getattr(state.scope_index, "ntotal", 0) == 0:
            # No scope index available -> skip gating.
            info = {}
        else:
            in_domain, info = domain_check(
                query=req.query,
                embedder=state.embedder,
                scope_index=state.scope_index,
                scope_meta=state.scope_meta,
                min_top1=settings.scope_min_top1,
                min_avg_topk=settings.scope_min_avg_topk,
                min_cohesion=settings.scope_min_cohesion,
                top_k=settings.scope_top_k,
            )
            if not in_domain:
                return RagOnlyResponse(
                    answer=(
                        "❌ Out of scope for this knowledge base (insufficient domain coverage).\n\n"
                        "Try:\n"
                        "• Rephrase with more specific medical context\n"
                        "• Ask about a condition/topic covered in the KB\n"
                        "• Provide a source excerpt to add to the KB"
                    ),
                    retrieved=[],
                    prompt="",
                    ood=True,
                    ood_info=info,
                )
    else:
        info = {}

    answer, retrieved, prompt, _raw = answer_question(
        query=req.query,
        embedder=state.embedder,
        index=state.index,
        metadata=state.metadata,
        top_k=top_k,
        ollama=state.ollama,
        generator_model=settings.ollama_generator_model,
        gen_max_tokens=settings.gen_max_tokens,
        temperature=req.temperature,
    )

    return RagOnlyResponse(
        answer=answer,
        retrieved=[RetrievedDoc(**r) for r in retrieved],
        prompt=prompt,
        ood=False,
        ood_info=info,
    )


@router.post("/query3")
def answer(req: AnswerRequest, state: StateDep) -> AnswerResponse:
    settings = state.settings

    rag_resp = rag_only(AnswerRequest(query=req.query, top_k=req.top_k, temperature=req.temperature, verify=False), state)

    if rag_resp.ood:
        return AnswerResponse(
            answer=rag_resp.answer,
            original_answer=rag_resp.answer,
            retrieved=[],
            prompt="",
            ood=True,
            ood_info=rag_resp.ood_info,
            verification=None,
        )

    if not req.verify:
        return AnswerResponse(
            answer=rag_resp.answer,
            original_answer=rag_resp.answer,
            retrieved=rag_resp.retrieved,
            prompt=rag_resp.prompt,
            ood=False,
            ood_info=rag_resp.ood_info,
            verification=None,
        )

    retrieved_texts = [r.text for r in rag_resp.retrieved]

    config = VerificationConfig(
        risk_threshold=settings.risk_threshold,
        max_evidence_chars=settings.max_evidence_chars,
        answer_supported_th=settings.answer_nli_supported_th,
        answer_unsupported_th=settings.answer_nli_unsupported_th,
        fast_verified_th=settings.fast_verified_th,
        fast_hallucinated_th=settings.fast_hallucinated_th,
        full_verified_th=settings.full_verified_th,
        full_hallucinated_th=settings.full_hallucinated_th,
        sim_for_regen=settings.sim_for_regen,
        regen_max_tokens=settings.regen_max_tokens,
        hf_judge_max_new_tokens=settings.hf_judge_max_new_tokens,
    )

    models = VerificationModels(
        nli_pipeline=state.nli_pipeline,
        nlp_sci=state.nlp_sci,
        nlp_bc5cdr=state.nlp_bc5cdr,
        biomed_ner=state.biomed_ner,
        judge_t2t=getattr(state, "judge_t2t", None),
    )

    ver = run_full_verification(
        original_answer=rag_resp.answer,
        retrieved_texts=retrieved_texts,
        embedder=state.embedder,
        ollama=state.ollama,
        generator_model=settings.ollama_generator_model,
        judge_model=settings.ollama_judge_model,
        config=config,
        models=models,
    )

    verification = VerificationResponse(
        final_answer=ver["final_answer"],
        original_answer=ver["original_answer"],
        answer_level_result=ver.get("answer_level_result"),
        sentence_level_results=ver.get("sentence_level_results", []),
        hallucination_candidates=ver.get("hallucination_candidates", []),
    )

    return AnswerResponse(
        answer=ver["final_answer"],
        original_answer=ver["original_answer"],
        retrieved=rag_resp.retrieved,
        prompt=rag_resp.prompt,
        ood=False,
        ood_info=rag_resp.ood_info,
        verification=verification,
    )


@router.post("/reindex")
def reindex(req: ReindexRequest, request: Request, state: StateDep):
    settings = state.settings

    if req.kb_file is not None:
        settings.kb_file = Path(req.kb_file)

    new_state = build_state(settings, force_reindex=req.force)
    request.app.state.rag_state = new_state

    return {
        "ok": True,
        "kb_docs": len(new_state.kb_docs),
        "index_ntotal": int(getattr(new_state.index, "ntotal", 0)),
        "scope_ntotal": int(getattr(new_state.scope_index, "ntotal", 0)),
        "ner_ready": bool(new_state.ner_ready),
        "nli_ready": bool(new_state.nli_ready),
        "scispacy_ready": bool(new_state.scispacy_ready),
    }


@router.post("/evaluate-stages")
def evaluate_stages(req: AnswerRequest, state: StateDep) -> StageWiseResponse:
    """
    Stage-wise evaluation endpoint for demonstration purposes.
    Shows what each pipeline stage outputs without excessive detail.
    """
    settings = state.settings
    
    # ============================================================
    # STAGE A: Domain / Scope Gate
    # ============================================================
    stage_a = {"stage": "A - Domain/Scope Gate"}
    
    if settings.enable_domain_gate and len(state.scope_meta) > 0 and getattr(state.scope_index, "ntotal", 0) > 0:
        in_domain, info = domain_check(
            query=req.query,
            embedder=state.embedder,
            scope_index=state.scope_index,
            scope_meta=state.scope_meta,
            min_top1=settings.scope_min_top1,
            min_avg_topk=settings.scope_min_avg_topk,
            min_cohesion=settings.scope_min_cohesion,
            top_k=settings.scope_top_k,
        )
        stage_a["decision"] = "IN_DOMAIN" if in_domain else "OUT_OF_DOMAIN"
        stage_a["top1_similarity"] = round(info.get("top1", 0.0), 3)
        stage_a["avg_topk_similarity"] = round(info.get("avgk", 0.0), 3)
        stage_a["cohesion"] = round(info.get("cohesion", 0.0), 3)
        
        if not in_domain:
            return StageWiseResponse(
                query=req.query,
                stage_a_domain_gate=stage_a,
                stage_b_rag={"skipped": "Query out of domain"},
                stage_c_risk_routing={"skipped": "Query out of domain"},
                stage_d_verification={"skipped": "Query out of domain"},
                stage_e_reconstruction={"skipped": "Query out of domain"},
                stage_f_transparency={"final_answer": "❌ Out of scope for this knowledge base"},
            )
    else:
        stage_a["decision"] = "GATE_DISABLED_OR_NO_INDEX"
    
    # ============================================================
    # STAGE B: RAG Answer Generation
    # ============================================================
    top_k = req.top_k or settings.top_k
    answer, retrieved, _prompt, _raw = answer_question(
        query=req.query,
        embedder=state.embedder,
        index=state.index,
        metadata=state.metadata,
        top_k=top_k,
        ollama=state.ollama,
        generator_model=settings.ollama_generator_model,
        gen_max_tokens=settings.gen_max_tokens,
        temperature=req.temperature,
    )
    
    stage_b = {
        "stage": "B - RAG Answer Generation",
        "generated_answer": answer,
        "num_retrieved_docs": len(retrieved),
        "top_doc_scores": [round(r["score"], 3) for r in retrieved[:3]],
    }
    
    # If verification disabled, return early
    if not req.verify:
        return StageWiseResponse(
            query=req.query,
            stage_a_domain_gate=stage_a,
            stage_b_rag=stage_b,
            stage_c_risk_routing={"skipped": "Verification disabled"},
            stage_d_verification={"skipped": "Verification disabled"},
            stage_e_reconstruction={"final_answer": answer},
            stage_f_transparency={"final_answer": answer},
        )
    
    # ============================================================
    # STAGE C: Risk Routing (entity extraction + risk scoring)
    # ============================================================
    from ..services.verification import (
        extract_entities_multi_ner,
        merge_entities,
        score_sentence_risk,
        split_sentences_with_ids,
    )
    
    retrieved_texts = [r["text"] for r in retrieved]
    config = VerificationConfig(
        risk_threshold=settings.risk_threshold,
        max_evidence_chars=settings.max_evidence_chars,
        answer_supported_th=settings.answer_nli_supported_th,
        answer_unsupported_th=settings.answer_nli_unsupported_th,
        fast_verified_th=settings.fast_verified_th,
        fast_hallucinated_th=settings.fast_hallucinated_th,
        full_verified_th=settings.full_verified_th,
        full_hallucinated_th=settings.full_hallucinated_th,
        sim_for_regen=settings.sim_for_regen,
        regen_max_tokens=settings.regen_max_tokens,
        hf_judge_max_new_tokens=settings.hf_judge_max_new_tokens,
    )
    
    models = VerificationModels(
        nli_pipeline=state.nli_pipeline,
        nlp_sci=state.nlp_sci,
        nlp_bc5cdr=state.nlp_bc5cdr,
        biomed_ner=state.biomed_ner,
        judge_t2t=getattr(state, "judge_t2t", None),
    )
    
    # Split sentences and extract entities
    claims_with_ids = split_sentences_with_ids(answer, nlp_sci=models.nlp_sci)
    
    risk_results = []
    high_risk_count = 0
    
    for item in claims_with_ids:
        claim = item["text"]
        sent_id = item["sent_id"]
        
        raw_entities = extract_entities_multi_ner(
            claim,
            nlp_bc5cdr=models.nlp_bc5cdr,
            biomed_ner=models.biomed_ner,
        )
        merged_entities = merge_entities(raw_entities)
        risk_score, reasons = score_sentence_risk(claim, merged_entities)
        
        risk_results.append({
            "sent_id": sent_id,
            "sentence": claim[:100] + "..." if len(claim) > 100 else claim,
            "num_entities": len(merged_entities),
            "risk_score": risk_score,
            "reasons": reasons,
            "is_high_risk": risk_score >= config.risk_threshold,
        })
        
        if risk_score >= config.risk_threshold:
            high_risk_count += 1
    
    stage_c = {
        "stage": "C - Risk Routing",
        "total_sentences": len(claims_with_ids),
        "high_risk_sentences": high_risk_count,
        "risk_threshold": config.risk_threshold,
        "sentence_risks": risk_results,
    }
    
    # ============================================================
    # STAGE D: Verification
    # ============================================================
    ver = run_full_verification(
        original_answer=answer,
        retrieved_texts=retrieved_texts,
        embedder=state.embedder,
        ollama=state.ollama,
        generator_model=settings.ollama_generator_model,
        judge_model=settings.ollama_judge_model,
        config=config,
        models=models,
    )
    
    if ver.get("answer_level_result"):
        # Answer-level verification path (no high-risk sentences)
        ans_res = ver["answer_level_result"]
        stage_d = {
            "stage": "D - Verification (Answer-Level)",
            "path": "D0 - No high-risk sentences detected",
            "nli_label": ans_res["nli"]["nli_label"],
            "entail_prob": round(ans_res["nli"]["entail_prob"], 3),
            "status": ans_res["final_status"],
            "judge_called": ans_res["judge"] is not None,
        }
    else:
        # Sentence-level verification path
        sent_results = ver.get("sentence_level_results", [])
        stage_d = {
            "stage": "D - Verification (Sentence-Level)",
            "path": "D1 - High-risk sentences detected",
            "verified_sentences": len([r for r in sent_results if r.get("final_status") == "VERIFIED"]),
            "regenerated_sentences": len([r for r in sent_results if r.get("regen_status") == "REGEN_ACCEPTED"]),
            "hallucinated_sentences": len([r for r in sent_results if "HALLUCINATED" in r.get("final_status", "")]),
            "uncertain_sentences": len([r for r in sent_results if "UNCERTAIN" in r.get("final_status", "")]),
            "sample_results": [
                {
                    "sent_id": r["sent_id"],
                    "similarity": round(r.get("similarity_score", 0), 3),
                    "nli_label": r.get("nli_label"),
                    "fast_status": r.get("fast_status"),
                    "final_status": r.get("final_status"),
                    "regenerated": r.get("regen_status") == "REGEN_ACCEPTED",
                }
                for r in sent_results[:3]  # Show first 3 for demo
            ],
        }
    
    # ============================================================
    # STAGE E: Reconstruction
    # ============================================================
    num_replacements = len([r for r in ver.get("sentence_level_results", []) 
                           if r.get("final_claim") != r.get("original_claim")])
    
    stage_e = {
        "stage": "E - Reconstruction",
        "replacements_made": num_replacements,
        "final_answer_excerpt": ver["final_answer"][:200] + "..." if len(ver["final_answer"]) > 200 else ver["final_answer"],
    }
    
    # ============================================================
    # STAGE F: Transparency Layer
    # ============================================================
    has_disclaimer = ver["final_answer"].startswith("⚠️")
    stage_f = {
        "stage": "F - Transparency Layer",
        "disclaimer_added": has_disclaimer,
        "final_answer": ver["final_answer"],
    }
    
    return StageWiseResponse(
        query=req.query,
        stage_a_domain_gate=stage_a,
        stage_b_rag=stage_b,
        stage_c_risk_routing=stage_c,
        stage_d_verification=stage_d,
        stage_e_reconstruction=stage_e,
        stage_f_transparency=stage_f,
    )
