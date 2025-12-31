from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from medirag.config import Settings
from medirag.llm.answer_runner import run_answer_llm
from medirag.llm.instructor_runner import run_instructor_llm
from medirag.llm.lmstudio_client import LMStudioClient
from medirag.prompting.definition_directive import render_final_prompt
from medirag.resources.loader import Dataset, load_jsonl_dataset
from medirag.retrieval.index_store import RetrievalAssets, load_or_build_faiss_index
from medirag.verification.s2_client import SemanticScholarClient
from medirag.verification.verify_rank import verify_and_rank_final_flat


@dataclass
class PipelineAssets:
    dataset: Dataset
    retrieval: RetrievalAssets
    s2_client: SemanticScholarClient
    llm_client: LMStudioClient


def init_assets(settings: Settings) -> PipelineAssets:
    dataset = load_jsonl_dataset(settings.jsonl_path)

    retrieval = load_or_build_faiss_index(
        embed_model_name=settings.embed_model_name,
        passages=dataset.passages,
        emb_path=settings.emb_path,
        index_path=settings.index_path,
    )

    if not settings.s2_api_key:
        raise RuntimeError("S2_API_KEY is not set (required for Semantic Scholar metadata verification).")

    s2_client = SemanticScholarClient(api_key=settings.s2_api_key)

    llm_client = LMStudioClient(base_url=settings.lmstudio_base_url, model=settings.lmstudio_model)

    return PipelineAssets(dataset=dataset, retrieval=retrieval, s2_client=s2_client, llm_client=llm_client)


def run_pipeline(*, assets: PipelineAssets, query: str, k: int) -> Dict[str, Any]:
    verified_docs = verify_and_rank_final_flat(
        query=query,
        data=assets.dataset.data,
        embedder=assets.retrieval.embedder,
        index=assets.retrieval.index,
        s2_client=assets.s2_client,
        k=k,
    )

    instruction_obj = run_instructor_llm(
        client=assets.llm_client,
        query=query,
        verified_docs=verified_docs,
        debug=False,
    )

    final_prompt = render_final_prompt(query, verified_docs, instruction_obj)

    answer_text = run_answer_llm(
        client=assets.llm_client,
        final_prompt=final_prompt,
        max_new_tokens=600,
        temperature=0.0,
    )

    return {
        "query": query,
        "verified_docs": verified_docs,
        "instruction_obj": instruction_obj,
        "final_prompt": final_prompt,
        "answer_text": answer_text,
    }
