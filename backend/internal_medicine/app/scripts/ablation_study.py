from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from app.core.config import Settings, get_settings
from app.services.domain_gate import domain_check
from app.services.rag import answer_question
from app.services.verification import VerificationConfig, VerificationModels, run_full_verification
from app.state import AppState, build_state


@dataclass(frozen=True)
class AblationVariant:
    name: str
    description: str
    settings_overrides: Dict[str, Any]
    use_verification: bool = True
    use_domain_gate: bool = True


DEFAULT_VARIANTS: Sequence[AblationVariant] = (
    AblationVariant(
        name="baseline",
        description="All stages enabled with default settings",
        settings_overrides={},
    ),
    AblationVariant(
        name="no_domain_gate",
        description="Skip scope/domain filtering",
        settings_overrides={"enable_domain_gate": False},
        use_domain_gate=False,
    ),
    AblationVariant(
        name="rag_only",
        description="Retrieval + generation only",
        settings_overrides={"enable_domain_gate": False},
        use_domain_gate=False,
        use_verification=False,
    ),
    AblationVariant(
        name="no_ner",
        description="Disable NER-driven risk routing",
        settings_overrides={"enable_ner": False},
    ),
    AblationVariant(
        name="no_nli",
        description="Disable NLI verification",
        settings_overrides={"enable_nli": False},
    ),
    AblationVariant(
        name="strict_retrieval",
        description="Tighten retrieval confidence filters",
        settings_overrides={
            "retrieval_min_top1": 0.55,
            "retrieval_min_avg_topk": 0.45,
            "retrieval_min_margin_top2": 0.05,
            "retrieval_keyword_overlap_min": 0.30,
            "retrieval_require_distinct_docs": True,
        },
    ),
    AblationVariant(
        name="loose_retrieval",
        description="Relax retrieval confidence filters",
        settings_overrides={
            "retrieval_min_top1": 0.0,
            "retrieval_min_avg_topk": 0.0,
            "retrieval_min_margin_top2": 0.0,
            "retrieval_keyword_overlap_min": 0.0,
            "retrieval_require_distinct_docs": False,
        },
    ),
)


def _load_examples(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    data = json.loads(raw)
    if isinstance(data, dict):
        if "examples" in data and isinstance(data["examples"], list):
            return list(data["examples"])
        raise ValueError("JSON input must contain a top-level list or an 'examples' field")
    if not isinstance(data, list):
        raise ValueError("JSON input must be a list of examples")
    return list(data)


def _make_variant_settings(base: Settings, variant: AblationVariant) -> Settings:
    payload = base.model_dump()
    payload.update(variant.settings_overrides)
    return Settings(**payload)


def _evaluate_variant(
    *,
    state: AppState,
    variant: AblationVariant,
    query: str,
    top_k: int | None,
    temperature: float,
    verify: bool,
) -> Dict[str, Any]:
    settings = _make_variant_settings(state.settings, variant)
    effective_top_k = top_k or settings.top_k

    stage_a: Dict[str, Any] = {"decision": "SKIPPED"}
    if variant.use_domain_gate and settings.enable_domain_gate and len(state.scope_meta) > 0 and getattr(state.scope_index, "ntotal", 0) > 0:
        in_domain, info = domain_check(
            query=query,
            embedder=state.embedder,
            scope_index=state.scope_index,
            scope_meta=state.scope_meta,
            min_top1=settings.scope_min_top1,
            min_avg_topk=settings.scope_min_avg_topk,
            min_cohesion=settings.scope_min_cohesion,
            top_k=settings.scope_top_k,
        )
        stage_a = {
            "decision": "IN_DOMAIN" if in_domain else "OUT_OF_DOMAIN",
            "info": info,
        }
        if not in_domain:
            return {
                "variant": variant.name,
                "query": query,
                "ood": True,
                "answer": "❌ Out of scope for this knowledge base",
                "original_answer": None,
                "stage_a": stage_a,
                "stage_b": {"skipped": True, "reason": "out_of_domain"},
                "verification": None,
            }

    answer, retrieved, prompt, _raw, ood, ood_info = answer_question(
        query=query,
        embedder=state.embedder,
        index=state.index,
        metadata=state.metadata,
        top_k=effective_top_k,
        min_retrieval_top1=settings.retrieval_min_top1,
        min_retrieval_avg_topk=settings.retrieval_min_avg_topk,
        retrieval_score_topk=settings.retrieval_min_topk,
        min_retrieval_margin_top2=settings.retrieval_min_margin_top2,
        retrieval_avg_topk_override=settings.retrieval_avg_topk_override,
        retrieval_keyword_overlap_min=settings.retrieval_keyword_overlap_min,
        retrieval_require_distinct_docs=settings.retrieval_require_distinct_docs,
        retrieval_min_distinct_docs=settings.retrieval_min_distinct_docs,
        nli_pipeline=state.nli_pipeline,
        ollama=state.ollama,
        generator_model=settings.ollama_generator_model,
        gen_max_tokens=settings.gen_max_tokens,
        temperature=temperature,
    )

    if ood:
        return {
            "variant": variant.name,
            "query": query,
            "ood": True,
            "answer": answer,
            "original_answer": answer,
            "retrieved_count": len(retrieved),
            "stage_a": stage_a,
            "stage_b": {
                "generated_answer": answer,
                "retrieved_count": len(retrieved),
                "ood_info": ood_info,
            },
            "verification": None,
        }

    if not (variant.use_verification and verify):
        return {
            "variant": variant.name,
            "query": query,
            "ood": False,
            "answer": answer,
            "original_answer": answer,
            "retrieved_count": len(retrieved),
            "stage_a": stage_a,
            "stage_b": {
                "generated_answer": answer,
                "retrieved_count": len(retrieved),
                "prompt": prompt,
            },
            "verification": None,
        }

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
        nli_pipeline=state.nli_pipeline if settings.enable_nli else None,
        nlp_sci=state.nlp_sci,
        nlp_bc5cdr=state.nlp_bc5cdr,
        biomed_ner=state.biomed_ner if settings.enable_ner else None,
        judge_t2t=getattr(state, "judge_t2t", None),
    )

    ver = run_full_verification(
        original_answer=answer,
        retrieved_texts=[r["text"] for r in retrieved],
        embedder=state.embedder,
        ollama=state.ollama,
        generator_model=settings.ollama_generator_model,
        judge_model=settings.ollama_judge_model,
        config=config,
        models=models,
    )

    return {
        "variant": variant.name,
        "query": query,
        "ood": False,
        "answer": ver["final_answer"],
        "original_answer": ver["original_answer"],
        "retrieved_count": len(retrieved),
        "stage_a": stage_a,
        "stage_b": {
            "generated_answer": answer,
            "retrieved_count": len(retrieved),
            "prompt": prompt,
        },
        "verification": {
            "answer_level_result": ver.get("answer_level_result"),
            "hallucination_candidates": len(ver.get("hallucination_candidates", [])),
            "sentence_level_results": ver.get("sentence_level_results", []),
        },
    }


def _score_prediction(prediction: str, reference: str | None) -> Dict[str, Any]:
    if reference is None:
        return {}

    pred_norm = " ".join((prediction or "").lower().split())
    ref_norm = " ".join((reference or "").lower().split())
    exact_match = float(pred_norm == ref_norm)

    pred_tokens = set(pred_norm.split())
    ref_tokens = set(ref_norm.split())
    overlap = len(pred_tokens & ref_tokens)
    precision = overlap / len(pred_tokens) if pred_tokens else 0.0
    recall = overlap / len(ref_tokens) if ref_tokens else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "exact_match": exact_match,
        "token_f1": f1,
    }


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "ood_rate": 0.0,
            "avg_retrieved_count": 0.0,
            "exact_match": None,
            "token_f1": None,
        }

    count = len(rows)
    ood_rate = sum(1 for row in rows if row.get("ood")) / count
    avg_retrieved = sum(float(row.get("retrieved_count", 0)) for row in rows) / count

    exact_values = [row["exact_match"] for row in rows if row.get("exact_match") is not None]
    f1_values = [row["token_f1"] for row in rows if row.get("token_f1") is not None]

    return {
        "ood_rate": ood_rate,
        "avg_retrieved_count": avg_retrieved,
        "exact_match": (sum(exact_values) / len(exact_values)) if exact_values else None,
        "token_f1": (sum(f1_values) / len(f1_values)) if f1_values else None,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an ablation study for the internal_medicine pipeline")
    parser.add_argument("--input", type=Path, required=True, help="JSON or JSONL file with query examples")
    parser.add_argument("--output-dir", type=Path, default=Path("ablation_results"), help="Directory to write outputs")
    parser.add_argument("--variants", nargs="*", default=[v.name for v in DEFAULT_VARIANTS], help="Variant names to run")
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top-k")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification for all variants")
    parser.add_argument("--force-reindex", action="store_true", help="Rebuild KB/scope indexes before running")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    examples = _load_examples(args.input)
    if not examples:
        raise SystemExit("No examples found in the input file")

    selected = [variant for variant in DEFAULT_VARIANTS if variant.name in set(args.variants)]
    if not selected:
        raise SystemExit("No matching variants selected")

    base_settings = get_settings()
    state = build_state(base_settings, force_reindex=bool(args.force_reindex))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summary: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []

    for variant in selected:
        variant_rows: List[Dict[str, Any]] = []
        for example in examples:
            query = str(example.get("query") or example.get("question") or "").strip()
            if not query:
                continue

            prediction = _evaluate_variant(
                state=state,
                variant=variant,
                query=query,
                top_k=args.top_k,
                temperature=float(args.temperature),
                verify=not bool(args.no_verify),
            )

            metrics = _score_prediction(prediction["answer"], example.get("reference_answer") or example.get("answer"))
            row = {
                "variant": variant.name,
                "query": query,
                "reference_answer": example.get("reference_answer") or example.get("answer"),
                "answer": prediction["answer"],
                "original_answer": prediction.get("original_answer"),
                "ood": prediction.get("ood"),
                "retrieved_count": prediction.get("retrieved_count", 0),
                **metrics,
            }
            variant_rows.append(row)
            all_rows.append(row)

        summary = {
            "variant": variant.name,
            "description": variant.description,
            **_aggregate(variant_rows),
        }
        all_summary.append(summary)

        variant_dir = args.output_dir / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(variant_dir / "rows.csv", variant_rows)
        (variant_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    (args.output_dir / "summary.json").write_text(json.dumps(all_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(args.output_dir / "all_rows.csv", all_rows)

    print(json.dumps(all_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()