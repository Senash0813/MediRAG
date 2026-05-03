#!/usr/bin/env python3
"""Offline retrieval evaluation for the MediRAG pipeline.

Usage:
    python evaluation/retrieval/code/evaluate_retrieval_api.py \
        --dataset_path evaluation/retrieval/data/eval_dataset_20.jsonl \
        --api_url http://localhost:8003/retrieve/detailed \
        --semantic_n 10 \
        --final_k 5 \
        --eval_k 5 \
        --output_dir evaluation/retrieval/results

This script evaluates two checkpoints from the detailed retrieval API:
- raw retrieval: response["retrieval"]
- final validated retrieval: response["phase2_validation"]

It computes Precision@k, Recall@k, MRR@k, and nDCG@k against gold_doc_ids.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests


@dataclass(frozen=True)
class EvalExample:
    """Single evaluation example from the dataset."""

    question_id: str
    question: str
    gold_doc_ids: List[str]
    source_paper_id: Optional[str] = None
    source_qa_id: Optional[str] = None
    specialty_tag: Optional[str] = None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Offline retrieval evaluation for /retrieve/detailed")
    parser.add_argument(
        "--dataset_path",
        default="evaluation/retrieval/data/eval_dataset_20.jsonl",
        help="Path to evaluation JSONL dataset",
    )
    parser.add_argument("--api_url", default="http://localhost:8003/retrieve/detailed", help="Detailed retrieval API URL")
    parser.add_argument("--semantic_n", type=int, default=10, help="Semantic candidate count sent to the API")
    parser.add_argument("--final_k", type=int, default=5, help="Final validated document count sent to the API")
    parser.add_argument("--eval_k", type=int, default=5, help="Evaluation cutoff for metrics")
    parser.add_argument("--output_dir", default="evaluation/retrieval/results", help="Directory to store results")
    parser.add_argument("--timeout", type=float, default=75.0, help="HTTP request timeout in seconds")
    parser.add_argument("--retries", type=int, default=3, help="Retry count for transient failures")
    parser.add_argument("--retry_sleep", type=float, default=1.5, help="Sleep between retries in seconds")
    parser.add_argument("--save_api_outputs", action="store_true", help="Save raw API responses to api_outputs.jsonl")
    return parser.parse_args()


def load_jsonl_dataset(path: str | Path) -> List[Dict[str, Any]]:
    """Load a JSONL dataset into memory."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    rows: List[Dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no} in {dataset_path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at line {line_no} in {dataset_path}")
            rows.append(row)
    return rows


def build_examples(rows: Sequence[Dict[str, Any]]) -> List[EvalExample]:
    """Normalize dataset rows into EvalExample records."""
    examples: List[EvalExample] = []
    seen_question_ids: set[str] = set()

    for index, row in enumerate(rows, start=1):
        question_id = str(row.get("question_id", "")).strip()
        question = str(row.get("question", "")).strip()
        gold_doc_ids = [str(item).strip() for item in row.get("gold_doc_ids", []) if str(item).strip()]
        source_paper_id = str(row.get("source_paper_id", "")).strip() or None
        source_qa_id = str(row.get("source_qa_id", "")).strip() or None
        specialty_tag = str(row.get("specialty_tag", "")).strip() or None

        if not question_id:
            raise ValueError(f"Missing question_id at dataset row {index}")
        if question_id in seen_question_ids:
            raise ValueError(f"Duplicate question_id detected: {question_id}")
        if not question:
            raise ValueError(f"Missing question text for question_id={question_id}")
        if not gold_doc_ids:
            raise ValueError(f"Missing gold_doc_ids for question_id={question_id}")

        seen_question_ids.add(question_id)
        examples.append(
            EvalExample(
                question_id=question_id,
                question=question,
                gold_doc_ids=gold_doc_ids,
                source_paper_id=source_paper_id,
                source_qa_id=source_qa_id,
                specialty_tag=specialty_tag,
            )
        )

    return examples


def extract_ranked_paper_ids(items: Sequence[Dict[str, Any]], id_key: str = "paper_id") -> List[str]:
    """Extract unique ranked paper IDs, preserving first-seen order."""
    ranked_ids: List[str] = []
    seen: set[str] = set()

    for item in items or []:
        if not isinstance(item, dict):
            continue
        paper_id = str(item.get(id_key, "")).strip()
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        ranked_ids.append(paper_id)

    return ranked_ids


def precision_at_k(ranked_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """Compute Precision@k using binary relevance."""
    if k <= 0:
        return 0.0
    top_k = list(ranked_ids[:k])
    if not top_k:
        return 0.0
    hits = sum(1 for paper_id in top_k if paper_id in gold_ids)
    return hits / float(k)


def recall_at_k(ranked_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """Compute Recall@k using binary relevance."""
    if k <= 0 or not gold_ids:
        return 0.0
    top_k = list(ranked_ids[:k])
    hits = sum(1 for paper_id in top_k if paper_id in gold_ids)
    return hits / float(len(gold_ids))


def mrr_at_k(ranked_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """Compute MRR@k using the first relevant hit."""
    if k <= 0:
        return 0.0
    for rank, paper_id in enumerate(ranked_ids[:k], start=1):
        if paper_id in gold_ids:
            return 1.0 / float(rank)
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """Compute binary nDCG@k."""
    if k <= 0 or not gold_ids:
        return 0.0

    dcg = 0.0
    for rank, paper_id in enumerate(ranked_ids[:k], start=1):
        relevance = 1.0 if paper_id in gold_ids else 0.0
        if relevance > 0:
            dcg += relevance / math.log2(rank + 1)

    ideal_hits = min(len(gold_ids), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg <= 1e-12:
        return 0.0
    return dcg / idcg


def post_retrieve_detailed(
    api_url: str,
    question: str,
    semantic_n: int,
    final_k: int,
    timeout: float,
    retries: int,
    retry_sleep: float,
) -> Dict[str, Any]:
    """Call the detailed retrieval API with retries for transient failures."""
    payload = {
        "query": question,
        "semantic_n": semantic_n,
        "final_k": final_k,
    }

    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(api_url, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("API response must be a JSON object")
            return data
        except Exception as exc:  # pragma: no cover - runtime network resilience
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep)
                continue
            raise RuntimeError(
                f"Failed to call {api_url} after {retries + 1} attempts: {last_error}"
            ) from last_error

    raise RuntimeError(f"Failed to call {api_url}") from last_error


def _rank_of_first_hit(ranked_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> int:
    """Return 1-based rank of first hit within top-k, or 0 if no hit."""
    for rank, paper_id in enumerate(ranked_ids[:k], start=1):
        if paper_id in gold_ids:
            return rank
    return 0


def evaluate_single_query(example: EvalExample, api_response: Dict[str, Any], eval_k: int) -> Dict[str, Any]:
    """Evaluate raw and final rankings for a single query."""
    in_scope = bool(api_response.get("scope_gate", {}).get("in_scope", False))

    raw_items = api_response.get("retrieval", []) or []
    final_items = api_response.get("phase2_validation", []) or []

    raw_ids = extract_ranked_paper_ids(raw_items)
    final_ids = extract_ranked_paper_ids(final_items)

    gold_ids = [str(item).strip() for item in example.gold_doc_ids if str(item).strip()]

    raw_first_hit_rank = _rank_of_first_hit(raw_ids, gold_ids, eval_k)
    final_first_hit_rank = _rank_of_first_hit(final_ids, gold_ids, eval_k)

    raw_metrics = {
        "precision_at_k": precision_at_k(raw_ids, gold_ids, eval_k),
        "recall_at_k": recall_at_k(raw_ids, gold_ids, eval_k),
        "mrr_at_k": mrr_at_k(raw_ids, gold_ids, eval_k),
        "ndcg_at_k": ndcg_at_k(raw_ids, gold_ids, eval_k),
    }
    final_metrics = {
        "precision_at_k": precision_at_k(final_ids, gold_ids, eval_k),
        "recall_at_k": recall_at_k(final_ids, gold_ids, eval_k),
        "mrr_at_k": mrr_at_k(final_ids, gold_ids, eval_k),
        "ndcg_at_k": ndcg_at_k(final_ids, gold_ids, eval_k),
    }

    raw_hit_count = sum(1 for paper_id in raw_ids[:eval_k] if paper_id in gold_ids)
    final_hit_count = sum(1 for paper_id in final_ids[:eval_k] if paper_id in gold_ids)
    raw_success_at_k = int(raw_first_hit_rank > 0)
    final_success_at_k = int(final_first_hit_rank > 0)

    return {
        "question_id": example.question_id,
        "question": example.question,
        "gold_doc_ids": json.dumps(gold_ids, ensure_ascii=False),
        "in_scope": in_scope,
        "raw_ids": json.dumps(raw_ids[:eval_k], ensure_ascii=False),
        "final_ids": json.dumps(final_ids[:eval_k], ensure_ascii=False),
        "raw_hit_count": raw_hit_count,
        "final_hit_count": final_hit_count,
        "raw_success_at_k": raw_success_at_k,
        "final_success_at_k": final_success_at_k,
        "raw_precision_at_k": raw_metrics["precision_at_k"],
        "raw_recall_at_k": raw_metrics["recall_at_k"],
        "raw_mrr_at_k": raw_metrics["mrr_at_k"],
        "raw_ndcg_at_k": raw_metrics["ndcg_at_k"],
        "final_precision_at_k": final_metrics["precision_at_k"],
        "final_recall_at_k": final_metrics["recall_at_k"],
        "final_mrr_at_k": final_metrics["mrr_at_k"],
        "final_ndcg_at_k": final_metrics["ndcg_at_k"],
        "raw_first_hit_rank": raw_first_hit_rank,
        "final_first_hit_rank": final_first_hit_rank,
        "error": "",
        "raw_api_response": api_response,
    }


def aggregate_metrics(per_query_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate retrieval metrics across queries."""
    successful_rows = [row for row in per_query_rows if not row.get("error")]
    metric_names = [
        "raw_precision_at_k",
        "raw_recall_at_k",
        "raw_mrr_at_k",
        "raw_ndcg_at_k",
        "final_precision_at_k",
        "final_recall_at_k",
        "final_mrr_at_k",
        "final_ndcg_at_k",
    ]

    if successful_rows:
        means = {name: mean(float(row[name]) for row in successful_rows) for name in metric_names}
    else:
        means = dict.fromkeys(metric_names, 0.0)

    deltas = {
        "delta_precision_at_k": means["final_precision_at_k"] - means["raw_precision_at_k"],
        "delta_recall_at_k": means["final_recall_at_k"] - means["raw_recall_at_k"],
        "delta_mrr_at_k": means["final_mrr_at_k"] - means["raw_mrr_at_k"],
        "delta_ndcg_at_k": means["final_ndcg_at_k"] - means["raw_ndcg_at_k"],
    }

    total_queries = len(per_query_rows)
    successful_queries = len(successful_rows)
    failed_queries = sum(1 for row in per_query_rows if row.get("error"))
    out_of_scope_queries = sum(1 for row in per_query_rows if not row.get("in_scope", True) and not row.get("error"))

    raw_hit_rate_at_k = mean(float(row["raw_success_at_k"]) for row in successful_rows) if successful_rows else 0.0
    final_hit_rate_at_k = mean(float(row["final_success_at_k"]) for row in successful_rows) if successful_rows else 0.0

    raw_hit_ranks = [int(row["raw_first_hit_rank"]) for row in successful_rows if int(row["raw_first_hit_rank"]) > 0]
    final_hit_ranks = [int(row["final_first_hit_rank"]) for row in successful_rows if int(row["final_first_hit_rank"]) > 0]
    mean_raw_first_hit_rank = mean(raw_hit_ranks) if raw_hit_ranks else 0.0
    mean_final_first_hit_rank = mean(final_hit_ranks) if final_hit_ranks else 0.0

    return {
        "total_queries": total_queries,
        "successful_queries": successful_queries,
        "failed_queries": failed_queries,
        "out_of_scope_queries": out_of_scope_queries,
        "mean_raw_hit_rate_at_k": raw_hit_rate_at_k,
        "mean_final_hit_rate_at_k": final_hit_rate_at_k,
        "mean_raw_first_hit_rank": mean_raw_first_hit_rank,
        "mean_final_first_hit_rank": mean_final_first_hit_rank,
        "mean_raw_precision_at_k": means["raw_precision_at_k"],
        "mean_raw_recall_at_k": means["raw_recall_at_k"],
        "mean_raw_mrr_at_k": means["raw_mrr_at_k"],
        "mean_raw_ndcg_at_k": means["raw_ndcg_at_k"],
        "mean_final_precision_at_k": means["final_precision_at_k"],
        "mean_final_recall_at_k": means["final_recall_at_k"],
        "mean_final_mrr_at_k": means["final_mrr_at_k"],
        "mean_final_ndcg_at_k": means["final_ndcg_at_k"],
        "delta_final_minus_raw": deltas,
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write per-query results to CSV."""
    if not rows:
        return

    fieldnames = [
        "question_id",
        "question",
        "gold_doc_ids",
        "in_scope",
        "raw_ids",
        "final_ids",
        "raw_hit_count",
        "final_hit_count",
        "raw_success_at_k",
        "final_success_at_k",
        "raw_precision_at_k",
        "raw_recall_at_k",
        "raw_mrr_at_k",
        "raw_ndcg_at_k",
        "final_precision_at_k",
        "final_recall_at_k",
        "final_mrr_at_k",
        "final_ndcg_at_k",
        "raw_first_hit_rank",
        "final_first_hit_rank",
        "error",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = load_jsonl_dataset(args.dataset_path)
    examples = build_examples(dataset_rows)

    per_query_rows: List[Dict[str, Any]] = []
    api_outputs_path = output_dir / "api_outputs.jsonl"

    api_outputs_handle = api_outputs_path.open("w", encoding="utf-8") if args.save_api_outputs else None
    try:
        total = len(examples)
        for idx, example in enumerate(examples, start=1):
            row: Dict[str, Any]
            api_response: Dict[str, Any] = {}
            try:
                api_response = post_retrieve_detailed(
                    api_url=args.api_url,
                    question=example.question,
                    semantic_n=args.semantic_n,
                    final_k=args.final_k,
                    timeout=args.timeout,
                    retries=args.retries,
                    retry_sleep=args.retry_sleep,
                )
                row = evaluate_single_query(example, api_response, eval_k=args.eval_k)
                row["error"] = ""

                if row["in_scope"] and (not json.loads(row["raw_ids"]) or not json.loads(row["final_ids"])):
                    row["error"] = "Empty retrieval list returned for in-scope query"

                print(
                    f"[{idx}/{total}] {example.question_id} "
                    f"in_scope={row['in_scope']} "
                    f"raw(P={row['raw_precision_at_k']:.3f},R={row['raw_recall_at_k']:.3f},"
                    f"MRR={row['raw_mrr_at_k']:.3f},nDCG={row['raw_ndcg_at_k']:.3f},hit_rank={row['raw_first_hit_rank']}) "
                    f"final(P={row['final_precision_at_k']:.3f},R={row['final_recall_at_k']:.3f},"
                    f"MRR={row['final_mrr_at_k']:.3f},nDCG={row['final_ndcg_at_k']:.3f},hit_rank={row['final_first_hit_rank']})"
                )
            except Exception as exc:  # pragma: no cover - runtime safety
                row = {
                    "question_id": example.question_id,
                    "question": example.question,
                    "gold_doc_ids": json.dumps(example.gold_doc_ids, ensure_ascii=False),
                    "in_scope": False,
                    "raw_ids": json.dumps([], ensure_ascii=False),
                    "final_ids": json.dumps([], ensure_ascii=False),
                    "raw_hit_count": 0,
                    "final_hit_count": 0,
                    "raw_success_at_k": 0,
                    "final_success_at_k": 0,
                    "raw_precision_at_k": 0.0,
                    "raw_recall_at_k": 0.0,
                    "raw_mrr_at_k": 0.0,
                    "raw_ndcg_at_k": 0.0,
                    "final_precision_at_k": 0.0,
                    "final_recall_at_k": 0.0,
                    "final_mrr_at_k": 0.0,
                    "final_ndcg_at_k": 0.0,
                    "raw_first_hit_rank": 0,
                    "final_first_hit_rank": 0,
                    "error": str(exc),
                }
                api_response = {"error": str(exc)}

                print(f"[{idx}/{total}] {example.question_id} ERROR: {row['error']}")

            per_query_rows.append(row)

            if api_outputs_handle is not None:
                debug_row = {
                    "question_id": example.question_id,
                    "question": example.question,
                    "api_response": api_response,
                }
                api_outputs_handle.write(json.dumps(debug_row, ensure_ascii=False) + "\n")

    finally:
        if api_outputs_handle is not None:
            api_outputs_handle.close()

    aggregate = aggregate_metrics(per_query_rows)
    summary = {
        **aggregate,
        "run_config": {
            "dataset_path": str(Path(args.dataset_path)),
            "api_url": args.api_url,
            "semantic_n": args.semantic_n,
            "final_k": args.final_k,
            "eval_k": args.eval_k,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    csv_path = output_dir / "per_query_results.csv"
    json_path = output_dir / "aggregate_results.json"
    _write_csv(csv_path, per_query_rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("API RETRIEVAL EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Dataset:      {args.dataset_path}")
    print(f"API URL:      {args.api_url}")
    print(f"Queries:      {summary['total_queries']}")
    print(f"Successful:   {summary['successful_queries']}")
    print(f"Failed:       {summary['failed_queries']}")
    print(f"Out-of-scope: {summary['out_of_scope_queries']}")
    print("\nPrimary metrics (single-gold friendly):")
    print(f"  Raw HitRate@k:        {summary['mean_raw_hit_rate_at_k']:.4f}")
    print(f"  Final HitRate@k:      {summary['mean_final_hit_rate_at_k']:.4f}")
    print(f"  Raw Mean First Hit:   {summary['mean_raw_first_hit_rank']:.4f}")
    print(f"  Final Mean First Hit: {summary['mean_final_first_hit_rank']:.4f}")
    print("\nRaw metrics:")
    #print(f"  Precision@k: {summary['mean_raw_precision_at_k']:.4f}")
    print(f"  Recall@k:    {summary['mean_raw_recall_at_k']:.4f}")
    print(f"  MRR@k:       {summary['mean_raw_mrr_at_k']:.4f}")
    print(f"  nDCG@k:      {summary['mean_raw_ndcg_at_k']:.4f}")
    print("\nFinal metrics:")
    #print(f"  Precision@k: {summary['mean_final_precision_at_k']:.4f}")
    print(f"  Recall@k:    {summary['mean_final_recall_at_k']:.4f}")
    print(f"  MRR@k:       {summary['mean_final_mrr_at_k']:.4f}")
    print(f"  nDCG@k:      {summary['mean_final_ndcg_at_k']:.4f}")
    print("\nDelta (final - raw):")
    for metric_name, metric_value in summary["delta_final_minus_raw"].items():
        print(f"  {metric_name}: {metric_value:+.4f}")
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {json_path}")
    if args.save_api_outputs:
        print(f"Wrote: {api_outputs_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()