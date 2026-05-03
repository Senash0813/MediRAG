#!/usr/bin/env python3
"""Build an answer-evaluation dataset by calling the /query4 API.

Usage:
    python evaluation/generation/llm_judge/code/build_answer_dataset.py \
        --dataset_path evaluation/retrieval/data/eval_dataset_20.jsonl \
        --api_url http://localhost:8003/query4 \
        --output_path evaluation/generation/llm_judge/data/answer_eval_dataset_20.jsonl

This script keeps the same questions as the gold evaluation dataset and stores
only the model's direct answer from /query4 for later similarity evaluation.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


@dataclass(frozen=True)
class SourceExample:
    """Single input example from the gold evaluation dataset."""

    question_id: str
    question: str
    ground_truth_answer: str
    source_paper_id: Optional[str] = None
    source_qa_id: Optional[str] = None
    specialty_tag: Optional[str] = None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build answer evaluation dataset from /query4")
    parser.add_argument(
        "--dataset_path",
        default="evaluation/retrieval/data/eval_dataset_20.jsonl",
        help="Path to gold evaluation dataset",
    )
    parser.add_argument("--api_url", default="http://localhost:8003/query4", help="/query4 API URL")
    parser.add_argument(
        "--output_path",
        default="evaluation/generation/llm_judge/data/answer_eval_dataset_20.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Top-k context size sent to /query4")
    parser.add_argument("--timeout", type=float, default=90.0, help="HTTP timeout in seconds")
    parser.add_argument("--retries", type=int, default=2, help="Retry count for transient failures")
    parser.add_argument("--retry_sleep", type=float, default=2.0, help="Sleep between retries in seconds")
    parser.add_argument("--save_api_outputs", action="store_true", help="Also store raw API responses in a sidecar JSONL")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into memory."""
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
                raise ValueError(f"Expected object at line {line_no} in {dataset_path}")
            rows.append(row)
    return rows


def build_examples(rows: List[Dict[str, Any]]) -> List[SourceExample]:
    """Normalize input rows into typed examples."""
    examples: List[SourceExample] = []
    seen_ids: set[str] = set()

    for index, row in enumerate(rows, start=1):
        question_id = str(row.get("question_id", "")).strip()
        question = str(row.get("question", "")).strip()
        ground_truth_answer = str(row.get("ground_truth_answer", "")).strip()
        source_paper_id = str(row.get("source_paper_id", "")).strip() or None
        source_qa_id = str(row.get("source_qa_id", "")).strip() or None
        specialty_tag = str(row.get("specialty_tag", "")).strip() or None

        if not question_id:
            raise ValueError(f"Missing question_id at dataset row {index}")
        if question_id in seen_ids:
            raise ValueError(f"Duplicate question_id detected: {question_id}")
        if not question:
            raise ValueError(f"Missing question text for question_id={question_id}")
        if not ground_truth_answer:
            raise ValueError(f"Missing ground_truth_answer for question_id={question_id}")

        seen_ids.add(question_id)
        examples.append(
            SourceExample(
                question_id=question_id,
                question=question,
                ground_truth_answer=ground_truth_answer,
                source_paper_id=source_paper_id,
                source_qa_id=source_qa_id,
                specialty_tag=specialty_tag,
            )
        )

    return examples


def post_query4(api_url: str, question: str, top_k: int, timeout: float, retries: int, retry_sleep: float) -> Dict[str, Any]:
    """Call /query4 with retry handling."""
    payload = {
        "query": question,
        "top_k": top_k,
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
        except Exception as exc:  # pragma: no cover - network/runtime resilience
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep)
                continue
            raise RuntimeError(f"Failed to call {api_url} after {retries + 1} attempts: {last_error}") from last_error

    raise RuntimeError(f"Failed to call {api_url}") from last_error


def build_output_row(example: SourceExample, api_response: Dict[str, Any]) -> Dict[str, Any]:
    """Create the output row with question, ground truth, and direct answer only."""
    direct_answer = str(api_response.get("direct_answer", "")).strip()

    return {
        "question_id": example.question_id,
        "question": example.question,
        "ground_truth_answer": example.ground_truth_answer,
        "model_answer": direct_answer,
        "source_paper_id": example.source_paper_id or "",
        "source_qa_id": example.source_qa_id or "",
        "specialty_tag": example.specialty_tag or "",
    }


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    input_rows = load_jsonl(args.dataset_path)
    examples = build_examples(input_rows)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    api_outputs_path = output_path.with_suffix(".api_outputs.jsonl")

    output_rows: List[Dict[str, Any]] = []
    api_outputs_handle = api_outputs_path.open("w", encoding="utf-8") if args.save_api_outputs else None

    try:
        total = len(examples)
        for index, example in enumerate(examples, start=1):
            try:
                api_response = post_query4(
                    api_url=args.api_url,
                    question=example.question,
                    top_k=args.top_k,
                    timeout=args.timeout,
                    retries=args.retries,
                    retry_sleep=args.retry_sleep,
                )
                row = build_output_row(example, api_response)
                row["error"] = ""

                print(f"[{index}/{total}] Question: {row['question']}")
                print(f"[{index}/{total}] Stored direct answer: {row['model_answer']}")
            except Exception as exc:  # pragma: no cover - runtime safety
                row = {
                    "question_id": example.question_id,
                    "question": example.question,
                    "ground_truth_answer": example.ground_truth_answer,
                    "model_answer": "",
                    "source_paper_id": example.source_paper_id or "",
                    "source_qa_id": example.source_qa_id or "",
                    "specialty_tag": example.specialty_tag or "",
                    "error": str(exc),
                }
                api_response = {"error": str(exc)}
                print(f"[{index}/{total}] {example.question_id} ERROR: {row['error']}")

            output_rows.append(row)

            if api_outputs_handle is not None:
                api_outputs_handle.write(
                    json.dumps(
                        {
                            "question_id": example.question_id,
                            "question": example.question,
                            "api_response": api_response,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    finally:
        if api_outputs_handle is not None:
            api_outputs_handle.close()

    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_path": str(Path(args.dataset_path)),
        "api_url": args.api_url,
        "output_path": str(output_path),
        "rows_written": len(output_rows),
        "successful_rows": sum(1 for row in output_rows if not row.get("error")),
        "failed_rows": sum(1 for row in output_rows if row.get("error")),
        "top_k": args.top_k,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    summary_path = output_path.with_suffix(".meta.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("ANSWER DATASET BUILT")
    print("=" * 80)
    print(f"Output:   {output_path}")
    print(f"Summary:  {summary_path}")
    print(f"Rows:     {summary['rows_written']}")
    print(f"Success:  {summary['successful_rows']}")
    print(f"Failed:   {summary['failed_rows']}")
    if args.save_api_outputs:
        print(f"API log:  {api_outputs_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
