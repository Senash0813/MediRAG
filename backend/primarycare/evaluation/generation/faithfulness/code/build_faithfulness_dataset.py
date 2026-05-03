#!/usr/bin/env python3
"""Build an intermediate dataset for faithfulness evaluation.

This script calls /retrieve/detailed for each question and stores:
- question
- evidence_summary (answer text to be checked)
- phase2_doc_ids (retrieved validated doc IDs)
- doc IDs mentioned in evidence_summary text (if any)

Usage:
    python evaluation/generation/faithfulness/code/build_faithfulness_dataset.py \
        --dataset_path evaluation/retrieval/data/eval_dataset_20.jsonl \
        --api_url http://localhost:8003/retrieve/detailed \
        --output_path evaluation/generation/faithfulness/data/faithfulness_dataset_20.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


@dataclass(frozen=True)
class EvalQuestion:
    question_id: str
    question: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build faithfulness dataset from /retrieve/detailed")
    parser.add_argument(
        "--dataset_path",
        default="evaluation/retrieval/data/eval_dataset_20.jsonl",
        help="Input question dataset JSONL",
    )
    parser.add_argument("--api_url", default="http://localhost:8003/retrieve/detailed", help="Detailed API URL")
    parser.add_argument(
        "--output_path",
        default="evaluation/generation/faithfulness/data/faithfulness_dataset_20.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--semantic_n", type=int, default=10, help="semantic_n for /retrieve/detailed")
    parser.add_argument("--final_k", type=int, default=5, help="final_k for /retrieve/detailed")
    parser.add_argument("--timeout", type=float, default=90.0, help="HTTP timeout seconds")
    parser.add_argument("--retries", type=int, default=2, help="Retries per query")
    parser.add_argument("--retry_sleep", type=float, default=2.0, help="Sleep between retries")
    parser.add_argument("--save_api_outputs", action="store_true", help="Save full API responses to sidecar JSONL")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")

    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no} in {p}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at line {line_no} in {p}")
            rows.append(obj)
    return rows


def build_questions(rows: List[Dict[str, Any]]) -> List[EvalQuestion]:
    out: List[EvalQuestion] = []
    seen: set[str] = set()

    for idx, row in enumerate(rows, start=1):
        qid = str(row.get("question_id", "")).strip() or f"row_{idx}"
        question = str(row.get("question", "")).strip()
        if not question:
            raise ValueError(f"Missing question at row {idx}")
        if qid in seen:
            raise ValueError(f"Duplicate question_id: {qid}")
        seen.add(qid)
        out.append(EvalQuestion(question_id=qid, question=question))

    return out


def post_retrieve_detailed(
    *,
    api_url: str,
    question: str,
    semantic_n: int,
    final_k: int,
    timeout: float,
    retries: int,
    retry_sleep: float,
) -> Dict[str, Any]:
    payload = {
        "query": question,
        "semantic_n": semantic_n,
        "final_k": final_k,
    }

    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(api_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            js = resp.json()
            if not isinstance(js, dict):
                raise ValueError("API response must be an object")
            return js
        except Exception as exc:  # pragma: no cover
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep)
                continue
            raise RuntimeError(f"Failed to call {api_url} after {retries + 1} attempts: {last_error}") from last_error

    raise RuntimeError(f"Failed to call {api_url}: {last_error}")


def dedup_phase2_doc_ids(phase2_docs: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for doc in phase2_docs or []:
        if not isinstance(doc, dict):
            continue
        pid = str(doc.get("paper_id", "")).strip()
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
    return out


def extract_doc_ids_from_text(text: str) -> List[str]:
    """Extract probable doc IDs referenced in text.

    Supports patterns like:
    - DOC_ID: 12345
    - [DOC_ID: 12345]
    - DOC 12345
    - [12345] (only if >=5 digits to avoid noise)
    """
    if not text:
        return []

    patterns = [
        r"DOC_ID\s*[:#-]?\s*([A-Za-z0-9_-]+)",
        r"\bDOC\s+([0-9]{5,})\b",
        r"\[([0-9]{5,})\]",
    ]

    found: List[str] = []
    seen: set[str] = set()
    for pat in patterns:
        for match in re.findall(pat, text, flags=re.IGNORECASE):
            value = str(match).strip()
            if value and value not in seen:
                seen.add(value)
                found.append(value)
    return found


def main() -> None:
    args = parse_args()

    in_rows = load_jsonl(args.dataset_path)
    questions = build_questions(in_rows)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    api_dump_path = output_path.with_suffix(".api_outputs.jsonl")
    api_dump_handle = api_dump_path.open("w", encoding="utf-8") if args.save_api_outputs else None

    out_rows: List[Dict[str, Any]] = []

    try:
        total = len(questions)
        for idx, q in enumerate(questions, start=1):
            try:
                detailed = post_retrieve_detailed(
                    api_url=args.api_url,
                    question=q.question,
                    semantic_n=args.semantic_n,
                    final_k=args.final_k,
                    timeout=args.timeout,
                    retries=args.retries,
                    retry_sleep=args.retry_sleep,
                )

                in_scope = bool(detailed.get("scope_gate", {}).get("in_scope", False))
                phase2_docs = detailed.get("phase2_validation", []) or []
                phase2_doc_ids = dedup_phase2_doc_ids(phase2_docs)

                final_answer = detailed.get("final_answer", {}) or {}
                evidence_summary = str(final_answer.get("evidence_summary", "")).strip()
                direct_answer = str(final_answer.get("direct_answer", "")).strip()
                limitations = str(final_answer.get("limitations", "")).strip()

                evidence_doc_ids = extract_doc_ids_from_text(evidence_summary)

                row = {
                    "question_id": q.question_id,
                    "question": q.question,
                    "in_scope": in_scope,
                    "evidence_summary": evidence_summary,
                    "direct_answer": direct_answer,
                    "limitations": limitations,
                    "phase2_doc_ids": phase2_doc_ids,
                    "doc_ids_in_evidence_summary": evidence_doc_ids,
                    "error": "",
                }

                print(
                    f"[{idx}/{total}] {q.question_id} "
                    f"phase2_docs={len(phase2_doc_ids)} evidence_doc_ids={len(evidence_doc_ids)}"
                )

            except Exception as exc:  # pragma: no cover
                detailed = {"error": str(exc)}
                row = {
                    "question_id": q.question_id,
                    "question": q.question,
                    "in_scope": False,
                    "evidence_summary": "",
                    "direct_answer": "",
                    "limitations": "",
                    "phase2_doc_ids": [],
                    "doc_ids_in_evidence_summary": [],
                    "error": str(exc),
                }
                print(f"[{idx}/{total}] {q.question_id} ERROR: {row['error']}")

            out_rows.append(row)

            if api_dump_handle is not None:
                api_dump_handle.write(
                    json.dumps(
                        {
                            "question_id": q.question_id,
                            "question": q.question,
                            "detailed_response": detailed,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    finally:
        if api_dump_handle is not None:
            api_dump_handle.close()

    with output_path.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_path": str(Path(args.dataset_path)),
        "output_path": str(output_path),
        "api_url": args.api_url,
        "semantic_n": args.semantic_n,
        "final_k": args.final_k,
        "rows_written": len(out_rows),
        "successful_rows": sum(1 for row in out_rows if not row.get("error")),
        "failed_rows": sum(1 for row in out_rows if row.get("error")),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    summary_path = output_path.with_suffix(".meta.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("FAITHFULNESS DATASET BUILT")
    print("=" * 80)
    print(f"Output:   {output_path}")
    print(f"Summary:  {summary_path}")
    print(f"Rows:     {summary['rows_written']}")
    print(f"Success:  {summary['successful_rows']}")
    print(f"Failed:   {summary['failed_rows']}")
    if args.save_api_outputs:
        print(f"API log:  {api_dump_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
