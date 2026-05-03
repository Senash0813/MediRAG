#!/usr/bin/env python3
"""LLM-as-judge answer evaluation using Ollama.

Usage:
    python evaluation/generation/llm_judge/code/evaluate_answers_llm_judge.py \
        --dataset_path evaluation/generation/llm_judge/data/answer_eval_dataset_20.jsonl \
        --model phi:2.7b \
        --output_dir evaluation/generation/llm_judge/results

Input rows are expected to include:
- question
- ground_truth_answer
- model_answer

The judge returns JSON with:
- similarity_score (0-10)
- factual_alignment_score (0-10)
- overall_score (0-10)
- verdict (PASS/FAIL)
- rationale
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

import requests


@dataclass(frozen=True)
class AnswerExample:
    question_id: str
    question: str
    ground_truth_answer: str
    model_answer: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate answers with LLM judge via Ollama")
    parser.add_argument(
        "--dataset_path",
        default="evaluation/generation/llm_judge/data/answer_eval_dataset_20.jsonl",
        help="Path to answer evaluation dataset",
    )
    parser.add_argument(
        "--output_dir",
        default="evaluation/generation/llm_judge/results",
        help="Output directory for result files",
    )
    parser.add_argument("--model", default="phi:2.7b", help="Ollama model name")
    parser.add_argument("--ollama_url", default="http://127.0.0.1:11434/api/generate", help="Ollama generate endpoint")
    parser.add_argument("--timeout", type=float, default=90.0, help="Request timeout in seconds")
    parser.add_argument("--retries", type=int, default=2, help="Retries per query")
    parser.add_argument("--retry_sleep", type=float, default=2.0, help="Seconds between retries")
    parser.add_argument("--pass_threshold", type=float, default=6.0, help="Overall score threshold for PASS")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
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


def build_examples(rows: List[Dict[str, Any]]) -> List[AnswerExample]:
    examples: List[AnswerExample] = []
    seen_ids: set[str] = set()

    for idx, row in enumerate(rows, start=1):
        qid = str(row.get("question_id", "")).strip() or f"row_{idx}"
        question = str(row.get("question", "")).strip()
        ground_truth = str(row.get("ground_truth_answer", "")).strip()
        model_answer = str(row.get("model_answer", "")).strip()

        if qid in seen_ids:
            raise ValueError(f"Duplicate question_id detected: {qid}")
        if not question:
            raise ValueError(f"Missing question at row {idx}")
        if not ground_truth:
            raise ValueError(f"Missing ground_truth_answer at row {idx}")

        seen_ids.add(qid)
        examples.append(
            AnswerExample(
                question_id=qid,
                question=question,
                ground_truth_answer=ground_truth,
                model_answer=model_answer,
            )
        )

    return examples


def build_judge_prompt(example: AnswerExample) -> str:
    return (
        "You are a strict medical QA evaluator. Compare the model answer to the ground truth answer. "
        "Score semantic similarity and factual alignment.\n\n"
        "Rules:\n"
        "1) Use only these outputs as JSON.\n"
        "2) similarity_score: 0-10, how semantically close the model answer is to ground truth.\n"
        "3) factual_alignment_score: 0-10, how factually consistent the model answer is with ground truth.\n"
        "4) overall_score: 0-10, weighted average emphasizing factual correctness.\n"
        "5) verdict: PASS or FAIL.\n"
        "6) rationale: <= 40 words.\n"
        "7) Return JSON only, no markdown.\n\n"
        f"QUESTION:\n{example.question}\n\n"
        f"GROUND_TRUTH_ANSWER:\n{example.ground_truth_answer}\n\n"
        f"MODEL_ANSWER:\n{example.model_answer}\n"
    )


def call_ollama_judge(
    *,
    ollama_url: str,
    model: str,
    prompt: str,
    timeout: float,
    retries: int,
    retry_sleep: float,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0,
        },
    }

    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(ollama_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            response_text = str(data.get("response", "")).strip()
            if not response_text:
                raise ValueError("Empty response text from Ollama")
            return parse_judge_json(response_text)
        except Exception as exc:  # pragma: no cover
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep)
                continue
            raise RuntimeError(f"Judge call failed after {retries + 1} attempts: {last_error}") from last_error

    raise RuntimeError(f"Judge call failed: {last_error}")


def parse_judge_json(text: str) -> Dict[str, Any]:
    """Parse judge output, allowing minor text around JSON payload."""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start : end + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        # Fallback for pseudo-JSON with single quotes/trailing commas.
        cleaned = re.sub(r",\s*([}\]])", r"\1", snippet)
        try:
            obj = ast.literal_eval(cleaned)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    raise ValueError("Could not parse judge JSON")


def clamp_score(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        x = default
    if x < 0:
        return 0.0
    if x > 10:
        return 10.0
    return x


def aggregate(rows: List[Dict[str, Any]], pass_threshold: float) -> Dict[str, Any]:
    successful = [r for r in rows if not r.get("error")]
    failed = [r for r in rows if r.get("error")]

    if successful:
        mean_similarity = mean(float(r["similarity_score"]) for r in successful)
        mean_factual = mean(float(r["factual_alignment_score"]) for r in successful)
        mean_overall = mean(float(r["overall_score"]) for r in successful)
    else:
        mean_similarity = 0.0
        mean_factual = 0.0
        mean_overall = 0.0

    pass_count = sum(1 for r in successful if float(r["overall_score"]) >= pass_threshold)

    return {
        "total_rows": len(rows),
        "successful_rows": len(successful),
        "failed_rows": len(failed),
        "mean_similarity_score": mean_similarity,
        "mean_factual_alignment_score": mean_factual,
        "mean_overall_score": mean_overall,
        "pass_threshold": pass_threshold,
        "pass_count": pass_count,
        "pass_rate": (pass_count / len(successful)) if successful else 0.0,
    }


def write_per_query_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = [
        "question_id",
        "question",
        "ground_truth_answer",
        "model_answer",
        "similarity_score",
        "factual_alignment_score",
        "overall_score",
        "verdict",
        "rationale",
        "error",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    args = parse_args()

    rows = load_jsonl(args.dataset_path)
    examples = build_examples(rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_query_rows: List[Dict[str, Any]] = []

    total = len(examples)
    for idx, ex in enumerate(examples, start=1):
        try:
            judge_raw = call_ollama_judge(
                ollama_url=args.ollama_url,
                model=args.model,
                prompt=build_judge_prompt(ex),
                timeout=args.timeout,
                retries=args.retries,
                retry_sleep=args.retry_sleep,
            )

            similarity = clamp_score(judge_raw.get("similarity_score"))
            factual = clamp_score(judge_raw.get("factual_alignment_score"))
            overall = clamp_score(judge_raw.get("overall_score", (0.4 * similarity + 0.6 * factual)))
            judge_verdict_raw = str(judge_raw.get("verdict", "")).strip().upper()
            verdict = "PASS" if overall >= args.pass_threshold else "FAIL"
            rationale = str(judge_raw.get("rationale", "")).strip()

            row = {
                "question_id": ex.question_id,
                "question": ex.question,
                "ground_truth_answer": ex.ground_truth_answer,
                "model_answer": ex.model_answer,
                "similarity_score": similarity,
                "factual_alignment_score": factual,
                "overall_score": overall,
                "verdict": verdict,
                "judge_verdict_raw": judge_verdict_raw,
                "rationale": rationale,
                "error": "",
            }

            print(
                f"[{idx}/{total}] {ex.question_id} "
                f"sim={similarity:.1f} fact={factual:.1f} overall={overall:.1f} verdict={verdict}"
            )
        except Exception as exc:  # pragma: no cover
            row = {
                "question_id": ex.question_id,
                "question": ex.question,
                "ground_truth_answer": ex.ground_truth_answer,
                "model_answer": ex.model_answer,
                "similarity_score": 0.0,
                "factual_alignment_score": 0.0,
                "overall_score": 0.0,
                "verdict": "ERROR",
                "judge_verdict_raw": "",
                "rationale": "",
                "error": str(exc),
            }
            print(f"[{idx}/{total}] {ex.question_id} ERROR: {row['error']}")

        per_query_rows.append(row)

    summary = aggregate(per_query_rows, pass_threshold=args.pass_threshold)
    summary["run_config"] = {
        "dataset_path": str(Path(args.dataset_path)),
        "model": args.model,
        "ollama_url": args.ollama_url,
        "timeout": args.timeout,
        "retries": args.retries,
        "pass_threshold": args.pass_threshold,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    per_query_csv = out_dir / "answer_judge_per_query.csv"
    per_query_json = out_dir / "answer_judge_per_query.json"
    summary_json = out_dir / "answer_judge_summary.json"

    write_per_query_csv(per_query_csv, per_query_rows)
    with per_query_json.open("w", encoding="utf-8") as handle:
        json.dump(per_query_rows, handle, indent=2, ensure_ascii=False)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("LLM JUDGE EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Dataset:      {args.dataset_path}")
    print(f"Model:        {args.model}")
    print(f"Total rows:   {summary['total_rows']}")
    print(f"Successful:   {summary['successful_rows']}")
    print(f"Failed:       {summary['failed_rows']}")
    print(f"Mean sim:     {summary['mean_similarity_score']:.3f}")
    print(f"Mean factual: {summary['mean_factual_alignment_score']:.3f}")
    print(f"Mean overall: {summary['mean_overall_score']:.3f}")
    print(f"Pass rate:    {summary['pass_rate']:.3f}")
    print(f"Wrote:        {per_query_csv}")
    print(f"Wrote:        {per_query_json}")
    print(f"Wrote:        {summary_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
