
#!/usr/bin/env python3
"""Evaluate faithfulness of direct_answer against passage_text evidence.

Usage:
    python evaluation/generation/faithfulness/code/evaluate_faithfulness_from_passages.py \
        --dataset_path evaluation/generation/faithfulness/data/faithfulness_with_passages_20.jsonl \
        --model phi:2.7b \
        --output_dir evaluation/generation/faithfulness/results

Expected input row fields:
- question
- direct_answer
- doc_ids_in_evidence_summary
- passage_text

The script checks whether the direct answer is grounded in the provided
passage_text evidence using an LLM judge through Ollama.
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
class FaithfulnessExample:
    question_id: str
    question: str
    direct_answer: str
    doc_ids: List[str]
    passages: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate answer faithfulness against passage evidence")
    parser.add_argument(
        "--dataset_path",
        default="evaluation/generation/faithfulness/data/faithfulness_with_passages_20.jsonl",
        help="Path to dataset with direct_answer and passage_text",
    )
    parser.add_argument(
        "--output_dir",
        default="evaluation/generation/faithfulness/results",
        help="Output directory for result files",
    )
    parser.add_argument("--model", default="phi:2.7b", help="Ollama model name")
    parser.add_argument(
        "--ollama_url",
        default="http://127.0.0.1:11434/api/generate",
        help="Ollama generate endpoint",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds")
    parser.add_argument("--retries", type=int, default=2, help="Retries per query")
    parser.add_argument("--retry_sleep", type=float, default=2.0, help="Seconds between retries")
    parser.add_argument(
        "--pass_threshold",
        type=float,
        default=6.0,
        help="Overall faithfulness score threshold for PASS",
    )
    parser.add_argument(
        "--max_context_chars",
        type=int,
        default=12000,
        help="Maximum evidence context characters sent to the judge",
    )
    return parser.parse_args()


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    rows: List[Dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            content = line.strip()
            if not content:
                continue
            try:
                row = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no} in {dataset_path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at line {line_no} in {dataset_path}")
            rows.append(row)
    return rows


def ensure_str_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    return []


def build_examples(rows: List[Dict[str, Any]]) -> List[FaithfulnessExample]:
    examples: List[FaithfulnessExample] = []
    seen_ids: set[str] = set()

    for idx, row in enumerate(rows, start=1):
        qid = str(row.get("question_id", "")).strip() or f"row_{idx}"
        question = str(row.get("question", "")).strip()
        direct_answer = str(row.get("direct_answer", "")).strip()
        doc_ids = ensure_str_list(row.get("doc_ids_in_evidence_summary", []))
        passages = ensure_str_list(row.get("passage_text", []))

        if qid in seen_ids:
            raise ValueError(f"Duplicate question_id detected: {qid}")
        if not question:
            raise ValueError(f"Missing question at row {idx}")

        seen_ids.add(qid)
        examples.append(
            FaithfulnessExample(
                question_id=qid,
                question=question,
                direct_answer=direct_answer,
                doc_ids=doc_ids,
                passages=passages,
            )
        )

    return examples


def build_evidence_blocks(doc_ids: List[str], passages: List[str], max_context_chars: int) -> str:
    blocks: List[str] = []
    total_chars = 0

    for idx, passage in enumerate(passages):
        doc_id = doc_ids[idx] if idx < len(doc_ids) else f"unknown_{idx + 1}"
        block = f"[DOC_ID: {doc_id}]\n{passage.strip()}"
        if not passage.strip():
            continue

        if total_chars + len(block) > max_context_chars:
            remaining = max_context_chars - total_chars
            if remaining > 120:
                block = block[:remaining]
                blocks.append(block)
            break

        blocks.append(block)
        total_chars += len(block)

    return "\n\n".join(blocks)


def build_judge_prompt(example: FaithfulnessExample, evidence_text: str) -> str:
    return (
        "You are a strict medical RAG faithfulness evaluator.\n"
        "Judge whether DIRECT_ANSWER is fully supported by EVIDENCE passages.\n\n"
        "Scoring policy:\n"
        "- groundedness_score (0-10): Are answer claims directly supported by evidence?\n"
        "- coverage_score (0-10): Does evidence cover major answer claims?\n"
        "- overall_faithfulness_score (0-10): prioritize groundedness over coverage.\n"
        "- verdict: PASS or FAIL.\n"
        "- unsupported_claims: short list of unsupported claims (empty list if none).\n"
        "- rationale: max 50 words.\n"
        "Return exactly these JSON keys: groundedness_score, coverage_score, overall_faithfulness_score, verdict, unsupported_claims, rationale.\n"
        "Return JSON only. No markdown.\n\n"
        f"QUESTION:\n{example.question}\n\n"
        f"DIRECT_ANSWER:\n{example.direct_answer}\n\n"
        f"EVIDENCE:\n{evidence_text}\n"
    )


def parse_judge_json(text: str) -> Dict[str, Any]:
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

        cleaned = re.sub(r",\s*([}\]])", r"\1", snippet)
        try:
            obj = ast.literal_eval(cleaned)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    raise ValueError("Could not parse judge JSON")


def normalize_judge_output(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common key variants produced by local LLM judges."""
    normalized = dict(parsed)

    aliases = {
        "groundedness_score": ["groundedness", "support_score", "faithfulness_score"],
        "coverage_score": ["evidence_coverage", "coverage"],
        "overall_faithfulness_score": ["overall_score", "faithfulness_score", "score"],
        "verdict": ["label", "decision", "pass_fail"],
        "unsupported_claims": ["unsupported", "unsupported_statements", "unsupported_points"],
        "rationale": ["explanation", "reason", "justification"],
    }

    for canonical_key, candidate_keys in aliases.items():
        if canonical_key in normalized and normalized[canonical_key] not in (None, "", []):
            continue
        for candidate_key in candidate_keys:
            if candidate_key in normalized and normalized[candidate_key] not in (None, "", []):
                normalized[canonical_key] = normalized[candidate_key]
                break

    verdict = str(normalized.get("verdict", "")).strip().upper()
    if verdict in {"TRUE", "SUPPORTED", "SUPPORTED_BY_EVIDENCE", "YES", "PASS"}:
        normalized["verdict"] = "PASS"
    elif verdict in {"FALSE", "UNSUPPORTED", "NO", "FAIL"}:
        normalized["verdict"] = "FAIL"

    return normalized


def extract_judge_payload(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the most likely judge payload from nested response shapes."""
    if not isinstance(parsed, dict):
        raise ValueError("Judge payload must be a JSON object")

    if any(key in parsed for key in ("groundedness_score", "coverage_score", "overall_faithfulness_score", "verdict", "rationale")):
        return parsed

    for nested_key in ("scored_text", "result", "data", "payload", "output"):
        nested_value = parsed.get(nested_key)
        if isinstance(nested_value, dict):
            extracted = extract_judge_payload(nested_value)
            if extracted:
                return extracted

    return parsed


def is_valid_judge_payload(parsed: Dict[str, Any]) -> bool:
    """Return True only if the payload looks like a real faithfulness judge result."""
    if not isinstance(parsed, dict):
        return False
    return any(key in parsed for key in ("groundedness_score", "coverage_score", "overall_faithfulness_score", "verdict", "rationale"))


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
            parsed = extract_judge_payload(parse_judge_json(response_text))
            parsed = normalize_judge_output(parsed)
            parsed["raw_response"] = response_text
            if not is_valid_judge_payload(parsed):
                raise ValueError(f"Unusable judge response: {response_text[:500]}")
            return parsed
        except Exception as exc:  # pragma: no cover
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep)
                continue
            raise RuntimeError(f"Judge call failed after {retries + 1} attempts: {last_error}") from last_error

    raise RuntimeError(f"Judge call failed: {last_error}")


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


def to_short_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    return []


def aggregate(rows: List[Dict[str, Any]], pass_threshold: float) -> Dict[str, Any]:
    successful = [r for r in rows if not r.get("error")]
    failed = [r for r in rows if r.get("error")]

    if successful:
        mean_groundedness = mean(float(r["groundedness_score"]) for r in successful)
        mean_coverage = mean(float(r["coverage_score"]) for r in successful)
        mean_overall = mean(float(r["overall_faithfulness_score"]) for r in successful)
        avg_unsupported_claims = mean(len(r.get("unsupported_claims", [])) for r in successful)
    else:
        mean_groundedness = 0.0
        mean_coverage = 0.0
        mean_overall = 0.0
        avg_unsupported_claims = 0.0

    pass_count = sum(1 for r in successful if float(r["overall_faithfulness_score"]) >= pass_threshold)

    return {
        "total_rows": len(rows),
        "successful_rows": len(successful),
        "failed_rows": len(failed),
        "mean_groundedness_score": mean_groundedness,
        "mean_coverage_score": mean_coverage,
        "mean_overall_faithfulness_score": mean_overall,
        "avg_unsupported_claims": avg_unsupported_claims,
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
        "direct_answer",
        "doc_ids_in_evidence_summary",
        "groundedness_score",
        "coverage_score",
        "overall_faithfulness_score",
        "verdict",
        "judge_verdict_raw",
        "unsupported_claims",
        "rationale",
        "raw_response",
        "error",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            safe_row = dict(row)
            safe_row["doc_ids_in_evidence_summary"] = json.dumps(row.get("doc_ids_in_evidence_summary", []), ensure_ascii=False)
            safe_row["unsupported_claims"] = json.dumps(row.get("unsupported_claims", []), ensure_ascii=False)
            writer.writerow({key: safe_row.get(key, "") for key in fieldnames})


def evaluate_one_example(example: FaithfulnessExample, args: argparse.Namespace) -> Dict[str, Any]:
    if not example.direct_answer:
        return {
            "question_id": example.question_id,
            "question": example.question,
            "direct_answer": example.direct_answer,
            "doc_ids_in_evidence_summary": example.doc_ids,
            "groundedness_score": 0.0,
            "coverage_score": 0.0,
            "overall_faithfulness_score": 0.0,
            "verdict": "ERROR",
            "judge_verdict_raw": "",
            "unsupported_claims": ["direct_answer is empty"],
            "rationale": "Cannot evaluate faithfulness because direct_answer is empty.",
            "error": "direct_answer is empty",
        }

    evidence_text = build_evidence_blocks(example.doc_ids, example.passages, args.max_context_chars)
    if not evidence_text:
        return {
            "question_id": example.question_id,
            "question": example.question,
            "direct_answer": example.direct_answer,
            "doc_ids_in_evidence_summary": example.doc_ids,
            "groundedness_score": 0.0,
            "coverage_score": 0.0,
            "overall_faithfulness_score": 0.0,
            "verdict": "ERROR",
            "judge_verdict_raw": "",
            "unsupported_claims": ["no passage_text evidence available"],
            "rationale": "Cannot evaluate faithfulness because evidence is missing.",
            "error": "no passage_text evidence available",
        }

    judge_raw = call_ollama_judge(
        ollama_url=args.ollama_url,
        model=args.model,
        prompt=build_judge_prompt(example, evidence_text),
        timeout=args.timeout,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
    )

    if not any(key in judge_raw for key in ["groundedness_score", "coverage_score", "overall_faithfulness_score", "verdict", "rationale"]):
        raw_response = str(judge_raw.get("raw_response", "")).strip()
        raise ValueError(
            f"Judge response did not contain expected fields. Raw response: {raw_response[:500]}"
        )

    groundedness = clamp_score(judge_raw.get("groundedness_score"))
    coverage = clamp_score(judge_raw.get("coverage_score"))
    overall = clamp_score(judge_raw.get("overall_faithfulness_score", (0.7 * groundedness + 0.3 * coverage)))
    judge_verdict_raw = str(judge_raw.get("verdict", "")).strip().upper()
    verdict = "PASS" if overall >= args.pass_threshold else "FAIL"
    unsupported_claims = to_short_list(judge_raw.get("unsupported_claims", []))
    rationale = str(judge_raw.get("rationale", "")).strip()
    raw_response = str(judge_raw.get("raw_response", "")).strip()

    if not judge_verdict_raw and not rationale and not raw_response:
        raise ValueError("Judge output was empty or unusable")

    return {
        "question_id": example.question_id,
        "question": example.question,
        "direct_answer": example.direct_answer,
        "doc_ids_in_evidence_summary": example.doc_ids,
        "groundedness_score": groundedness,
        "coverage_score": coverage,
        "overall_faithfulness_score": overall,
        "verdict": verdict,
        "judge_verdict_raw": judge_verdict_raw,
        "unsupported_claims": unsupported_claims,
        "rationale": rationale,
        "raw_response": raw_response,
        "error": "",
    }


def main() -> None:
    args = parse_args()

    rows = load_jsonl(args.dataset_path)
    examples = build_examples(rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_query_rows: List[Dict[str, Any]] = []

    total = len(examples)
    for idx, example in enumerate(examples, start=1):
        try:
            row = evaluate_one_example(example, args)
            if row.get("error"):
                print(f"[{idx}/{total}] {example.question_id} ERROR: {row['error']}")
            else:
                print(
                    f"[{idx}/{total}] {example.question_id} "
                    f"grounded={row['groundedness_score']:.1f} "
                    f"coverage={row['coverage_score']:.1f} "
                    f"overall={row['overall_faithfulness_score']:.1f} "
                    f"verdict={row['verdict']}"
                )
        except Exception as exc:  # pragma: no cover
            row = {
                "question_id": example.question_id,
                "question": example.question,
                "direct_answer": example.direct_answer,
                "doc_ids_in_evidence_summary": example.doc_ids,
                "groundedness_score": 0.0,
                "coverage_score": 0.0,
                "overall_faithfulness_score": 0.0,
                "verdict": "ERROR",
                "judge_verdict_raw": "",
                "unsupported_claims": [],
                "rationale": "",
                "raw_response": "",
                "error": str(exc),
            }
            print(f"[{idx}/{total}] {example.question_id} ERROR: {row['error']}")

        per_query_rows.append(row)

    summary = aggregate(per_query_rows, pass_threshold=args.pass_threshold)
    summary["run_config"] = {
        "dataset_path": str(Path(args.dataset_path)),
        "model": args.model,
        "ollama_url": args.ollama_url,
        "timeout": args.timeout,
        "retries": args.retries,
        "pass_threshold": args.pass_threshold,
        "max_context_chars": args.max_context_chars,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    per_query_csv = out_dir / "faithfulness_from_passages_per_query.csv"
    per_query_json = out_dir / "faithfulness_from_passages_per_query.json"
    summary_json = out_dir / "faithfulness_from_passages_summary.json"

    write_per_query_csv(per_query_csv, per_query_rows)
    with per_query_json.open("w", encoding="utf-8") as handle:
        json.dump(per_query_rows, handle, indent=2, ensure_ascii=False)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("FAITHFULNESS EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Dataset:          {args.dataset_path}")
    print(f"Model:            {args.model}")
    print(f"Total rows:       {summary['total_rows']}")
    print(f"Successful:       {summary['successful_rows']}")
    print(f"Failed:           {summary['failed_rows']}")
    print(f"Mean grounded:    {summary['mean_groundedness_score']:.3f}")
    print(f"Mean coverage:    {summary['mean_coverage_score']:.3f}")
    print(f"Mean overall:     {summary['mean_overall_faithfulness_score']:.3f}")
    print(f"Avg unsupported:  {summary['avg_unsupported_claims']:.3f}")
    print(f"Pass rate:        {summary['pass_rate']:.3f}")
    print(f"Wrote:            {per_query_csv}")
    print(f"Wrote:            {per_query_json}")
    print(f"Wrote:            {summary_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
