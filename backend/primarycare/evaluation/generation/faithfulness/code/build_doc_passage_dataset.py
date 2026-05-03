#!/usr/bin/env python3
"""Build a passage-linked dataset from a faithfulness JSONL dataset.

This script creates a new dataset containing:
- question
- direct_answer
- doc_ids_in_evidence_summary
- passage_text (aligned to doc_ids_in_evidence_summary)

Passages are resolved from resources/selected_specialties.jsonl by matching
paper_id == doc_id.

Usage:
    python evaluation/generation/faithfulness/code/build_doc_passage_dataset.py \
        --input_path evaluation/generation/faithfulness/data/faithfulness_dataset_20.jsonl \
        --corpus_path resources/selected_specialties.jsonl \
        --output_path evaluation/generation/faithfulness/data/faithfulness_with_passages_20.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build dataset with question/direct_answer/doc_ids/passage_text"
    )
    parser.add_argument(
        "--input_path",
        default="evaluation/generation/faithfulness/data/faithfulness_dataset_20.jsonl",
        help="Input faithfulness dataset JSONL path",
    )
    parser.add_argument(
        "--corpus_path",
        default="resources/selected_specialties.jsonl",
        help="Corpus JSONL path containing paper_id and passage_text",
    )
    parser.add_argument(
        "--output_path",
        default="evaluation/generation/faithfulness/data/faithfulness_with_passages_20.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--summary_path",
        default="",
        help="Optional output summary JSON path",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(obj)
    return rows


def normalize_doc_ids(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    return []


def collect_needed_doc_ids(rows: Iterable[Dict[str, Any]]) -> Set[str]:
    needed: Set[str] = set()
    for row in rows:
        for doc_id in normalize_doc_ids(row.get("doc_ids_in_evidence_summary", [])):
            needed.add(doc_id)
    return needed


def maybe_resolve_corpus_path(path: Path) -> Path:
    if path.is_file():
        return path

    # Handle common spelling variant in user messages.
    alt = Path(str(path).replace("specialities", "specialties"))
    if alt.is_file():
        return alt

    return path


def parse_corpus_line(raw_line: str) -> Dict[str, Any] | None:
    """Parse a single corpus JSONL line into a dict, returning None on failure."""
    line = raw_line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def extract_doc_and_passage(obj: Dict[str, Any]) -> tuple[str, str]:
    """Extract normalized paper_id and passage_text from a corpus object."""
    paper_id = str(obj.get("paper_id", "")).strip()
    passage_text = str(obj.get("passage_text", "")).strip()
    return paper_id, passage_text


def build_doc_to_passage(corpus_path: Path, needed_doc_ids: Set[str]) -> Dict[str, str]:
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    doc_to_passage: Dict[str, str] = {}

    with corpus_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            obj = parse_corpus_line(raw_line)
            if obj is None:
                continue

            paper_id, passage_text = extract_doc_and_passage(obj)
            if not paper_id or paper_id not in needed_doc_ids or not passage_text:
                continue

            # Keep the first non-empty passage for each paper_id for deterministic output.
            doc_to_passage.setdefault(paper_id, passage_text)

            if len(doc_to_passage) == len(needed_doc_ids):
                break

    return doc_to_passage


def build_output_rows(
    source_rows: List[Dict[str, Any]],
    doc_to_passage: Dict[str, str],
) -> List[Dict[str, Any]]:
    output_rows: List[Dict[str, Any]] = []

    for row in source_rows:
        doc_ids = normalize_doc_ids(row.get("doc_ids_in_evidence_summary", []))
        passages = [doc_to_passage.get(doc_id, "") for doc_id in doc_ids]
        missing_doc_ids = [doc_id for doc_id in doc_ids if not doc_to_passage.get(doc_id)]

        output_rows.append(
            {
                "question_id": str(row.get("question_id", "")).strip(),
                "question": str(row.get("question", "")).strip(),
                "direct_answer": str(row.get("direct_answer", "")).strip(),
                "doc_ids_in_evidence_summary": doc_ids,
                "passage_text": passages,
                "missing_doc_ids": missing_doc_ids,
                "error": "" if not missing_doc_ids else f"Missing passages for: {missing_doc_ids}",
            }
        )

    return output_rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_path)
    corpus_path = maybe_resolve_corpus_path(Path(args.corpus_path))
    output_path = Path(args.output_path)

    source_rows = load_jsonl(input_path)
    needed_doc_ids = collect_needed_doc_ids(source_rows)
    doc_to_passage = build_doc_to_passage(corpus_path, needed_doc_ids)
    output_rows = build_output_rows(source_rows, doc_to_passage)

    write_jsonl(output_path, output_rows)

    total = len(output_rows)
    missing_rows = sum(1 for row in output_rows if row.get("missing_doc_ids"))
    found_doc_count = len(doc_to_passage)

    print("=" * 80)
    print("PASSAGE DATASET BUILT")
    print("=" * 80)
    print(f"Input:            {input_path}")
    print(f"Corpus:           {corpus_path}")
    print(f"Output:           {output_path}")
    print(f"Rows:             {total}")
    print(f"Rows with misses: {missing_rows}")
    print(f"Resolved doc IDs: {found_doc_count}/{len(needed_doc_ids)}")
    print("=" * 80)

    summary_path = Path(args.summary_path) if args.summary_path else output_path.with_suffix(".meta.json")
    summary = {
        "input_path": str(input_path),
        "corpus_path": str(corpus_path),
        "output_path": str(output_path),
        "rows_written": total,
        "rows_with_missing_doc_ids": missing_rows,
        "resolved_doc_ids": found_doc_count,
        "requested_doc_ids": len(needed_doc_ids),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Summary:          {summary_path}")


if __name__ == "__main__":
    main()
