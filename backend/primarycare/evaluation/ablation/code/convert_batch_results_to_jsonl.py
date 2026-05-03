#!/usr/bin/env python3
"""Convert messy batch-results text into clean JSONL.

This parser is tolerant to malformed JSON-like text where object commas
or key order may be broken, as long as each item still contains:
- "question"
- "no_validation_answer"
- "static_prompt_answer"
- "answer"

Usage:
    python evaluation/ablation/code/convert_batch_results_to_jsonl.py \
        --input_file evaluation/ablation/data/raw_batch_results.txt \
        --output_file evaluation/ablation/data/cleaned_batch_results.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

RE_ITEM_START = re.compile(r'"question"\s*:\s*"', flags=re.IGNORECASE)
RE_FIELD = re.compile(
    r'"(?P<key>question|no_validation_answer|static_prompt_answer|answer)"\s*:\s*"(?P<val>(?:\\.|[^"\\])*)"',
    flags=re.IGNORECASE | re.DOTALL,
)

REQUIRED_KEYS = ("question", "no_validation_answer", "static_prompt_answer", "answer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert messy batch results text to clean JSONL")
    parser.add_argument("--input_file", required=True, help="Path to raw text file")
    parser.add_argument("--output_file", required=True, help="Path to output JSONL file")
    parser.add_argument(
        "--report_file",
        default="",
        help="Optional path for parse report JSON (defaults to output .report.json)",
    )
    return parser.parse_args()


def _decode_json_string_fragment(raw_value: str) -> str:
    """Decode a JSON-string fragment safely."""
    try:
        return json.loads(f'"{raw_value}"')
    except Exception:
        return raw_value.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').strip()


def _slice_candidate_blocks(text: str) -> List[str]:
    """Slice text into candidate item blocks by each 'question' occurrence."""
    starts = [m.start() for m in RE_ITEM_START.finditer(text)]
    if not starts:
        return []

    blocks: List[str] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        blocks.append(text[start:end])
    return blocks


def _extract_fields(block: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for match in RE_FIELD.finditer(block):
        key = match.group("key").strip().lower()
        value = _decode_json_string_fragment(match.group("val"))
        fields[key] = value
    return fields


def parse_raw_text(text: str) -> Tuple[List[Dict[str, str]], List[Dict[str, object]]]:
    records: List[Dict[str, str]] = []
    issues: List[Dict[str, object]] = []

    for idx, block in enumerate(_slice_candidate_blocks(text), start=1):
        fields = _extract_fields(block)
        missing = [k for k in REQUIRED_KEYS if not fields.get(k, "").strip()]

        if missing:
            issues.append(
                {
                    "item_index": idx,
                    "missing_keys": missing,
                    "preview": block[:280],
                }
            )
            continue

        records.append(
            {
                "question": fields["question"].strip(),
                "no_validation_answer": fields["no_validation_answer"].strip(),
                "static_prompt_answer": fields["static_prompt_answer"].strip(),
                "answer": fields["answer"].strip(),
            }
        )

    return records, issues


def write_jsonl(path: Path, records: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    report_path = Path(args.report_file) if args.report_file else output_path.with_suffix(".report.json")

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    text = input_path.read_text(encoding="utf-8")
    records, issues = parse_raw_text(text)

    write_jsonl(output_path, records)

    report = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "records_written": len(records),
        "skipped_items": len(issues),
        "issues": issues,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 80)
    print("CONVERSION COMPLETE")
    print("=" * 80)
    print(f"Input:          {input_path}")
    print(f"Output JSONL:   {output_path}")
    print(f"Report JSON:    {report_path}")
    print(f"Records written:{len(records)}")
    print(f"Skipped items:  {len(issues)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
