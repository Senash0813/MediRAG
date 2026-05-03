#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

TARGET_QUESTION_IDS = [
    "38_79733545_0_2",
    "38_6276512_2_3",
    "22_23496682_6_1",
    "57_40274863_3_1",
    "50_53217809_4_1",
    "8_33630656_4_2",
    "75_845829_4_3",
    "60_46827041_1_1",
    "38_2622398_0_2",
    "38_1296417_2_2",
    "36_26154596_0_3",
    "36_58695026_0_3",
    "36_18533795_2_1",
    "36_11167310_4_1",
    "36_4230700_0_2",
    "36_15640360_1_1",
    "70_7187892_4_3",
    "10_17064827_1_1",
    "70_16028130_2_1",
    "0_54538893_7_3",
]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build eval dataset from fixed question IDs")
    p.add_argument("--input", default="resources/selected_specialties.jsonl")
    p.add_argument("--output", default="evaluation/retrieval/data/eval_dataset_20.jsonl")
    p.add_argument("--strict", action="store_true", help="Fail if any ID is missing")
    return p.parse_args()

def normalize_row(raw: Dict[str, object]) -> Dict[str, object]:
    qa_id = str(raw.get("qa_id", "")).strip()
    paper_id = str(raw.get("paper_id", "")).strip()
    question = str(raw.get("question", "")).strip()
    answer = str(raw.get("answer", "")).strip()
    specialty_tag = qa_id.split("_", 1)[0].strip() if qa_id else "unknown"

    return {
        "question_id": qa_id,
        "question": question,
        "ground_truth_answer": answer,
        "gold_doc_ids": [paper_id] if paper_id else [],
        "source_paper_id": paper_id,
        "source_qa_id": qa_id,
        "specialty_tag": specialty_tag,
    }

def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.is_file():
        raise SystemExit(f"Input not found: {input_path}")

    target_set: Set[str] = set(TARGET_QUESTION_IDS)
    found: Dict[str, Dict[str, object]] = {}
    scanned = 0
    invalid = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            scanned += 1
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue

            qa_id = str(raw.get("qa_id", "")).strip()
            if qa_id in target_set and qa_id not in found:
                row = normalize_row(raw)

                if not row["question"] or not row["ground_truth_answer"] or not row["gold_doc_ids"]:
                    invalid += 1
                    continue

                found[qa_id] = row

            if len(found) == len(target_set):
                break

    missing = [qid for qid in TARGET_QUESTION_IDS if qid not in found]

    if args.strict and missing:
        raise SystemExit(
            "Missing required question IDs:\n" + "\n".join(missing)
        )

    # Preserve your requested order exactly
    ordered_rows: List[Dict[str, object]] = [found[qid] for qid in TARGET_QUESTION_IDS if qid in found]

    with output_path.open("w", encoding="utf-8") as f:
        for row in ordered_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "requested_count": len(TARGET_QUESTION_IDS),
        "written_count": len(ordered_rows),
        "missing_count": len(missing),
        "missing_ids": missing,
        "scanned_rows": scanned,
        "invalid_rows": invalid,
    }

    meta_path = output_path.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("ID-BASED EVALUATION DATASET CREATED")
    print("=" * 72)
    print(f"Output:    {output_path}")
    print(f"Metadata:  {meta_path}")
    print(f"Requested: {len(TARGET_QUESTION_IDS)}")
    print(f"Written:   {len(ordered_rows)}")
    print(f"Missing:   {len(missing)}")
    if missing:
        print("Missing IDs:")
        for qid in missing:
            print(f"  - {qid}")
    print("=" * 72)

if __name__ == "__main__":
    main()