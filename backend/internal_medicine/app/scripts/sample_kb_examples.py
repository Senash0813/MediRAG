from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterator

from app.services.kb import _iter_items_from_path


def reservoir_sample(iterator: Iterator[Dict[str, Any]], k: int) -> list[Dict[str, Any]]:
    sample: list[Dict[str, Any]] = []
    for i, item in enumerate(iterator):
        if i < k:
            sample.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                sample[j] = item
    return sample


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sample examples from KB with reservoir sampling")
    p.add_argument("--kb", type=Path, required=True, help="Path to KB JSON or JSONL")
    p.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    p.add_argument("--n", type=int, default=30, help="Number of examples to sample")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    it = _iter_items_from_path(str(args.kb))
    sampled = reservoir_sample(it, args.n)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for item in sampled:
            # Normalize to a minimal example: query and optional reference answer
            query = item.get("question") or item.get("text") or item.get("content") or item.get("query") or ""
            answer = item.get("answer") or item.get("response") or None
            out = {"query": str(query).strip()}
            if answer:
                out["reference_answer"] = str(answer).strip()
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sampled)} examples to {args.out}")


if __name__ == "__main__":
    main()
