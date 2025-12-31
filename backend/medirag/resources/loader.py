from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class Dataset:
    data: List[Dict[str, Any]]
    passages: List[str]


def load_jsonl_dataset(jsonl_path: Path) -> Dataset:
    data: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    passages = [row["passage_text"] for row in data]
    return Dataset(data=data, passages=passages)
