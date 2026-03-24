from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional


def _iter_items_from_path(path: str) -> Iterable[Any]:
    """Yield raw KB items from JSON array or JSONL without reading whole file."""

    # Peek first non-whitespace character to detect format
    with open(path, "r", encoding="utf-8") as f:
        first_non_ws = ""
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_non_ws = ch
                break

    if first_non_ws == "[":
        file_size = os.path.getsize(path)
        # For large JSON arrays, require a streaming parser.
        # Stdlib json.load will try to materialize the full structure in RAM.
        if file_size > 200 * 1024 * 1024:
            try:
                import ijson  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "KB is a large JSON array (>200MB). Install `ijson` for streaming parse "
                    "(pip install ijson) or convert the KB to JSONL."
                ) from e

            with open(path, "rb") as fb:
                for item in ijson.items(fb, "item"):
                    yield item
            return

        # Small JSON array: ok to load directly
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON root is not a list")
        for item in data:
            yield item
        return

    # JSONL (one JSON object per line) or dict-with-list fallback
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                yield json.loads(ln)


def load_kb(paths_try: List[str], *, max_docs: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load a JSON/JSONL knowledge base and normalize to {id,text,meta} dicts.

    Notes:
    - Supports huge JSON arrays via streaming (`ijson`).
    - For very large KBs, set `max_docs` to avoid runaway memory usage.
    """

    path = None
    for candidate in paths_try:
        if candidate and os.path.exists(candidate):
            path = candidate
            break

    if path is None:
        raise FileNotFoundError(f"Could not find knowledge base. Tried: {paths_try}")

    normalized: list[dict[str, Any]] = []
    for i, it in enumerate(_iter_items_from_path(path)):
        if max_docs is not None and len(normalized) >= int(max_docs):
            break
        if not isinstance(it, dict):
            continue

        nid = it.get("id") or it.get("qa_id") or it.get("doc_id") or f"doc_{i}"

        parts: list[str] = []
        if it.get("question"):
            parts.append("Q: " + str(it.get("question")).strip())
        if it.get("answer"):
            parts.append("A: " + str(it.get("answer")).strip())

        # Keep metadata small; large fields (e.g., passage_text) can blow up RAM/disk for big KBs.
        meta = {k: v for k, v in it.items() if k not in ("question", "answer", "passage_text")}
        doc_text = " ".join(parts).strip()

        if not doc_text:
            doc_text = (it.get("text") or it.get("content") or "").strip()

        if not doc_text:
            continue

        normalized.append({"id": str(nid), "text": doc_text, "meta": meta})

    return normalized
