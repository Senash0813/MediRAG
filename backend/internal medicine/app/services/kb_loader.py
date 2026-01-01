from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def load_kb(paths_try: List[str]) -> List[Dict[str, Any]]:
    """
    Try to load a JSON/JSONL file. Returns list of dicts.
    Acceptable item keys (recommended): 'id', 'question', 'answer', 'specialty', 'source'
    If your items are plain Q/A pairs, this will still work.
    """
    path = None
    for p in paths_try:
        if os.path.exists(p):
            path = p
            break
    if path is None:
        raise FileNotFoundError(f"Could not find knowledge base. Tried: {paths_try}")

    items: List[Dict[str, Any]] = []
    # try JSON lines
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        if text.startswith("["):
            # JSON array
            data = json.loads(text)
            if isinstance(data, list):
                items = data
            else:
                raise ValueError("JSON root is not a list.")
        else:
            # try parse as JSONL
            lines = text.splitlines()
            try:
                for ln in lines:
                    if ln.strip():
                        items.append(json.loads(ln))
            except Exception as e:
                # fallback: try to parse as a single JSON object with a key containing list
                data = json.loads(text)
                if isinstance(data, dict):
                    # find first list-like value
                    for v in data.values():
                        if isinstance(v, list):
                            items = v
                            break
                    if not items:
                        raise ValueError("Couldn't find a list-of-items structure in JSON.")
                else:
                    raise e

    # normalize: ensure each item is a dict with 'id', 'text' fields
    normalized = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        nid = it.get("id", f"doc_{i}")
        # Build a text field combining question and answer and any metadata
        parts = []
        if it.get("question"):
            parts.append("Q: " + str(it.get("question")).strip())
        if it.get("answer"):
            parts.append("A: " + str(it.get("answer")).strip())
        # keep other fields as metadata
        meta = {k: v for k, v in it.items() if k not in ("question", "answer")}
        text = " ".join(parts).strip()
        if not text:
            # maybe the JSON uses 'text' or 'content'
            text = it.get("text") or it.get("content") or ""
        if not text:
            # if still empty, skip
            continue
        normalized.append({"id": nid, "text": text, "meta": meta})

    return normalized
