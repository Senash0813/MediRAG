from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass
class LMStudioClient:
    """Minimal OpenAI-compatible client for LM Studio's local server."""

    base_url: str
    model: str
    timeout_s: float = 120.0

    def chat(
        self,
        *,
        messages: List[Dict[str, str]],
        max_tokens: int = 700,
        temperature: float = 0.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if extra:
            payload.update(extra)

        r = requests.post(url, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()

        return data["choices"][0]["message"]["content"]
