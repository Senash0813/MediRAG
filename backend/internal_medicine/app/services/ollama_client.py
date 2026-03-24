from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx


@dataclass(frozen=True)
class OllamaClient:
    host: str
    timeout_s: float = 120.0

    def _url(self, path: str) -> str:
        return self.host.rstrip("/") + path

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        seed: Optional[int] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            },
        }
        if seed is not None:
            payload["options"]["seed"] = int(seed)

        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(self._url("/api/generate"), json=payload)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                details = ""
                try:
                    details = resp.json().get("error", "")
                except Exception:
                    details = resp.text

                msg = (
                    f"Ollama request failed ({resp.status_code}) at {resp.request.url}. "
                    f"{details}".strip()
                )
                raise RuntimeError(
                    msg
                    + "\n\nIf this is a 404, the most common cause is a missing model tag. "
                    + "Run `ollama list` and set OLLAMA_GENERATOR_MODEL/OLLAMA_JUDGE_MODEL to an exact tag, "
                    + 'e.g. "phi:2.7b".'
                ) from exc

            data = resp.json()

        # Ollama returns: {"response": "...", "done": true, ...}
        return str(data.get("response", ""))
