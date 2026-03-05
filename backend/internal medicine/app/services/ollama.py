from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class OllamaGenerator:
    base_url: str
    model: str
    timeout_s: float = 60.0

    def __call__(
        self,
        prompt: str,
        *,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
        **_: Any,
    ) -> List[Dict[str, str]]:
        # The rest of the codebase expects a HF-pipeline-like return:
        # [{'generated_text': '...'}]
        url = self.base_url.rstrip("/") + "/api/generate"

        options: Dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = float(temperature)
        if max_length is not None:
            # Ollama uses num_predict (max tokens to generate).
            options["num_predict"] = int(max_length)

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if options:
            payload["options"] = options

        data = json.dumps(payload).encode("utf-8")
        req = Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")
            except Exception:
                pass
            raise RuntimeError(f"Ollama HTTP error {e.code}: {body or e.reason}") from e
        except URLError as e:
            raise RuntimeError(f"Failed to reach Ollama at {url}: {e.reason}") from e

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON from Ollama: {raw[:500]}") from e

        # /api/generate returns: { response: '...', done: true, ... }
        text = parsed.get("response", "")
        if not isinstance(text, str):
            text = str(text)

        return [{"generated_text": text.strip()}]
