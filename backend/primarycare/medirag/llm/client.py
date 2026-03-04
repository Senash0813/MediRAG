from __future__ import annotations

from typing import Dict, List

import requests

from medirag.config import AppSettings


class LMStudioClient:
	"""HTTP client for LM Studio serving Qwen2.5-3B-Instruct.

	This assumes an OpenAI-compatible /v1/chat/completions API. If your
	LM Studio setup uses a different route or payload, adjust the _endpoint
	and body structure accordingly.
	"""

	def __init__(self, settings: AppSettings) -> None:
		self._settings = settings

	def generate_chat(
		self,
		system_prompt: str,
		user_prompt: str,
		*,
		max_new_tokens: int | None = None,
		temperature: float | None = None,
	) -> str:
		settings = self._settings
		url = settings.lmstudio_base_url.rstrip("/") + "/v1/chat/completions"

		body: Dict = {
			"model": settings.lmstudio_model,
			"messages": [
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt},
			],
			"temperature": temperature if temperature is not None else settings.lm_temperature,
			"max_tokens": max_new_tokens if max_new_tokens is not None else settings.lm_max_new_tokens,
		}

		resp = requests.post(url, json=body, timeout=120)
		resp.raise_for_status()

		js = resp.json()
		# OpenAI-style: choices[0].message.content
		choices: List[Dict] = js.get("choices") or []
		if not choices:
			raise RuntimeError("LM Studio response missing 'choices'")
		content = choices[0].get("message", {}).get("content")
		if not content:
			raise RuntimeError("LM Studio response missing message content")
		return content

