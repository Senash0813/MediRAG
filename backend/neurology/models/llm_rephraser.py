# models/llm_rephraser.py

import requests
from prompts.system_prompt import SYSTEM_PROMPT

class LLMRephraser:
    def __init__(self, model_name, base_url="http://localhost:11434"):
        self.model = model_name
        self.base_url = base_url

    def rephrase(self, question, retrieved_answers):
        evidence = "\n".join(
            [f"- {r['answer']}" for r in retrieved_answers]
        )

        prompt = f"""
{SYSTEM_PROMPT}

Question:
{question}

Retrieved Answers:
{evidence}

Final Answer:
"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0
            }
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120
        )

        response.raise_for_status()
        return response.json()["response"].strip()
