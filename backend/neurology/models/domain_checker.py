# models/domain_checker.py
import requests
import json


class DomainChecker:
    """
    Uses the Ollama LLM to determine whether a user query is within
    the domain of Neurology and Neurosurgery before entering the RAG pipeline.
    """

    def __init__(self, model_name: str, base_url: str):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def is_in_domain(self, query: str) -> bool:
        """
        Returns True if the query is related to Neurology or Neurosurgery,
        False otherwise.
        """
        prompt = f"""You are a strict domain classifier for a medical AI assistant that specializes exclusively in Neurology and Neurosurgery.

Your task: Decide whether the user's question is within the domain of Neurology or Neurosurgery.

Domain includes (but is not limited to):
- Neurological diseases and disorders (e.g. epilepsy, stroke, Parkinson's, multiple sclerosis, dementia, migraines, neuropathy)
- Neurosurgical conditions and procedures (e.g. brain tumors, spinal surgery, deep brain stimulation, aneurysm clipping, craniotomy)
- Neuroanatomy and neurophysiology
- Diagnostic workup for neurological symptoms (e.g. headache, seizure, weakness, numbness, tremor, gait disturbance)
- Neuro-imaging interpretation (MRI, CT brain/spine)
- Cerebrospinal fluid analysis
- Neuro-oncology, neurovascular conditions, neuroinfectious disease

Out of domain includes:
- General medicine topics unrelated to neurology (e.g. diabetes management, cardiac conditions, dermatology)
- Non-medical questions (e.g. coding, history, geography, recipes, sports)
- Mental health / psychiatry (unless directly overlapping with neurology)

User question: "{query}"

Respond with ONLY a single word: YES if in domain, NO if out of domain. No explanation, no punctuation, just YES or NO."""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,   # Deterministic — no creativity needed
                        "num_predict": 5,     # We only need one word
                    }
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            answer = result.get("response", "").strip().upper()
            # Accept YES even if model adds minor noise like "YES." or "YES\n"
            return answer.startswith("YES")

        except requests.exceptions.RequestException as e:
            # If Ollama is unreachable, fail open (allow query through)
            # so the pipeline doesn't break on a connectivity issue
            print(f"[DomainChecker] Warning: Ollama request failed: {e}. Allowing query through.")
            return True
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[DomainChecker] Warning: Unexpected response format: {e}. Allowing query through.")
            return True