# models/llm_rephraser.py
from prompts.system_prompt import SYSTEM_PROMPT


class LLMRephraser:
    def __init__(self, shared_phi):
        self.phi = shared_phi

    def rephrase(self, question: str, retrieved_answers: list) -> str:
        evidence = "\n".join([f"- {r['answer']}" for r in retrieved_answers])

        prompt = f"""{SYSTEM_PROMPT}

Question:
{question}

Retrieved Answers:
{evidence}

Final Answer:"""

        return self.phi.generate(prompt, max_new_tokens=512, temperature=0.0)
