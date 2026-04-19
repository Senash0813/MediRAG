# models/domain_checker.py

INTENT_IN_SCOPE = "IN_SCOPE"
INTENT_OUT_OF_SCOPE = "OUT_OF_SCOPE"
INTENT_CHITCHAT = "CHITCHAT"


class DomainChecker:
    """
    Uses the shared Phi model to classify user intent into one of three categories:
    IN_SCOPE (neurology/neurosurgery), OUT_OF_SCOPE, or CHITCHAT.
    """

    def __init__(self, shared_phi):
        self.phi = shared_phi

    def classify_intent(self, query: str) -> str:
        """
        Returns one of: IN_SCOPE, OUT_OF_SCOPE, CHITCHAT.
        Fails open to IN_SCOPE on any error.
        """
        prompt = f"""You are a strict intent classifier for a medical AI assistant that specializes exclusively in Neurology and Neurosurgery.

Your task: Classify the user's message into exactly one of three categories.

Categories:
- IN_SCOPE: The message is a medical question about Neurology or Neurosurgery.
  Examples: neurological diseases (epilepsy, stroke, Parkinson's, MS, dementia, migraines), neurosurgical conditions/procedures (brain tumors, spinal surgery, aneurysm), neuroanatomy, neuro-imaging, CSF analysis, neurological symptoms (headache, seizure, weakness, tremor).

- OUT_OF_SCOPE: The message is a question or request that is not about Neurology/Neurosurgery.
  Examples: diabetes, cardiology, dermatology, coding, geography, sports, recipes, general knowledge.

- CHITCHAT: The message is casual conversation with no medical question intent.
  Examples: greetings ("hi", "hello", "how are you"), thanks ("thank you"), farewells ("bye"), compliments, or small talk.

User message: "{query}"

Respond with ONLY one word: IN_SCOPE, OUT_OF_SCOPE, or CHITCHAT. No explanation, no punctuation.
Answer:"""

        try:
            answer = self.phi.generate(prompt, max_new_tokens=10, temperature=0.0).upper()

            if answer.startswith("CHITCHAT"):
                return INTENT_CHITCHAT
            if answer.startswith("OUT_OF_SCOPE"):
                return INTENT_OUT_OF_SCOPE
            return INTENT_IN_SCOPE

        except Exception as e:
            print(f"[DomainChecker] Warning: classify_intent failed: {e}. Allowing query through.")
            return INTENT_IN_SCOPE

    def generate_chitchat_response(self, query: str) -> str:
        """
        Generates a friendly conversational reply for chitchat messages.
        """
        prompt = f"""You are MediRAG, a friendly AI assistant specialized in Neurology and Neurosurgery.
The user has sent you a casual message. Respond in a warm, concise, and friendly way.
Naturally let them know you are here to help with neurology or neurosurgery questions when they are ready.

User message: "{query}"

Your response:"""

        try:
            return self.phi.generate(prompt, max_new_tokens=80, temperature=0.7)
        except Exception as e:
            print(f"[DomainChecker] Warning: generate_chitchat_response failed: {e}.")
            return "Hello! I'm MediRAG, your Neurology and Neurosurgery assistant. Feel free to ask me any neurology-related questions!"
