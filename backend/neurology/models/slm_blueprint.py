# models/slm_blueprint.py
import json
import torch
from typing import List


class SLMBlueprintGenerator:
    """
    Blueprint generation using fine-tuned SLM with LoRA adapter.
    Uses shared base model with dynamic adapter switching.
    """

    def __init__(self, shared_slm):
        """
        Args:
            shared_slm: Instance of SharedSLMManager
        """
        self.shared_slm = shared_slm

        # The exact system prompt used during fine-tuning
        self.system_prompt = """You are a senior Neurologist and Neurosurgeon. 
        
Your task is to generate an ordered blueprint describing what a strong answer to a clinical question must contain. 
Generate 3 to 4 ordered requirements, concise and clinically meaningful. Output valid JSON.
"""

    def generate_blueprint(self, user_query: str) -> List[str]:
        """
        Generates the clinical blueprint for the instruction-following re-ranker.

        Args:
            user_query: Original user question

        Returns:
            List of string requirements (the blueprint)
        """

        # Activate blueprint adapter — must remain active for entire generation
        self.shared_slm.set_adapter("blueprint")
        model = self.shared_slm.model
        tokenizer = self.shared_slm.tokenizer
        device = self.shared_slm.device

        print(f"[SLM Blueprint] Generating blueprint for query...")

        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_query}
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(text, return_tensors="pt").to(device)
            input_length = inputs.input_ids.shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            generated_tokens = outputs[0][input_length:]
            prediction_text = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            ).strip()

            # Parse JSON response
            try:
                prediction = json.loads(prediction_text)
                blueprint = prediction.get("blueprint", [])
            except json.JSONDecodeError:
                print("[SLM Blueprint] ✗ JSON Decode Error. Returning fallback blueprint.")
                return ["1. Answer the specific clinical query directly and accurately."]

            # Guard against empty blueprint
            if not blueprint:
                print("[SLM Blueprint] ✗ Empty blueprint returned. Using fallback.")
                return ["1. Answer the specific clinical query directly and accurately."]

            print(f"[SLM Blueprint] ✓ Blueprint generated with {len(blueprint)} requirements.")
            return blueprint

        except Exception as e:
            print(f"[SLM Blueprint] ✗ Blueprint generation FAILED: {e}")
            return ["1. Answer the specific clinical query directly and accurately."]