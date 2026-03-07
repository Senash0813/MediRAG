# models/slm_blueprint.py

# import json
# import torch
# from typing import List
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from peft import PeftModel
# import os

# class SLMBlueprintGenerator:
#     """
#     Blueprint generation using fine-tuned SLM with LoRA adapter.
#     Lazy-loaded - model only initialized when first used.
#     """
    
#     def __init__(self, base_model: str, adapter_path: str, hf_token: str = None):
#         self.base_model = base_model
#         self.adapter_path = adapter_path
#         self.hf_token = hf_token or os.getenv("HF_TOKEN")
        
#         self.model = None
#         self.tokenizer = None
#         self._initialized = False
        
#         # The exact system prompt used during fine-tuning
#         self.system_prompt = """You are a senior Neurologist and Neurosurgeon. 
        
#         Your task is to generate an ordered blueprint describing what a strong answer to a clinical question must contain. 
#         Generate 3 to 4 ordered requirements, concise and clinically meaningful. Output valid JSON.
        
#         """
    
#     def _initialize(self):
#         """Lazy initialization - load model with adapter when first needed."""
#         if self._initialized:
#             return
        
#         print("[SLM Blueprint] Initializing model (this may take 30-60 seconds)...")
        
#         try:
#             bnb_config = BitsAndBytesConfig(
#                 load_in_4bit=True,
#                 bnb_4bit_quant_type="nf4",
#                 bnb_4bit_compute_dtype=torch.float16,
#                 bnb_4bit_use_double_quant=True
#             )
            
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 self.base_model,
#                 token=self.hf_token
#             )
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#             self.tokenizer.padding_side = "right"
            
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 self.base_model,
#                 quantization_config=bnb_config,
#                 device_map="auto",
#                 torch_dtype=torch.float16,
#                 token=self.hf_token
#             )
            
#             print(f"[SLM Blueprint] Loading LoRA adapter from: {self.adapter_path}")
#             self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
#             self.model.eval()
            
#             self._initialized = True
#             print("[SLM Blueprint] ✓ Model loaded successfully")
            
#         except Exception as e:
#             print(f"[SLM Blueprint] ✗ FAILED to initialize: {e}")
#             raise

#     def generate_blueprint(self, user_query: str) -> List[str]:
#         """
#         Generates the clinical blueprint for the instruction-following re-ranker.
        
#         Args:
#             user_query: Original user question
            
#         Returns:
#             List of string requirements (the blueprint)
#         """
#         self._initialize()
        
#         print(f"[SLM Blueprint] Generating blueprint for query...")
        
#         try:
#             messages = [
#                 {"role": "system", "content": self.system_prompt},
#                 {"role": "user", "content": user_query}
#             ]
            
#             text = self.tokenizer.apply_chat_template(
#                 messages,
#                 tokenize=False,
#                 add_generation_prompt=True
#             )
            
#             inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
#             input_length = inputs.input_ids.shape[1]
            
#             with torch.no_grad():
#                 outputs = self.model.generate(
#                     **inputs,
#                     max_new_tokens=256,
#                     do_sample=False,
#                     pad_token_id=self.tokenizer.eos_token_id
#                 )
            
#             generated_tokens = outputs[0][input_length:]
#             prediction_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
#             # Parse JSON response
#             try:
#                 prediction = json.loads(prediction_text)
#                 blueprint = prediction.get("blueprint", [])
#             except json.JSONDecodeError:
#                 print("[SLM Blueprint] ✗ JSON Decode Error. Returning fallback blueprint.")
#                 return ["1. Answer the specific clinical query directly and accurately."]
            
#             print(f"[SLM Blueprint] ✓ Blueprint generated with {len(blueprint)} requirements.")
#             return blueprint
            
#         except Exception as e:
#             print(f"[SLM Blueprint] ✗ Blueprint generation FAILED: {e}")
#             return ["1. Answer the specific clinical query directly and accurately."] # Fail-safe


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