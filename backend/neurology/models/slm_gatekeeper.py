# # models/slm_gatekeeper.py

# import json
# import torch
# from typing import List, Dict, Any
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from peft import PeftModel
# import os


# class SLMGatekeeper:
#     """
#     Constraint-aware chunk filtering using fine-tuned SLM with LoRA adapter.
#     Lazy-loaded - model only initialized when first used.
#     """
    
#     def __init__(self, base_model: str, adapter_path: str, hf_token: str = None):
#         """
#         Args:
#             base_model: HuggingFace model name (e.g., 'meta-llama/Llama-3.2-3B-Instruct')
#             adapter_path: Local path to LoRA adapter weights
#             hf_token: HuggingFace API token (or None to use HF_TOKEN env variable)
#         """
#         self.base_model = base_model
#         self.adapter_path = adapter_path
#         self.hf_token = hf_token or os.getenv("HF_TOKEN")
        
#         self.model = None
#         self.tokenizer = None
#         self._initialized = False
        
#         self.system_prompt = """You are a medical constraint-aware verifier for a Neurology/Neurosurgery RAG system.

# Your job is STRICT binary classification.

# You must output FAIL if ANY of the following are true:

# 1) The Target Chunk explicitly contradicts a hard constraint in the query.
# 2) The Target Chunk discusses a different pathology than the one in the query.
# 3) The Target Chunk discusses a different anatomical region unrelated to the query.
# 4) The Target Chunk is clearly unrelated to answering the question.
# 5) The Target Chunk cannot reasonably help answer the query.

# You must output PASS only if:
# - The chunk is applicable to the query AND
# - No hard constraint violations exist.

# Output STRICT JSON in this format:

# {
#   "constraints": {...},
#   "reasoning": "...",
#   "label": "PASS or FAIL"
# }
# """
    
#     def _initialize(self):
#         """Lazy initialization - load model with adapter when first needed."""
#         if self._initialized:
#             return
        
#         print("[SLM Gatekeeper] Initializing model (this may take 30-60 seconds)...")
        
#         try:
#             # 4-bit quantization for memory efficiency
#             bnb_config = BitsAndBytesConfig(
#                 load_in_4bit=True,
#                 bnb_4bit_quant_type="nf4",
#                 bnb_4bit_compute_dtype=torch.float16,
#                 bnb_4bit_use_double_quant=True
#             )
            
#             # Load tokenizer
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 self.base_model,
#                 token=self.hf_token
#             )
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#             self.tokenizer.padding_side = "right"
            
#             # Load base model
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 self.base_model,
#                 quantization_config=bnb_config,
#                 device_map="auto",
#                 torch_dtype=torch.float16,
#                 token=self.hf_token
#             )
            
#             # Load LoRA adapter
#             print(f"[SLM Gatekeeper] Loading LoRA adapter from: {self.adapter_path}")
#             self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
#             self.model.eval()
            
#             self._initialized = True
#             print("[SLM Gatekeeper] ✓ Model loaded successfully")
            
#         except Exception as e:
#             print(f"[SLM Gatekeeper] ✗ FAILED to initialize: {e}")
#             raise
    
#     def filter_chunks(self, user_query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """
#         Filter chunks using the SLM gatekeeper.
        
#         Args:
#             user_query: Original user question
#             chunks: List of retrieved chunks (must have 'answer' field)
            
#         Returns:
#             List of chunks labeled as PASS
#         """
#         # Lazy load on first use
#         self._initialize()
        
#         print(f"[SLM Gatekeeper] Starting filtration on {len(chunks)} chunks...")
        
#         passed_chunks = []
        
#         try:
#             for idx, chunk in enumerate(chunks):
#                 # Prepare messages
#                 messages = [
#                     {"role": "system", "content": self.system_prompt},
#                     {
#                         "role": "user",
#                         "content": f"""Query:
# {user_query}

# Target Chunk:
# {chunk.get('answer', '')}"""
#                     }
#                 ]
                
#                 # Format with chat template
#                 text = self.tokenizer.apply_chat_template(
#                     messages,
#                     tokenize=False,
#                     add_generation_prompt=True
#                 )
                
#                 # Tokenize and generate
#                 inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
#                 input_length = inputs.input_ids.shape[1]
                
#                 with torch.no_grad():
#                     outputs = self.model.generate(
#                         **inputs,
#                         max_new_tokens=256,
#                         do_sample=False,
#                         pad_token_id=self.tokenizer.eos_token_id
#                     )
                
#                 # Decode only generated tokens
#                 generated_tokens = outputs[0][input_length:]
#                 prediction_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                
#                 # Parse JSON response
#                 try:
#                     prediction = json.loads(prediction_text)
#                     label = prediction.get("label", "UNKNOWN")
#                 except json.JSONDecodeError:
#                     # Fallback parsing
#                     prediction_upper = prediction_text.upper()
#                     if "PASS" in prediction_upper and "FAIL" not in prediction_upper:
#                         label = "PASS"
#                     elif "FAIL" in prediction_upper:
#                         label = "FAIL"
#                     else:
#                         label = "UNKNOWN"
                
#                 # Keep only PASS chunks
#                 if label == "PASS":
#                     passed_chunks.append(chunk)
            
#             print(f"[SLM Gatekeeper] ✓ Filtration completed: {len(passed_chunks)}/{len(chunks)} chunks passed")
#             return passed_chunks
            
#         except Exception as e:
#             print(f"[SLM Gatekeeper] ✗ Filtration FAILED: {e}")
#             print(f"[SLM Gatekeeper] Returning all chunks unfiltered (fail-open mode)")
#             return chunks  # Fail-open: return all chunks on error

# models/slm_gatekeeper.py

import json
import torch
from typing import List, Dict, Any


class SLMGatekeeper:
    """
    Constraint-aware chunk filtering using fine-tuned SLM with LoRA adapter.
    Uses shared base model with dynamic adapter switching.
    """

    def __init__(self, shared_slm):
        """
        Args:
            shared_slm: Instance of SharedSLMManager
        """
        self.shared_slm = shared_slm

        self.system_prompt = """You are a medical constraint-aware verifier for a Neurology/Neurosurgery RAG system.

Your job is STRICT binary classification.

You must output FAIL if ANY of the following are true:

1) The Target Chunk explicitly contradicts a hard constraint in the query.
2) The Target Chunk discusses a different pathology than the one in the query.
3) The Target Chunk discusses a different anatomical region unrelated to the query.
4) The Target Chunk is clearly unrelated to answering the question.
5) The Target Chunk cannot reasonably help answer the query.

You must output PASS only if:
- The chunk is applicable to the query AND
- No hard constraint violations exist.

Output STRICT JSON in this format:

{
  "constraints": {...},
  "reasoning": "...",
  "label": "PASS or FAIL"
}
"""

    def filter_chunks(self, user_query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter chunks using the SLM gatekeeper.

        Args:
            user_query: Original user question
            chunks: List of retrieved chunks (must have 'answer' field)

        Returns:
            List of chunks labeled as PASS
        """

        # Activate gatekeeper adapter — must remain active for entire loop
        self.shared_slm.set_adapter("gatekeeper")
        model = self.shared_slm.model
        tokenizer = self.shared_slm.tokenizer
        device = self.shared_slm.device

        print(f"[SLM Gatekeeper] Starting filtration on {len(chunks)} chunks...")

        passed_chunks = []

        try:
            for idx, chunk in enumerate(chunks):
                # Prepare messages
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": f"""Query:
{user_query}

Target Chunk:
{chunk.get('answer', '')}"""
                    }
                ]

                # Format with chat template
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # Tokenize and generate
                inputs = tokenizer(text, return_tensors="pt").to(device)
                input_length = inputs.input_ids.shape[1]

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )

                # Decode only generated tokens
                generated_tokens = outputs[0][input_length:]
                prediction_text = tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True
                ).strip()

                # Parse JSON response
                try:
                    prediction = json.loads(prediction_text)
                    label = prediction.get("label", "UNKNOWN")
                except json.JSONDecodeError:
                    # Fallback parsing
                    prediction_upper = prediction_text.upper()
                    if "PASS" in prediction_upper and "FAIL" not in prediction_upper:
                        label = "PASS"
                    elif "FAIL" in prediction_upper:
                        label = "FAIL"
                    else:
                        label = "UNKNOWN"

                # Keep only PASS chunks
                if label == "PASS":
                    passed_chunks.append(chunk)

            print(f"[SLM Gatekeeper] ✓ Filtration completed: {len(passed_chunks)}/{len(chunks)} chunks passed")
            return passed_chunks

        except Exception as e:
            print(f"[SLM Gatekeeper] ✗ Filtration FAILED: {e}")
            print(f"[SLM Gatekeeper] Returning all chunks unfiltered (fail-open mode)")
            return chunks  # Fail-open