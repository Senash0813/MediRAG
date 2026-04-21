# models/shared_phi_manager.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import sys


class SharedPhiManager:
    """
    Loads microsoft/phi-2 once and exposes a generate() method.
    Shared between DomainChecker and LLMRephraser to avoid loading the model twice.
    """

    def __init__(self, model_name: str, force_gpu: bool = False, use_4bit: bool = True):
        self.model_name = model_name
        self.force_gpu = force_gpu
        self.use_4bit = use_4bit
        self.model = None
        self.tokenizer = None
        self._initialized = False

    def _get_load_kwargs(self):
        cuda_available = torch.cuda.is_available()

        if self.force_gpu and not cuda_available:
            print("[SharedPhi] ERROR: GPU required but CUDA not available.")
            sys.exit(1)

        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"[SharedPhi] Using CUDA GPU: {gpu_name} ({vram_gb:.1f}GB VRAM)")

            if self.use_4bit:
                print("[SharedPhi] Using 4-bit quantization")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                return {"quantization_config": bnb_config, "device_map": "auto", "torch_dtype": torch.float16}
            else:
                return {"device_map": "auto", "torch_dtype": torch.float16}

        print("[SharedPhi] WARNING: Using CPU; no quantization")
        return {"device_map": "cpu", "torch_dtype": torch.float16, "low_cpu_mem_usage": True}

    def initialize(self):
        if self._initialized:
            return

        print(f"[SharedPhi] Loading {self.model_name}...")
        load_kwargs = self._get_load_kwargs()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            **load_kwargs,
        )
        self.model.eval()
        self._initialized = True
        print(f"[SharedPhi] ✓ {self.model_name} loaded successfully")

    @property
    def device(self):
        if self.model is None:
            return torch.device("cpu")
        return next(self.model.parameters()).device

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
