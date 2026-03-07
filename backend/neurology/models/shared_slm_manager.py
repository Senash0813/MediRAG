# models/shared_slm_manager.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login
import os

class SharedSLMManager:
    """
    Loads ONE base model and attaches multiple LoRA adapters.
    Allows switching between adapters dynamically.
    Uses GPU (4-bit quantized) when CUDA is available, otherwise CPU/MPS (full precision).
    """

    VALID_ADAPTERS = {"gatekeeper", "blueprint"}

    def __init__(self, base_model: str, hf_token: str = None):
        self.base_model = base_model
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.model = None
        self.tokenizer = None
        self._initialized = False

    def _get_load_kwargs(self):
        """Choose device and quantization based on what is available."""
        if torch.cuda.is_available():
            # NVIDIA GPU: use 4-bit quantization to fit in VRAM
            print("[Shared SLM] Using CUDA GPU with 4-bit quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            return {
                "quantization_config": bnb_config,
                "device_map": "auto",
                "torch_dtype": torch.float16,
            }
        # Mac (MPS) or CPU: no BNB (CUDA-only), use auto device and float16
        if torch.backends.mps.is_available():
            print("[Shared SLM] Using Apple MPS (Metal); no quantization")
        else:
            print("[Shared SLM] Using CPU; no quantization")
        return {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
        }

    def initialize(self, gatekeeper_adapter_path: str, blueprint_adapter_path: str):
        if self._initialized:
            return
        
        login(token=self.hf_token, add_to_git_credential=False)

        print("[Shared SLM] Initializing base model (this may take 30-60 seconds)...")

        load_kwargs = self._get_load_kwargs()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            token=self.hf_token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Load base model (quantized on CUDA, full precision on Mac/CPU)
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            token=self.hf_token,
            **load_kwargs,
        )

        # Attach first adapter (gatekeeper)
        self.model = PeftModel.from_pretrained(
            base,
            gatekeeper_adapter_path,
            adapter_name="gatekeeper"
        )

        # Load second adapter (blueprint)
        self.model.load_adapter(
            blueprint_adapter_path,
            adapter_name="blueprint"
        )

        self.model.eval()
        self._initialized = True
        print("[Shared SLM] ✓ Base model + adapters loaded successfully")

    def set_adapter(self, adapter_name: str):
        if adapter_name not in self.VALID_ADAPTERS:
            raise ValueError(
                f"[Shared SLM] Unknown adapter: '{adapter_name}'. "
                f"Must be one of {self.VALID_ADAPTERS}"
            )
        self.model.set_adapter(adapter_name)