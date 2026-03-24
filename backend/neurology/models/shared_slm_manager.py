# models/shared_slm_manager.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login
import os
import sys

class SharedSLMManager:
    """
    Loads ONE base model and attaches multiple LoRA adapters.
    Allows switching between adapters dynamically.
    Uses GPU (4-bit quantized) when CUDA is available, otherwise CPU/MPS (full precision).
    """

    VALID_ADAPTERS = {"gatekeeper", "blueprint"}

    def __init__(self, base_model: str, hf_token: str = None, force_gpu: bool = False, use_4bit: bool = True):
        self.base_model = base_model
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.force_gpu = force_gpu
        self.use_4bit = use_4bit
        self.model = None
        self.tokenizer = None
        self._initialized = False

    def _get_load_kwargs(self):
        """Choose device and quantization based on what is available."""
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        
        if self.force_gpu and not cuda_available:
            print("\n" + "="*80)
            print("[ERROR] GPU is required but CUDA is not available!")
            print("="*80)
            print("\nTo fix this, install PyTorch with CUDA support:")
            print("\n1. Uninstall current PyTorch:")
            print("   pip uninstall torch torchvision torchaudio")
            print("\n2. Install CUDA-enabled PyTorch (for CUDA 11.8):")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            print("\n3. Or for CUDA 12.1:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            print("\n4. Verify CUDA is working:")
            print("   python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\"")
            print("\nAlternatively, set FORCE_GPU=False in config.py to use CPU (requires ~16GB RAM)")
            print("="*80 + "\n")
            sys.exit(1)
        
        if cuda_available:
            # NVIDIA GPU: use 4-bit quantization to fit in VRAM
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown GPU"
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.device_count() > 0 else 0
            print(f"[Shared SLM] Using CUDA GPU: {gpu_name} ({vram_gb:.1f}GB VRAM)")
            
            if self.use_4bit:
                print("[Shared SLM] Using 4-bit quantization (saves VRAM)")
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
            else:
                print("[Shared SLM] Using full precision (requires more VRAM)")
                return {
                    "device_map": "auto",
                    "torch_dtype": torch.float16,
                }
        
        # Fallback to CPU
        if torch.backends.mps.is_available():
            print("[Shared SLM] WARNING: Using CPU (MPS skipped to avoid large-buffer limits); no quantization")
        else:
            print("[Shared SLM] WARNING: Using CPU; no quantization (requires ~16GB RAM)")
        
        return {
            "device_map": "cpu",
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

    @property
    def device(self):
        """Device of the model (for moving inputs). Uses first parameter's device when using device_map."""
        if self.model is None:
            return torch.device("cpu")
        return next(self.model.parameters()).device

    def set_adapter(self, adapter_name: str):
        if adapter_name not in self.VALID_ADAPTERS:
            raise ValueError(
                f"[Shared SLM] Unknown adapter: '{adapter_name}'. "
                f"Must be one of {self.VALID_ADAPTERS}"
            )
        self.model.set_adapter(adapter_name)