#!/usr/bin/env python3
"""
Quick script to check GPU/CUDA availability for the MediRAG neurology backend.
Run this before starting the server to verify your setup.
"""

import sys

print("=" * 80)
print("MediRAG Neurology Backend - GPU/CUDA Check")
print("=" * 80)

# Check PyTorch
try:
    import torch
    print("✓ PyTorch is installed")
    print(f"  Version: {torch.__version__}")
except ImportError:
    print("✗ PyTorch is NOT installed")
    print("\nInstall with CUDA support:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)

# Check CUDA
cuda_available = torch.cuda.is_available()
if cuda_available:
    print("✓ CUDA is available")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  Device Count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024**3)
        print(f"  GPU {i}: {props.name}")
        print(f"    VRAM: {vram_gb:.2f} GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")
else:
    print("✗ CUDA is NOT available")
    print("\nPossible reasons:")
    print("  1. PyTorch CPU version is installed (instead of CUDA version)")
    print("  2. NVIDIA GPU is not present")
    print("  3. CUDA drivers are not installed")
    print("\nTo install PyTorch with CUDA support:")
    print("  1. Uninstall current PyTorch:")
    print("     pip uninstall torch torchvision torchaudio")
    print("  2. Install CUDA version (for CUDA 11.8):")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("  3. Or for CUDA 12.1:")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

# Check bitsandbytes
try:
    import bitsandbytes
    print("✓ bitsandbytes is installed (required for 4-bit quantization)")
    if cuda_available:
        # Test if bitsandbytes can use CUDA
        try:
            test_tensor = torch.tensor([1.0]).cuda()
            print("  ✓ bitsandbytes can use CUDA")
        except Exception as e:
            print(f"  ✗ bitsandbytes CUDA test failed: {e}")
except ImportError:
    print("✗ bitsandbytes is NOT installed")
    print("  Install with: pip install bitsandbytes")

# Check transformers
try:
    import transformers
    print(f"✓ transformers is installed (version: {transformers.__version__})")
except ImportError:
    print("✗ transformers is NOT installed")

# Check accelerate
try:
    import accelerate
    print(f"✓ accelerate is installed (version: {accelerate.__version__})")
except ImportError:
    print("✗ accelerate is NOT installed")

# Check peft
try:
    import peft
    print(f"✓ peft is installed (version: {peft.__version__})")
except ImportError:
    print("✗ peft is NOT installed")

print("=" * 80)

if cuda_available:
    print("\n✓ Your system is ready to use GPU acceleration!")
    print("The model will load with 4-bit quantization (~2-3GB VRAM).")
    
    # Check VRAM
    min_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if min_vram_gb < 4:
        print(f"\n⚠ WARNING: Your GPU has only {min_vram_gb:.1f}GB VRAM.")
        print("  The model might not fit. Consider using a smaller model or CPU fallback.")
else:
    print("\n⚠ GPU is not available. The model will load on CPU (~16GB RAM required).")
    print("To avoid memory errors, either:")
    print("  1. Install CUDA-enabled PyTorch (recommended)")
    print("  2. Set FORCE_GPU=False in config.py and run with: uvicorn main:app --no-reload")
    print("  3. Use a smaller model (e.g., Llama-3.2-1B-Instruct)")

print()
