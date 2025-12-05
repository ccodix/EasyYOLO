#!/usr/bin/env python3
"""
Quick CUDA check script for training
"""
import sys

try:
    import torch
except ImportError:
    print("ERROR: PyTorch not installed!")
    print("Install with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)

print("CUDA Status Check")

print(f"\nPyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version (PyTorch): {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"GPU Compute Capability: {torch.cuda.get_device_capability(0)}")
    
    print("RESULT: CUDA is WORKING - You can train on GPU")
    
else:
    print(f"PyTorch CUDA Build: {torch.version.cuda if torch.version.cuda else 'CPU-only version'}")
    
    print("RESULT: CUDA is NOT WORKING - Training will use CPU")
    print("=" * 70)
    print("\nExpected training speed: VERY SLOW")
    
    print("HOW TO FIX:")
    
    print("\n1. Check if you have NVIDIA GPU:")
    print("   Run: nvidia-smi")
    print("   You should see your GPU listed")
    
    print("\n2. Install correct PyTorch version:")
    print("   For Python 3.13:")
    print("     pip uninstall torch torchvision")
    print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n   For Python 3.8-3.12:")
    print("     pip uninstall torch torchvision")
    print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    
    print("\n3. Restart your terminal/IDE and run this script again")
    

print("\n")
