import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Number of GPUs:", torch.cuda.device_count())
print("Current GPU:", torch.cuda.get_device_name(0))

"""
PyTorch version: 2.8.0+cu129
CUDA available: True
CUDA version: 12.9
Number of GPUs: 1
Current GPU: NVIDIA GeForce RTX 2060 SUPER
"""


