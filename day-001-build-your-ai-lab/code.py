"""Day 1: Build your AI lab.

Your first tensor, running on your Mac's GPU (MPS) — or CPU fallback.
Run:  python3 code.py   (inside the ~/ai-lab venv)
"""
import torch

print("torch:", torch.__version__)
print("MPS (Apple GPU) available:", torch.backends.mps.is_available())

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("using device:", device)

a = torch.tensor([[1., 2.], [3., 4.]], device=device)
b = torch.tensor([[5., 6.], [7., 8.]], device=device)

print("a + b =")
print(a + b)

print("a @ b =   (matrix multiply — the operation at the heart of deep learning)")
print(a @ b)
