"""Day 2: Tensors -- the universal container.

Every number in AI lives in a tensor. Today: read any tensor's
.shape and picture exactly what's inside it.
Run:  python3 code.py   (inside the ~/ai-lab venv)
"""
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("using device:", device)

print("\n--- 1. The ladder: scalar -> vector -> matrix ---")
scalar = torch.tensor(42.0)                            # a single number
vector = torch.tensor([101.90, 103.70, 99.80])         # one ticker's closes
matrix = torch.tensor([[1., 2., 3.],
                       [4., 5., 6.]])                  # rows x cols

print("scalar:", scalar.shape, " ndim:", scalar.ndim, " numel:", scalar.numel())
print("vector:", vector.shape, " ndim:", vector.ndim, " numel:", vector.numel())
print("matrix:", matrix.shape, " ndim:", matrix.ndim, " numel:", matrix.numel())
print(matrix)

print("\n--- 2. One ticker's candles = a (2, 5) matrix ---")
# rows = days (dim 0), cols = O H L C V, volume in millions (dim 1)
aapl = torch.tensor([
    [100.50, 102.30,  99.80, 101.90, 1.25],   # day 0
    [101.90, 104.10, 101.20, 103.70, 1.48],   # day 1
], device=device)
print("shape:", aapl.shape, "| dtype:", aapl.dtype, "| device:", aapl.device.type)
print("day-0 candle:", aapl[0])
print("day-1 close :", aapl[1, 3].item())

print("\n--- 3. Batch two tickers = a (2, 2, 5) tensor ---")
nvda = torch.tensor([
    [187.20, 189.50, 186.10, 188.80, 2.10],
    [188.80, 191.40, 188.00, 190.60, 2.44],
], device=device)
batch = torch.stack([aapl, nvda])   # new dim 0 = which ticker
print("shape:", batch.shape, "| ndim:", batch.ndim, "| numel:", batch.numel())
print("NVDA, day-1 close:", batch[1, 1, 3].item())

print("\n--- 4. Shape is just layout metadata ---")
buf = torch.zeros(2, 5, device=device)   # empty candle buffer, same layout as aapl
print("zeros:", buf.shape, "numel:", buf.numel())
flat = aapl.reshape(-1)                  # flatten; -1 means 'figure it out'
print("flattened:", flat.shape)

print("\n--- 5. Integers vs floats: dtype matters ---")
print("int tensor dtype:  ", torch.tensor([1, 2, 3]).dtype)
print("float tensor dtype:", torch.tensor([1., 2., 3.]).dtype)
