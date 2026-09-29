"""Day 3: Tensor ops & broadcasting.

Element-wise math, reductions, matmul, and the two broadcasting
rules -- the shape-debugging skill every later lesson builds on.
Run:  python3 code.py   (inside the ~/ai-lab venv)
"""
import time
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("using device:", device)

print("\n--- 1. Vectorized: the loop lives in C, not Python ---")
closes = torch.tensor([101.90, 103.70, 105.10, 104.40, 106.20], device=device)
pct = (closes[1:] - closes[:-1]) / closes[:-1] * 100   # daily % change, one line
print("closes shape:", closes.shape)
print("daily % change:", " ".join(f"{x:.3f}" for x in pct))

n = 200_000
big = torch.linspace(100, 110, n, device=device)
t0 = time.perf_counter()
looped = torch.empty_like(big)
for i in range(n):
    looped[i] = big[i] * 1.02
t1 = time.perf_counter()
vec = big * 1.02
t2 = time.perf_counter()
assert torch.allclose(looped, vec)
print(f"python loop over {n:,} cells: {(t1-t0)*1000:7.1f} ms")
print(f"one vectorized op       : {(t2-t1)*1000:7.2f} ms")

print("\n--- 2. Broadcasting: small shapes stretch ---")
candles = torch.tensor([                    # (3, 5): 3 days x O H L C V
    [100.50, 102.30,  99.80, 101.90, 1.25],
    [101.90, 104.10, 101.20, 103.70, 1.48],
    [103.70, 105.20, 102.90, 104.80, 1.61],
], device=device)
col_mean = candles.mean(dim=0)              # (5,): one mean per column
demeaned = candles - col_mean               # (3,5) - (5,) -> (3,5)
print("col means :", " ".join(f"{x:6.2f}" for x in col_mean))
print("shapes    :", candles.shape, "-", col_mean.shape, "->", demeaned.shape)

ohlc = candles[:, :4]                       # (3, 4)
day_open = ohlc[:, :1]                      # (3, 1)
rel = ohlc / day_open                       # (3,4) / (3,1): prices vs their open
print("day-0 as % of open:", " ".join(f"{x:5.1f}" for x in rel[0] * 100))
print("shapes    :", ohlc.shape, "/", day_open.shape, "->", rel.shape)

print("\n--- 3. The two rules, and the classic error ---")
a = torch.zeros(4, 3) + torch.zeros(3)        # trailing dims align      -> (4, 3)
b = torch.zeros(4, 3) + torch.zeros(4, 1)     # size-1 stretches         -> (4, 3)
print("ok  :", a.shape, b.shape)
try:
    torch.zeros(4, 3) + torch.zeros(2)        # 3 vs 2: match? no. 1? no. -> error
except RuntimeError as e:
    print("fail:", str(e).splitlines()[0])

print("\n--- 4. Reductions + matmul: collapse, then combine ---")
rng = candles[:, 1] - candles[:, 2]          # daily range H - L
print("daily range:", " ".join(f"{x:.2f}" for x in rng))
print(f"widest day: {rng.max().item():.2f} | avg range: {rng.mean().item():.2f}")

rets = torch.tensor([[ 0.012, -0.004,  0.018],   # META, 3 days
                     [ 0.031,  0.022, -0.009]],  # NVDA, 3 days
                    device=device)               # (2, 3)
w = torch.tensor([0.6, 0.4], device=device)      # (2,) portfolio weights
port = w @ rets                                  # (2,) @ (2, 3) -> (3,)
print("shapes:", w.shape, "@", rets.shape, "->", port.shape)
print("portfolio daily returns:", " ".join(f"{x:+.2%}" for x in port))
