"""Day 38 - Scaled Dot-Product Attention, By Hand in Torch.

Day 37 gave every token three roles: a query (what am I
looking for), a key (what do I advertise), and a value (what
do I hand over if picked). Tonight those roles become one
formula - the single most reused line in modern AI:

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

Five moves, every one a primitive you already own: a matmul
(Day 12), a divide, a row-wise softmax (Day 14), another
matmul. Nothing else. If you can run this by hand, you can
read any transformer diagram ever drawn.

  Part 1  The formula by hand on 3 tokens, d_k = 4. Print
          every intermediate: scores QK^T (3, 3), the scaled
          scores, the softmax weights (each row a budget
          summing to 1), and the output (3, 2) - then check
          the whole chain against PyTorch's fused
          F.scaled_dot_product_attention.
  Part 2  Why the / sqrt(d_k) is there. A dot product is a
          sum of d_k coordinate products, so it grows with
          dimension: with every coordinate pair contributing
          0.25, the raw dot is 1.00 at d = 4 but 64.00 at
          d = 256. Feed scores like that to softmax and the
          winner takes literally everything - top weight
          1.0000, gradient factor p(1-p) = 0.000000, nothing
          can learn. Dividing by sqrt(d_k) keeps the winner
          confident (0.9820) and the gradient alive.
  Part 3  The causal mask. A decoder generating token i may
          not read tokens after i - that would be peeking at
          the answer. Set the future scores (j > i) to -inf
          BEFORE the softmax and exp(-inf) = 0 does the rest:
          position 0's weights become [1, 0, 0] and its output
          is exactly its own value row. Same mask PyTorch
          applies with is_causal=True.

Runs on the ~/ai-lab venv: torch only (free), CPU on
purpose (tiny hand-set tensors, exact digits - the same
ops run unchanged on MPS), no CUDA, no paid APIs, no
downloads. No RNG anywhere: every tensor is set by hand.

Expected output:

  Part 1 - attention by hand: softmax(QK^T / sqrt(d_k)) V
    Q (3, 4), K (3, 4), V (3, 2) - d_k = 4, sqrt(d_k) = 2
    scores = Q @ K^T (3, 3):
      [1.0000, 1.0000, 2.0000]
      [1.0000, 1.0000, 0.0000]
      [1.0000, 1.0000, 1.0000]
    scaled = scores / 2 (3, 3):
      [0.5000, 0.5000, 1.0000]
      [0.5000, 0.5000, 0.0000]
      [0.5000, 0.5000, 0.5000]
    weights = softmax row by row (3, 3):
      [0.2741, 0.2741, 0.4519]
      [0.3837, 0.3837, 0.2327]
      [0.3333, 0.3333, 0.3333]
    every weight row sums to 1.0000, 1.0000, 1.0000 - a budget (Day 14)
    out = weights @ V (3, 2):
      [1.6297, 0.8222]
      [1.0817, 1.1510]
      [1.3333, 1.0000]
    F.scaled_dot_product_attention vs by-hand: max diff 0.000000
  Part 2 - why divide by sqrt(d_k): dot products grow with d
    q . k where every coordinate pair contributes 0.25:
      d =   4: raw dot 1.00 -> scaled 0.50 | top weight raw 0.7311 vs scaled 0.6225 | grad p(1-p) raw 0.196612 vs scaled 0.235004
      d =  64: raw dot 16.00 -> scaled 2.00 | top weight raw 1.0000 vs scaled 0.8808 | grad p(1-p) raw 0.000000 vs scaled 0.104994
      d = 256: raw dot 64.00 -> scaled 4.00 | top weight raw 1.0000 vs scaled 0.9820 | grad p(1-p) raw 0.000000 vs scaled 0.017663
    unscaled: winner takes all and the gradient is ~0 - nothing learns
    scaled: still confident, gradient alive - that is the whole trick
  Part 3 - causal mask: a decoder may not see the future
    future scores (j > i) set to -inf before the softmax
    causal weights (3, 3):
      [1.0000, 0.0000, 0.0000]
      [0.5000, 0.5000, 0.0000]
      [0.3333, 0.3333, 0.3333]
    causal out (3, 2):
      [1.0000, 2.0000]
      [0.5000, 1.5000]
      [1.3333, 1.0000]
    position 0 out == V row 0 exactly: True (it may only read itself)
    F.scaled_dot_product_attention(is_causal=True) vs by-hand: max diff 0.000000
"""
import math

import torch
import torch.nn.functional as F

# Three tokens, hand-set. d_k = 4 (Q/K width), d_v = 2 (V width).
Q = torch.tensor([[1.0, 0.0, 1.0, 0.0],
                  [0.0, 1.0, 0.0, 1.0],
                  [1.0, 1.0, 0.0, 0.0]])
K = torch.tensor([[1.0, 0.0, 0.0, 1.0],
                  [0.0, 1.0, 1.0, 0.0],
                  [1.0, 0.0, 1.0, 0.0]])
V = torch.tensor([[1.0, 2.0],
                  [0.0, 1.0],
                  [3.0, 0.0]])


def show(label, t):
    print(f"  {label}")
    for r in t.tolist():
        print("    [" + ", ".join(f"{v:.4f}" for v in r) + "]")


# ---------- Part 1: the formula, step by step ----------
print("Part 1 - attention by hand: softmax(QK^T / sqrt(d_k)) V")
d_k = Q.shape[-1]
print(f"  Q {tuple(Q.shape)}, K {tuple(K.shape)}, V {tuple(V.shape)}"
      f" - d_k = {d_k}, sqrt(d_k) = {math.sqrt(d_k):.0f}")
scores = Q @ K.T                       # every query vs every key
show("scores = Q @ K^T (3, 3):", scores)
scaled = scores / math.sqrt(d_k)       # the scaling - Part 2 justifies it
show("scaled = scores / 2 (3, 3):", scaled)
weights = torch.softmax(scaled, dim=-1)          # Day 14, row by row
show("weights = softmax row by row (3, 3):", weights)
sums = ", ".join(f"{v:.4f}" for v in weights.sum(dim=-1).tolist())
print(f"  every weight row sums to {sums} - a budget (Day 14)")
out = weights @ V                      # blend the values by the weights
show("out = weights @ V (3, 2):", out)
# PyTorch's fused op. [None] adds the batch slot the API expects.
ref = F.scaled_dot_product_attention(Q[None], K[None], V[None])[0]
diff = (out - ref).abs().max().item()
print(f"  F.scaled_dot_product_attention vs by-hand: max diff {diff:.6f}")

# ---------- Part 2: why divide by sqrt(d_k) ----------
print("Part 2 - why divide by sqrt(d_k): dot products grow with d")
print("  q . k where every coordinate pair contributes 0.25:")
for d in (4, 64, 256):
    q = torch.full((d,), 0.5)
    k = torch.full((d,), 0.5)
    raw = (q * k).sum().item()                   # sum of d terms of 0.25
    sc = raw / math.sqrt(d)
    top_raw = torch.softmax(torch.tensor([raw, 0.0]), dim=0)[0].item()
    top_sc = torch.softmax(torch.tensor([sc, 0.0]), dim=0)[0].item()
    print(f"    d = {d:>3}: raw dot {raw:.2f} -> scaled {sc:.2f} | "
          f"top weight raw {top_raw:.4f} vs scaled {top_sc:.4f} | "
          f"grad p(1-p) raw {top_raw * (1 - top_raw):.6f} vs "
          f"scaled {top_sc * (1 - top_sc):.6f}")
print("  unscaled: winner takes all and the gradient is ~0 - "
      "nothing learns")
print("  scaled: still confident, gradient alive - that is the "
      "whole trick")

# ---------- Part 3: the causal mask ----------
print("Part 3 - causal mask: a decoder may not see the future")
print("  future scores (j > i) set to -inf before the softmax")
T = Q.shape[0]
future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
masked = scaled.masked_fill(future, float("-inf"))
weights_c = torch.softmax(masked, dim=-1)        # exp(-inf) = 0
show("causal weights (3, 3):", weights_c)
out_c = weights_c @ V
show("causal out (3, 2):", out_c)
print(f"  position 0 out == V row 0 exactly: "
      f"{bool(torch.equal(out_c[0], V[0]))} (it may only read itself)")
ref_c = F.scaled_dot_product_attention(Q[None], K[None], V[None],
                                       is_causal=True)[0]
diff_c = (out_c - ref_c).abs().max().item()
print(f"  F.scaled_dot_product_attention(is_causal=True) vs "
      f"by-hand: max diff {diff_c:.6f}")
