"""Day 39 - Multi-Head Attention: Why Multiple Heads.

Day 38 built ONE attention head: project X into Q, K, V, take
softmax(Q K^T / sqrt(d_k)) V. It works - but one head means one
set of projections, which means every token gets exactly ONE
weighted average of the sequence, computed in ONE subspace.
One read of the chart.

Multi-head attention runs h copies of that same mechanism in
parallel, each with its OWN W_Q, W_K, W_V. Head i only sees a
slice of the model dimension: with d_model = 4 and h = 2, each
head works in d_k = 2. The heads' outputs are concatenated
back to d_model and mixed by one final projection, W_O.

  Part 1  Two hand-built heads on a 3-token sequence that
          provably attend differently. Head 1's projections
          read only the position features, and its scores work
          out to (i-1)^2 - (i-1-j)^2: maximized at j = i-1,
          the PREVIOUS token. Head 2's query is a constant
          "find the flagged token" and its keys read only the
          flag, so every row peaks at token 0, the subject.
          Same input, two different reads - that is the point.
  Part 2  Concatenate the head outputs and apply W_O, the
          mixer that lets the heads' findings blend before
          anything downstream sees them.
  Part 3  The split is free. Shapes: d_model splits into
          h x d_k and glues back exactly. Parameters: the two
          small heads + W_O cost 80 - identical to ONE big
          head with d_k = d_model. And the per-head Q
          projections stack into a single (4, 4) W_Q whose
          one matmul, split afterwards, reproduces both
          heads' Qs exactly: in practice it is one fused
          projection, sliced.
  Part 4  Copy the same hand-set weights into
          nn.MultiheadAttention (its in_proj_weight is exactly
          [W_Q; W_K; W_V] stacked, so the copy is
          deterministic) and check the outputs agree.

Runs on the ~/ai-lab venv: torch only (free), CPU on
purpose (tiny data, exact digits), no CUDA, no paid APIs,
no downloads. Every weight is set by hand - no RNG anywhere.

Expected output:

  Part 1 - one sequence, two heads, two attention patterns
    head 1 (previous-token head) weights:
      token 0 reads: 0.8902, 0.1067, 0.0031  -> peak token 0
      token 1 reads: 0.6443, 0.3177, 0.0381  -> peak token 0
      token 2 reads: 0.2483, 0.5035, 0.2483  -> peak token 1
    head 1 output rows: [1.8964, 0.8902] / [1.7204, 0.6443] / [1.7448, 0.2483]
    hand softmax vs F.scaled_dot_product_attention (head 1): max diff 0.000000
    head 2 (subject head) weights - every row identical:
      token 0 reads: 0.8066, 0.0967, 0.0967  -> peak token 0
      token 1 reads: 0.8066, 0.0967, 0.0967  -> peak token 0
      token 2 reads: 0.8066, 0.0967, 0.0967  -> peak token 0
    head 2 output rows: [0.8066, 2.0000] / [0.8066, 2.0000] / [0.8066, 2.0000]
    heads disagree: mean |w1 - w2| = 0.1940 - same input, different reads
  Part 2 - concat + W_O: the mixer
    concat rows: [1.8964, 0.8902, 0.8066, 2.0000] / [1.7204, 0.6443, 0.8066, 2.0000] / [1.7448, 0.2483, 0.8066, 2.0000]
    output rows: [2.2997, 1.8902, 1.7548, 2.4451] / [2.1237, 1.6443, 1.6668, 2.3221] / [2.1481, 1.2483, 1.6790, 2.1241]
  Part 3 - the split is free: shapes and parameter accounting
    shapes: X (3, 4) -> Q/K/V per head (3, 2) -> weights (3, 3) -> head out (3, 2) -> concat (3, 4) -> output (3, 4)
    fused W_Q (4, 4), one matmul, split afterwards: max diff vs per-head Q = 0.000000
    params: 2 heads x 30 = 60, + W_O 20 = 80 total
            one big head (d_k = 4): Q/K/V 60 + W_O 20 = 80 total
    verdict: splitting d_model into h heads costs the same as one wide head - you buy viewpoints, not parameters
  Part 4 - the same weights inside nn.MultiheadAttention
    matches the hand-built output (atol 1e-05): True
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# 3 tokens, d_model = 4. Features per token: [content, flag, pos, pos^2].
# Token 0 is the flagged "subject"; pos / pos^2 are position features
# (real models inject richer ones - Day 41's topic; today they are
# just input columns the projections can read).
X = torch.tensor([[2.0, 1.0, 0.0, 0.0],
                  [1.0, 0.0, 1.0, 1.0],
                  [3.0, 0.0, 2.0, 4.0]])
D_MODEL, H = 4, 2
D_K = D_MODEL // H                                   # 2 per head

# Head 1 - the previous-token head. Its Q reads pos (plus bias),
# its K reads pos and pos^2, so score(i, j) = (i-1)^2 - (i-1-j)^2
# up to the constant (i-1)^2 softmax ignores: peak at j = i - 1.
W_Q1 = torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
B_Q1 = torch.tensor([-1.0, 1.0])
W_K1 = torch.tensor([[0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, -1.0]])
B_K1 = torch.tensor([0.0, 0.0])
W_V1 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
B_V1 = torch.tensor([0.0, 0.0])

# Head 2 - the subject head. Its Q is a constant (bias only):
# every token asks the same question, "where is the flag?" Its K
# reads only the flag column, so every row peaks at token 0.
W_Q2 = torch.zeros(2, 4)
B_Q2 = torch.tensor([3.0, 0.0])
W_K2 = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
B_K2 = torch.tensor([0.0, 0.0])
W_V2 = torch.tensor([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
B_V2 = torch.tensor([0.0, 0.0])

# W_O: the output projection (mixer) applied after concatenation.
W_O = torch.tensor([[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.5],
                    [0.5, 0.0, 1.0, 0.0], [0.0, 0.5, 0.0, 1.0]])
B_O = torch.zeros(4)

fmt = lambda t: ", ".join(f"{v:.4f}" for v in t.flatten().tolist())


def attention(X, W_Q, B_Q, W_K, B_K, W_V, B_V):
    """Day 38's single head, unchanged: project, score, softmax, mix."""
    Q = X @ W_Q.T + B_Q
    K = X @ W_K.T + B_K
    V = X @ W_V.T + B_V
    weights = torch.softmax(Q @ K.T / math.sqrt(D_K), dim=-1)
    return Q, K, V, weights, weights @ V


# ---------- Part 1: two heads, two attention patterns ----------
print("Part 1 - one sequence, two heads, two attention patterns")
Q1, K1, V1, w1, out1 = attention(X, W_Q1, B_Q1, W_K1, B_K1, W_V1, B_V1)
Q2, K2, V2, w2, out2 = attention(X, W_Q2, B_Q2, W_K2, B_K2, W_V2, B_V2)
print("  head 1 (previous-token head) weights:")
for i, row in enumerate(w1):
    print(f"    token {i} reads: {fmt(row)}  -> peak token "
          f"{int(row.argmax())}")
print(f"  head 1 output rows: [{fmt(out1[0])}] / [{fmt(out1[1])}] / "
      f"[{fmt(out1[2])}]")
with torch.no_grad():
    sdpa_out1 = F.scaled_dot_product_attention(Q1, K1, V1)
print("  hand softmax vs F.scaled_dot_product_attention (head 1): "
      f"max diff {(sdpa_out1 - out1).abs().max().item():.6f}")
print("  head 2 (subject head) weights - every row identical:")
for i, row in enumerate(w2):
    print(f"    token {i} reads: {fmt(row)}  -> peak token "
          f"{int(row.argmax())}")
print(f"  head 2 output rows: [{fmt(out2[0])}] / [{fmt(out2[1])}] / "
      f"[{fmt(out2[2])}]")
print(f"  heads disagree: mean |w1 - w2| = "
      f"{(w1 - w2).abs().mean().item():.4f} - same input, different reads")

# ---------- Part 2: concat + W_O ----------
print("Part 2 - concat + W_O: the mixer")
concat = torch.cat([out1, out2], dim=-1)             # (3, 2 + 2) = (3, 4)
output = concat @ W_O.T + B_O
print(f"  concat rows: [{fmt(concat[0])}] / [{fmt(concat[1])}] / "
      f"[{fmt(concat[2])}]")
print(f"  output rows: [{fmt(output[0])}] / [{fmt(output[1])}] / "
      f"[{fmt(output[2])}]")

# ---------- Part 3: the split is free ----------
print("Part 3 - the split is free: shapes and parameter accounting")
print(f"  shapes: X {tuple(X.shape)} -> Q/K/V per head {tuple(Q1.shape)}"
      f" -> weights {tuple(w1.shape)} -> head out {tuple(out1.shape)}"
      f" -> concat {tuple(concat.shape)} -> output {tuple(output.shape)}")
W_Q_FUSED = torch.cat([W_Q1, W_Q2], dim=0)           # one (4, 4) matrix
B_Q_FUSED = torch.cat([B_Q1, B_Q2])
Q_all = X @ W_Q_FUSED.T + B_Q_FUSED
split_diff = max((Q_all[:, :D_K] - Q1).abs().max().item(),
                 (Q_all[:, D_K:] - Q2).abs().max().item())
print(f"  fused W_Q (4, 4), one matmul, split afterwards: max diff vs "
      f"per-head Q = {split_diff:.6f}")


def n_params(*tensors):
    return sum(t.numel() for t in tensors)


per_head = n_params(W_Q1, B_Q1, W_K1, B_K1, W_V1, B_V1)
wo = n_params(W_O, B_O)
big_qkv = 3 * (D_MODEL * D_MODEL + D_MODEL)          # one head, d_k = 4
print(f"  params: {H} heads x {per_head} = {H * per_head}, + W_O {wo}"
      f" = {H * per_head + wo} total")
print(f"          one big head (d_k = 4): Q/K/V {big_qkv} + W_O {wo}"
      f" = {big_qkv + wo} total")
print("  verdict: splitting d_model into h heads costs the same as one"
      " wide head - you buy viewpoints, not parameters")

# ---------- Part 4: nn.MultiheadAttention with the same weights ----------
print("Part 4 - the same weights inside nn.MultiheadAttention")
mha = nn.MultiheadAttention(embed_dim=D_MODEL, num_heads=H,
                            bias=True, batch_first=True)
with torch.no_grad():
    mha.in_proj_weight.copy_(torch.cat([
        W_Q_FUSED,
        torch.cat([W_K1, W_K2], dim=0),
        torch.cat([W_V1, W_V2], dim=0)]))
    mha.in_proj_bias.copy_(torch.cat([
        B_Q_FUSED,
        torch.cat([B_K1, B_K2]),
        torch.cat([B_V1, B_V2])]))
    mha.out_proj.weight.copy_(W_O)
    mha.out_proj.bias.copy_(B_O)
mha.eval()
with torch.no_grad():
    mha_out, _ = mha(X.unsqueeze(0), X.unsqueeze(0), X.unsqueeze(0),
                     need_weights=False)
print("  matches the hand-built output (atol 1e-05): "
      f"{bool(torch.allclose(mha_out.squeeze(0), output, atol=1e-5))}")
