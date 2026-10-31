"""Day 49 - Context Windows & Why Long Context Is Expensive.

Day 48 gave you the KV cache: past keys and values are kept so
each new token only attends to stored history. Tonight is the
bill for the whole window - the two costs that scale with the
context length T, for ONE forward pass over a prompt of T
tokens (prefill), batch size 1, on a fixed model:

    L = 32 layers, d = 4096 (model width), h = 32 heads,
    MLP hidden size 4d (the standard expansion).

The formulas (stated, so every number below is auditable):

  Score entries   Each head builds a T x T score matrix, so
                  entries = L * h * T^2. Quadratic in T.
  Attention FLOPs The two big matmuls per layer: QK^T and
                  (scores)V. A matmul of (T x d) by (d x T)
                  costs 2*T^2*d FLOPs (one multiply + one add
                  per term), so attention = 4 * T^2 * d per
                  layer, times L layers. Quadratic in T.
  MLP FLOPs       Two matmuls per layer, d -> 4d and 4d -> d:
                  2*T*d*4d + 2*T*4d*d = 16 * T * d^2 per
                  layer, times L. Only linear in T.
  KV cache bytes  K and V for every token in every layer,
                  fp16 (2 bytes): 2 * L * T * d * 2. Linear
                  in T - but the constant is huge.

  Part 1  The doubling rule. Take T from 4K to 8K and divide:
          attention cost x4, MLP cost x2, cache x2. Doubling
          the context is never a x2 decision for attention.
  Part 2  The cost table at T = 1K / 8K / 32K / 128K / 1M:
          score entries, attention vs MLP FLOPs, and the
          cache in GiB. Watch which column explodes.
  Part 3  The crossover. Solve 4*T^2*d = 16*T*d^2 on paper:
          T* = 4d = 16,384. Below it the MLP is the bigger
          bill; above it attention takes over, and by 1M it
          is 64x the MLP. There is no regime where a longer
          window gets cheaper per token.

Pure arithmetic - no torch needed, standard library only.
Runs anywhere Python 3.11+ runs, including the ~/ai-lab
venv on the MacBook. No CUDA, no paid APIs, no downloads,
no RNG: every number is exact integer arithmetic.

Expected output:

  Part 1 - the doubling rule: T = 4,096 -> 8,192
    attention FLOPs  x4.00   (quadratic: 2^2)
    MLP FLOPs        x2.00   (linear)
    KV cache         x2.00   (linear, but wait for Part 2)
  Part 2 - the bill at five context lengths (L=32, d=4096, h=32, fp16 cache)
      T         scores    attn     MLP  attn/MLP   KV cache
      1,024     1.07e+09    0.5T    8.8T     0.06     0.5 GiB
      8,192     6.87e+10   35.2T   70.4T     0.50     4.0 GiB
     32,768     1.10e+12  562.9T  281.5T     2.00    16.0 GiB
    131,072     1.76e+13    9.0P    1.1P     8.00    64.0 GiB
    1,048,576   1.13e+15  576.5P    9.0P    64.00   512.0 GiB
  Part 3 - the crossover: where attention catches the MLP
    4*T^2*d = 16*T*d^2  ->  T* = 4d = 16,384 tokens
    check at T*: attn 140.7T FLOPs = MLP 140.7T FLOPs  (equal: True)
    at T = 1M attention is 64.00x the MLP - and still climbing
    cache check: 0.5 MiB per token x 131,072 tokens = 64.0 GiB at 128K
    verdict: the window is a cost curve, not a spec - 4x attention compute and 2x cache for every 2x of context
"""
# The model - fixed all lesson. Roughly a 7B-class transformer
# with the standard 4x MLP expansion.
LAYERS = 32
D_MODEL = 4096
HEADS = 32
FP16_BYTES = 2

CONTEXTS = [1_024, 8_192, 32_768, 131_072, 1_048_576]


def score_entries(T):
    """Every head, every layer: one T x T matrix of scores."""
    return LAYERS * HEADS * T * T


def attn_flops(T):
    """QK^T plus (scores)V, all layers: 4 * T^2 * d per layer."""
    return 4 * T * T * D_MODEL * LAYERS


def mlp_flops(T):
    """d -> 4d -> d, all layers: 16 * T * d^2 per layer."""
    return 16 * T * D_MODEL * D_MODEL * LAYERS


def kv_bytes(T):
    """K and V, every token, every layer, stored in fp16."""
    return 2 * LAYERS * T * D_MODEL * FP16_BYTES


def fmt_flops(f):
    """Human units: T = teraFLOP (1e12), P = petaFLOP (1e15)."""
    if f >= 1e15:
        return f"{f / 1e15:.1f}P"
    return f"{f / 1e12:.1f}T"


def fmt_gib(b):
    return f"{b / 2**30:.1f} GiB"


# ---------- Part 1: the doubling rule ----------
print("Part 1 - the doubling rule: T = 4,096 -> 8,192")
t_small, t_big = 4_096, 8_192
print(f"  attention FLOPs  x{attn_flops(t_big) / attn_flops(t_small):.2f}"
      "   (quadratic: 2^2)")
print(f"  MLP FLOPs        x{mlp_flops(t_big) / mlp_flops(t_small):.2f}"
      "   (linear)")
print(f"  KV cache         x{kv_bytes(t_big) / kv_bytes(t_small):.2f}"
      "   (linear, but wait for Part 2)")

# ---------- Part 2: the bill at five context lengths ----------
print("Part 2 - the bill at five context lengths "
      "(L=32, d=4096, h=32, fp16 cache)")
print(f"  {'T':>9} {'scores':>10} {'attn':>7} {'MLP':>7}"
      f" {'attn/MLP':>8} {'KV cache':>9}")
for T in CONTEXTS:
    print(f"  {T:>9,} {score_entries(T):>10.2e}"
          f" {fmt_flops(attn_flops(T)):>7} {fmt_flops(mlp_flops(T)):>7}"
          f" {attn_flops(T) / mlp_flops(T):>8.2f}"
          f" {fmt_gib(kv_bytes(T)):>9}")

# ---------- Part 3: the crossover, and the cache constant ----------
print("Part 3 - the crossover: where attention catches the MLP")
t_star = 4 * D_MODEL                     # from 4*T^2*d = 16*T*d^2
print(f"  4*T^2*d = 16*T*d^2  ->  T* = 4d = {t_star:,} tokens")
print(f"  check at T*: attn {fmt_flops(attn_flops(t_star))} FLOPs"
      f" = MLP {fmt_flops(mlp_flops(t_star))} FLOPs"
      f"  (equal: {attn_flops(t_star) == mlp_flops(t_star)})")
big = CONTEXTS[-1]
print(f"  at T = 1M attention is "
      f"{attn_flops(big) / mlp_flops(big):.2f}x the MLP"
      " - and still climbing")
per_token_mib = kv_bytes(1) / 2**20
print(f"  cache check: {per_token_mib} MiB per token"
      f" x 131,072 tokens = {fmt_gib(kv_bytes(131_072))} at 128K")
print("  verdict: the window is a cost curve, not a spec - "
      "4x attention compute and 2x cache for every 2x of context")
