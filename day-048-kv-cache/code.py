"""Day 48 - The KV Cache: Never Recompute the Past.

Day 47's generation loop produces one token at a time - and
pays for every one of them with a full forward pass over the
whole prefix: every Q, K, V projection recomputed, the full
L x L score grid rebuilt, for a step whose only genuinely new
input is ONE token. Tonight that waste dies.

The invariant: token j's K and V are computed from token j
(and its position) only, and the causal mask guarantees
nothing about them can depend on tokens after j - in a
multi-layer stack the same holds layer by layer, because no
layer ever lets a position look forward. So a past token's
K and V can NEVER change when a new token is appended.
Cache them. Each decode step then projects only the newest
token, appends its K/V to the cache, and runs ONE query
against everything stored: L score dots instead of L^2,
1 projection instead of L.

  Part 1  The proof: generate 4 tokens the naive way and
          the cached way with the same hand-set tiny causal
          attention (one layer, one head, d_model 4, greedy
          argmax - no RNG). Identical tokens, per-step
          outputs matching to 0.000000, and the cache's
          contents matching a fresh recompute of K/V to
          0.000000.
  Part 2  The work: per-step counts, measured in the code.
          A naive step costs L^2 score dots and L token
          projections; a cached step costs L and 1. Prefill
          (the prompt's first pass) is the one step where
          both ways pay the same - that is what fills the
          cache.
  Part 3  The bill: cache memory is 2 (K and V) x layers x
          d_model values per token, append-only, forever.
          Toy numbers first, then a realistic config -
          32 layers, d_model 4096, fp16 - where one token
          costs about half a megabyte and 1,000 tokens cost
          over half a gigabyte.

Prefill vs decode, in one breath: prefill processes the
whole prompt in parallel (big matmuls, compute-bound - it
is the slow first token); decode then emits one token per
step with a tiny matvec that must re-read the entire cache
from memory (memory-bandwidth-bound). Full serving
treatment is Phase E; tonight is the cache itself.

Runs on the ~/ai-lab venv: torch only (free), MPS if
available else CPU, no CUDA, no paid APIs, no downloads.
Every weight is set by hand - no RNG anywhere, and greedy
argmax margins are wide (> 0.25), so every printed digit
is exact on either device.

Expected output:

  Part 1 - two ways to generate, one answer (greedy argmax, no RNG)
    prompt: [0, 1, 2, 3]
    naive   generated: [0, 1, 2, 3]
    cached  generated: [0, 1, 2, 3]
    per-step outputs, cached vs naive: max diff 0.000000
    cache K/V vs fresh recompute at the end: max diff 0.000000
    why exactly zero: causality - a past token's K and V cannot see
      the future, so they can never change; only the newest token is new
  Part 2 - attention work per step (1 score = one q.k dot product)
    step 1 (prefill, 4 tokens in): naive 16 scores, 4 projected | cached 16 scores, 4 projected
    step 2 (decode, 5 tokens in): naive 25 scores, 5 projected | cached  5 scores, 1 projected
    step 3 (decode, 6 tokens in): naive 36 scores, 6 projected | cached  6 scores, 1 projected
    step 4 (decode, 7 tokens in): naive 49 scores, 7 projected | cached  7 scores, 1 projected
    totals: naive 126 scores / 22 projected - cached 34 scores / 7 projected (3.7x less attention work)
    the pattern: a naive step costs L^2 and grows; a cached step costs L - one query row
  Part 3 - the bill: cache memory = 2 (K and V) x layers x d_model values per token
    toy (1 layer, d_model 4, fp32): 8 values = 32 bytes per token
      cache after generating: 7 tokens stored -> 56 values = 224 bytes
    real (32 layers, d_model 4096, fp16): 262,144 values = 524,288 bytes per token (~0.5 MB)
      1,000 tokens: 524,288,000 bytes = 524.3 MB
      100,000 tokens: 52,428,800,000 bytes = 52.4 GB - the cache becomes the memory wall
    verdict: the cache trades memory for compute - recompute nothing, store everything
"""
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Toy config: vocab 6, d_model 4, ONE layer, ONE head.
# Every weight hand-set below - this exact toy cycled cleanly
# in verification (generated tokens repeat the prompt's cycle).
E = torch.tensor([                     # token embeddings (Day 15)
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.5, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.5]], device=device)
POS = torch.tensor([                   # positional vectors (Day 41),
    [0.6, 0.0, 0.0, 0.0],              # cycling with period 4
    [0.0, 0.6, 0.0, 0.0],
    [0.0, 0.0, 0.6, 0.0],
    [0.0, 0.0, 0.0, 0.6]], device=device)
WQ = torch.tensor([
    [1.0, 0.0, 0.5, 0.0],
    [0.0, 1.0, 0.0, 0.5],
    [0.5, 0.0, 1.0, 0.0],
    [0.0, 0.5, 0.0, 1.0]], device=device)
WK = torch.tensor([
    [1.0, 0.5, 0.0, 0.0],
    [0.5, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.5],
    [0.0, 0.0, 0.5, 1.0]], device=device)
WV = torch.tensor([
    [1.0, 0.0, 0.0, 0.5],
    [0.0, 1.0, 0.5, 0.0],
    [0.0, 0.5, 1.0, 0.0],
    [0.5, 0.0, 0.0, 1.0]], device=device)
WO = torch.tensor([                    # head: d_model -> vocab logits;
    [0.0, 2.0, 0.0, 0.0, 1.0, 0.0],    # dim i votes for token i+1,
    [0.0, 0.0, 2.0, 0.0, 1.0, 0.0],    # so the toy cycles 0,1,2,3
    [0.0, 0.0, 0.0, 2.0, 0.0, 1.0],
    [2.0, 0.0, 0.0, 0.0, 0.0, 1.0]], device=device)
D = 4
PROMPT = [0, 1, 2, 3]
N_GEN = 4


def softmax_rows(S):
    """Day 14's softmax, row-wise, max-subtracted for stability."""
    S = S - S.max(dim=-1, keepdim=True).values
    e = torch.exp(S)
    return e / e.sum(dim=-1, keepdim=True)


def embed(seq):
    idx = torch.tensor(seq, device=device)
    pos = torch.arange(len(seq), device=device) % 4
    return E[idx] + POS[pos]


def full_forward(seq):
    """Day 47's way: project EVERYTHING, full masked grid.
    Returns (outputs, K, V) for the whole sequence."""
    X = embed(seq)
    Q, K, V = X @ WQ.T, X @ WK.T, X @ WV.T
    S = Q @ K.T / D ** 0.5
    mask = torch.triu(torch.ones(len(seq), len(seq), dtype=torch.bool,
                                 device=device), diagonal=1)
    S = torch.where(mask, torch.tensor(-1e9, device=device), S)
    return softmax_rows(S) @ V, K, V


def generate_naive(prompt, n_gen, stats):
    """Every step recomputes the whole prefix from scratch."""
    seq, outs = list(prompt), []
    for _ in range(n_gen):
        out, _, _ = full_forward(seq)
        outs.append(out[-1])
        stats.append((len(seq), len(seq) ** 2, len(seq)))
        seq.append(int(torch.argmax(out[-1] @ WO).item()))
    return seq[len(prompt):], outs


def generate_cached(prompt, n_gen, stats):
    """Prefill once, then per step: project ONE token, append
    its K/V, run ONE query against the cache."""
    seq, outs = list(prompt), []
    out, K, V = full_forward(seq)                    # prefill
    outs.append(out[-1])
    stats.append((len(seq), len(seq) ** 2, len(seq)))
    seq.append(int(torch.argmax(out[-1] @ WO).item()))
    for _ in range(n_gen - 1):                       # decode steps
        x = E[seq[-1]] + POS[(len(seq) - 1) % 4]     # newest token only
        q = x @ WQ.T                                 # ONE new query
        K = torch.cat([K, (x @ WK.T).unsqueeze(0)])  # append-only:
        V = torch.cat([V, (x @ WV.T).unsqueeze(0)])  # past rows untouched
        out = softmax_rows(q @ K.T / D ** 0.5) @ V
        outs.append(out)
        stats.append((len(seq), len(seq), 1))
        seq.append(int(torch.argmax(out @ WO).item()))
    return seq[len(prompt):], outs, K, V


# ---------- Part 1: the proof - same answers, two ways ----------
print("Part 1 - two ways to generate, one answer (greedy argmax, no RNG)")
stats_n, stats_c = [], []
gen_n, outs_n = generate_naive(PROMPT, N_GEN, stats_n)
gen_c, outs_c, K_cache, V_cache = generate_cached(PROMPT, N_GEN, stats_c)
print(f"  prompt: {PROMPT}")
print(f"  naive   generated: {gen_n}")
print(f"  cached  generated: {gen_c}")
with torch.no_grad():
    odiff = max((a - b).abs().max().item()
                for a, b in zip(outs_n, outs_c))
    _, K_fresh, V_fresh = full_forward((PROMPT + gen_c)[:-1])
    kdiff = max((K_cache - K_fresh).abs().max().item(),
                (V_cache - V_fresh).abs().max().item())
print(f"  per-step outputs, cached vs naive: max diff {odiff:.6f}")
print(f"  cache K/V vs fresh recompute at the end: max diff {kdiff:.6f}")
print("  why exactly zero: causality - a past token's K and V cannot see")
print("    the future, so they can never change; "
      "only the newest token is new")

# ---------- Part 2: the work ----------
print("Part 2 - attention work per step (1 score = one q.k dot product)")
for i, ((L, ns, nproj), (_, cs, cproj)) in enumerate(
        zip(stats_n, stats_c), start=1):
    kind = "prefill" if i == 1 else "decode"
    print(f"  step {i} ({kind}, {L} tokens in): naive {ns:>2} scores, "
          f"{nproj} projected | cached {cs:>2} scores, {cproj} projected")
tn = sum(s[1] for s in stats_n); tc = sum(s[1] for s in stats_c)
tp = sum(s[2] for s in stats_n); cp = sum(s[2] for s in stats_c)
print(f"  totals: naive {tn} scores / {tp} projected - "
      f"cached {tc} scores / {cp} projected "
      f"({tn / tc:.1f}x less attention work)")
print("  the pattern: a naive step costs L^2 and grows; "
      "a cached step costs L - one query row")

# ---------- Part 3: the bill ----------
print("Part 3 - the bill: cache memory = "
      "2 (K and V) x layers x d_model values per token")
toy_vals = 2 * 1 * D                       # 1 layer, d_model 4
rows = K_cache.shape[0]                    # tokens actually stored
print(f"  toy (1 layer, d_model 4, fp32): {toy_vals} values = "
      f"{toy_vals * 4} bytes per token")
print(f"    cache after generating: {rows} tokens stored -> "
      f"{rows * toy_vals} values = {rows * toy_vals * 4} bytes")
real_vals = 2 * 32 * 4096                  # 32 layers, d_model 4096
real_bytes = real_vals * 2                 # fp16 = 2 bytes per value
print(f"  real (32 layers, d_model 4096, fp16): {real_vals:,} values = "
      f"{real_bytes:,} bytes per token (~0.5 MB)")
print(f"    1,000 tokens: {1000 * real_bytes:,} bytes = "
      f"{1000 * real_bytes / 1e6:.1f} MB")
print(f"    100,000 tokens: {100000 * real_bytes:,} bytes = "
      f"{100000 * real_bytes / 1e9:.1f} GB - "
      "the cache becomes the memory wall")
print("  verdict: the cache trades memory for compute - "
      "recompute nothing, store everything")
