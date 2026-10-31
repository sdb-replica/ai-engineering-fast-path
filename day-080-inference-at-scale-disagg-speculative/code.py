"""Day 80 - Inference at Scale: Disaggregated Serving and
Speculative Decoding (a survey with a calculator).

Phase F. Phase E sized ONE engine (Days 64-66, 72): continuous
batching, quantization, the batch-vs-latency tradeoff on a
single pool of GPUs. Tonight is a survey of the two scale-out
patterns the frontier serving stacks (vLLM, TensorRT-LLM,
SGLang and friends) are converging on - patterns, not product
tours. Details vary by vendor; the shapes do not.

  Pattern 1  DISAGGREGATED PREFILL/DECODE. Day 66's two clocks
             have opposite appetites: prefill is compute-bound
             (the whole prompt in parallel, big matmuls), while
             decode is bandwidth-bound (one token per step, the
             step dominated by re-reading weights + KV). In one
             engine they share GPUs, so a long prompt's prefill
             stalls everyone's decode. Split the two clocks onto
             separate pools and each scales and schedules on its
             own curve - the price is a handoff: the prompt's
             KV cache (Day 48) must move from the prefill pool
             to the decode pool before the first decode step.
             Part 2 prices that handoff.

  Pattern 2  SPECULATIVE DECODING. Decode is serial: token n+1
             needs token n. So cheat the way a CPU does - a tiny
             DRAFT model (branch prediction for tokens) guesses
             the next k tokens autoregressively, which is cheap
             because it is small; then the big TARGET model
             checks all k guesses in ONE parallel pass, exactly
             the kind of pass decode never gets to make. Accept
             the longest prefix the target agrees with; at the
             first disagreement, resample that one token from
             the target's own distribution; if all k are
             accepted, the pass also yields one bonus token.
             With the standard accept/reject rule - accept each
             draft token with probability min(1, p_target /
             p_draft) - the output distribution is EXACTLY the
             target model's own. The speedup is lossless:
             faster, not different. (Stated, not proved - the
             one-line rule above is what makes it exact.)
             Part 1 prices it: expected tokens per target pass
             E = 1 + a + a^2 + ... + a^k
               = (1 - a^(k+1)) / (1 - a),
             where a (alpha) is the per-token acceptance rate.

Stated constants, so every number is auditable:
  k = 4 draft tokens per cycle; one draft pass costs c = 0.10
  of one target pass (a roughly 10x cheaper model); alpha in
  {0.60, 0.75, 0.90}. KV sizing reuses Day 48/49's model:
  32 layers, d_model 4096, fp16 -> 524,288 bytes of KV per
  token. Interconnects: NVLink 900 GB/s, PCIe 5.0 x16 64 GB/s.
  One more anchor: Day 66's operating point, TPOT 40 ms =
  25 tokens/sec per user.

Runs on the ~/ai-lab venv: pure standard library - no torch,
no GPU, no paid APIs, no downloads. Deterministic arithmetic
only, so every digit below is exact.

Expected output:

  Part 1 - speculative decoding: what one verify pass buys
    naive decode: 1 token per target pass, 1.00 pass-units per token
    spec cycle: 1 verify pass + 4 draft passes x 0.10 = 1.40 pass-units
    alpha 0.60: E[tokens] 2.3056 per cycle -> speedup 1.65x
    alpha 0.75: E[tokens] 3.0508 per cycle -> speedup 2.18x
    alpha 0.90: E[tokens] 4.0951 per cycle -> speedup 2.93x
    ceiling: alpha 1.00 gives k+1 = 5 tokens per cycle; a FREE draft
      (c = 0) at alpha 0.75 would give 3.05x - draft cost is real
    break-even: first alpha on a 0.05 grid whose speedup beats 1.00x: 0.30
    at Day 66's point (TPOT 40 ms = 25 tok/s per user), alpha 0.75
      streams ~54.5 tok/s to that user - same distribution, faster drip
    the guarantee: accept/reject keeps the output distribution exactly
      the target model's own - the draft only ever proposes
  Part 2 - disaggregation: the KV handoff tax (Day 48's recipe)
    KV per token: 2 (K and V) x 32 layers x 4096 x 2 bytes = 524,288 bytes
    context 2,048 tokens: KV = 1.07 GB -> NVLink 900 GB/s: 1.19 ms | PCIe 64 GB/s: 16.78 ms
    context 32,768 tokens: KV = 17.18 GB -> NVLink 900 GB/s: 19.09 ms | PCIe 64 GB/s: 268.44 ms
    the tax is paid ONCE per request, before decode step one - it lands on TTFT (Day 66)
    at 32k context the PCIe handoff alone is 6.7x the whole 40 ms TPOT budget
    verdict: disagg pays when the fabric is fast (NVLink-class) or prompts
      are short, and wins most when the mix is lopsided - huge prompts,
      short answers (or the reverse). On a slow link, keep the clocks together.
"""

# ---------- Part 1: speculative decoding, priced ----------
K = 4            # draft tokens proposed per cycle
C_DRAFT = 0.10   # one draft pass, in target-pass units


def expected_tokens(alpha, k):
    """E[tokens per cycle] = 1 + a + a^2 + ... + a^k: the bonus
    token always lands (1), plus each draft token that survives
    every check before it (a^i). Closed form of the same sum."""
    return (1 - alpha ** (k + 1)) / (1 - alpha)


cycle_cost = 1.0 + K * C_DRAFT
print("Part 1 - speculative decoding: what one verify pass buys")
print("  naive decode: 1 token per target pass, "
      "1.00 pass-units per token")
print(f"  spec cycle: 1 verify pass + {K} draft passes x "
      f"{C_DRAFT:.2f} = {cycle_cost:.2f} pass-units")
for alpha in (0.60, 0.75, 0.90):
    e = expected_tokens(alpha, K)
    print(f"  alpha {alpha:.2f}: E[tokens] {e:.4f} per cycle "
          f"-> speedup {e / cycle_cost:.2f}x")
e75 = expected_tokens(0.75, K)
print(f"  ceiling: alpha 1.00 gives k+1 = {K + 1} tokens per cycle; "
      "a FREE draft")
print(f"    (c = 0) at alpha 0.75 would give {e75:.2f}x "
      "- draft cost is real")
grid = [i / 20 for i in range(1, 20)]          # 0.05, 0.10, ..., 0.95
breakeven = next(a for a in grid
                 if expected_tokens(a, K) / cycle_cost > 1.0)
print(f"  break-even: first alpha on a 0.05 grid whose speedup "
      f"beats 1.00x: {breakeven:.2f}")
print("  at Day 66's point (TPOT 40 ms = 25 tok/s per user), "
      "alpha 0.75")
print(f"    streams ~{25 * e75 / cycle_cost:.1f} tok/s to that user "
      "- same distribution, faster drip")
print("  the guarantee: accept/reject keeps the output distribution "
      "exactly")
print("    the target model's own - the draft only ever proposes")

# ---------- Part 2: disaggregation, the handoff tax ----------
LAYERS, D_MODEL, BYTES = 32, 4096, 2      # Day 48/49 model, fp16
KV_PER_TOKEN = 2 * LAYERS * D_MODEL * BYTES

print("Part 2 - disaggregation: the KV handoff tax (Day 48's recipe)")
print(f"  KV per token: 2 (K and V) x {LAYERS} layers x {D_MODEL} "
      f"x {BYTES} bytes = {KV_PER_TOKEN:,} bytes")
for tokens in (2_048, 32_768):
    kv = KV_PER_TOKEN * tokens
    nvlink_ms = kv / (900 * 1e9) * 1000
    pcie_ms = kv / (64 * 1e9) * 1000
    print(f"  context {tokens:,} tokens: KV = {kv / 1e9:.2f} GB "
          f"-> NVLink 900 GB/s: {nvlink_ms:.2f} ms "
          f"| PCIe 64 GB/s: {pcie_ms:.2f} ms")
print("  the tax is paid ONCE per request, before decode step one "
      "- it lands on TTFT (Day 66)")
pcie_32k = KV_PER_TOKEN * 32_768 / (64 * 1e9) * 1000
print(f"  at 32k context the PCIe handoff alone is "
      f"{pcie_32k / 40:.1f}x the whole 40 ms TPOT budget")
print("  verdict: disagg pays when the fabric is fast (NVLink-class) "
      "or prompts")
print("    are short, and wins most when the mix is lopsided - huge "
      "prompts,")
print("    short answers (or the reverse). On a slow link, keep the "
      "clocks together.")
