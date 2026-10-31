"""Day 51 - Pretraining: What It Actually Is.

Phase D opener. There is no new training loop tonight.
Pretraining is the Day 42 loop - read text, predict the next
token, cross-entropy, backward, step - run at civilizational
scale. What changes is not the algorithm; it is three dials:

    N  parameters        (Day 42: 814,643 - GPT-2: billions)
    D  training tokens   (Day 42: 2.4K chars, seen ~1,043x -
                          frontier runs: trillions, seen ~once)
    C  compute, in FLOPs (floating-point operations)

and the three are welded together by one rule of thumb:

    C ~ 6 * N * D

Why 6? One token's forward pass costs about 2 FLOPs per
parameter (a multiply + an add for every weight it touches,
Day 12), and the backward pass costs about twice the
forward (Day 23's blame has to flow through every weight
again, both ways). 2 + 4 = 6 FLOPs per parameter per token.
It is an approximation - it ignores attention's own term -
and it is the formula the whole industry budgets with.

  Part 1  Your Day 42 run, priced. Tokens processed =
          1,200 steps x batch 32 x 64 chars. Same corpus
          recycled ~1,043 times: that is why it memorized.
  Part 2  Two published configs, priced with the same
          formula. GPT-2 XL (1.5B params, ~10B tokens,
          reported figures) and a Llama-3-class run (8B
          params, 15T tokens, reported "over 15 trillion").
          Ratios vs Day 42 are the lesson: scale is not
          a dial you nudge, it is orders of magnitude.
  Part 3  GPU time. At one stated assumption - an H100
          sustaining 3.0e14 FLOP/s in a real training run
          (roughly 30% of its peak; big models never see
          peak) - turn FLOPs into GPU-days.
  Part 4  The split that frames all of Phase D. Fine-tune
          the same 8B model on a large SFT corpus (100M
          tokens) and its compute is a rounding error next
          to pretraining. Days 52-56 live in that sliver:
          pretraining is where the capability - and the
          bill - is; everything after is steering.

Published model figures are reported / approximate, as
labeled - GPT-2's token count is the commonly cited ~10B
for WebText, Llama 3's 15T is Meta's reported "over 15
trillion tokens". The arithmetic on them is exact.

Runs anywhere: pure stdlib Python, no torch needed, no
downloads, no APIs. Deterministic - every number below
is reproducible to the digit.

Expected output:

  Part 1 - your Day 42 run, priced with C ~ 6ND
    N = 814,643 params, corpus = 2,357 chars
    D = 1,200 steps x 32 batch x 64 chars = 2,457,600 tokens processed
    corpus recycled: 1042.7 epochs (why it memorized, Day 42)
    C = 6 x 814,643 x 2,457,600 = 1.20e+13 FLOPs
  Part 2 - published configs, same formula (reported / approximate)
    GPT-2 XL:        N = 1.50e+09, D = 1.00e+10  ->  C = 9.00e+19 FLOPs  (7.49e+06 x Day 42)
      tokens per param: 6.7
    Llama-3-class:   N = 8.00e+09, D = 1.50e+13  ->  C = 7.20e+23 FLOPs  (5.99e+10 x Day 42)
      tokens per param: 1,875.0
  Part 3 - GPU time at 3.0e14 FLOP/s per H100 (stated assumption)
    Day 42 run:      0.0400 GPU-seconds (your MacBook took minutes - a 815K model cannot fill even one GPU; the rate assumes a big model can)
    GPT-2 XL:        3.47 GPU-days
    Llama-3-class:   27,778 GPU-days = 76.1 GPU-years on ONE H100
      ...or about 13.6 days on 2,048 H100s (illustrative arithmetic, not Meta's actual schedule)
  Part 4 - pretrain vs fine-tune: where Phase D's compute lives
    pretrain 8B on 15T tokens:      C = 7.20e+23 FLOPs
    fine-tune 8B on 100M tokens:    C = 4.80e+18 FLOPs
    fine-tune is 1/150,000 of pretraining - pretraining share of total: 99.9993%
    verdict: pretraining buys the capability; Days 52-56 just steer it
"""


def flops(n_params, n_tokens):
    """The industry rule of thumb: C ~ 6 * N * D."""
    return 6 * n_params * n_tokens


# ---------- Part 1: your Day 42 run, priced ----------
print("Part 1 - your Day 42 run, priced with C ~ 6ND")
N42 = 814_643                      # Day 42's exact parameter count
CORPUS = 2_357                     # chars of Shakespeare (Day 42)
D42 = 1_200 * 32 * 64              # steps x batch x context
C42 = flops(N42, D42)
print(f"  N = {N42:,} params, corpus = {CORPUS:,} chars")
print(f"  D = 1,200 steps x 32 batch x 64 chars = {D42:,} tokens "
      "processed")
print(f"  corpus recycled: {D42 / CORPUS:.1f} epochs "
      "(why it memorized, Day 42)")
print(f"  C = 6 x {N42:,} x {D42:,} = {C42:.2e} FLOPs")

# ---------- Part 2: published configs, same formula ----------
print("Part 2 - published configs, same formula "
      "(reported / approximate)")
CONFIGS = (
    ("GPT-2 XL:", 1.5e9, 1.0e10),        # WebText, ~10B tokens
    ("Llama-3-class:", 8.0e9, 1.5e13),   # reported: 15T+ tokens
)
for name, n, d in CONFIGS:
    c = flops(n, d)
    print(f"  {name:<16} N = {n:.2e}, D = {d:.2e}  ->  "
          f"C = {c:.2e} FLOPs  ({c / C42:.2e} x Day 42)")
    print(f"    tokens per param: {d / n:,.1f}")

# ---------- Part 3: GPU time at a stated assumption ----------
print("Part 3 - GPU time at 3.0e14 FLOP/s per H100 "
      "(stated assumption)")
RATE = 3.0e14                      # ~30% of H100 peak, sustained
DAY = 86_400                       # seconds
print(f"  Day 42 run:      {C42 / RATE:.4f} GPU-seconds "
      "(your MacBook took minutes - a 815K model cannot fill "
      "even one GPU; the rate assumes a big model can)")
for name, n, d in CONFIGS:
    gpu_days = flops(n, d) / RATE / DAY
    extra = (f" = {gpu_days / 365.25:.1f} GPU-years on ONE H100"
             if gpu_days > 1_000 else "")
    days = (f"{gpu_days:,.0f}" if gpu_days >= 100
            else f"{gpu_days:.2f}")
    print(f"  {name:<16} {days} GPU-days{extra}")
llama_days = flops(8.0e9, 1.5e13) / RATE / DAY
print(f"    ...or about {llama_days / 2048:.1f} days on 2,048 H100s "
      "(illustrative arithmetic, not Meta's actual schedule)")

# ---------- Part 4: pretrain vs fine-tune ----------
print("Part 4 - pretrain vs fine-tune: where Phase D's compute "
      "lives")
C_PRE = flops(8.0e9, 1.5e13)
C_FT = flops(8.0e9, 1.0e8)         # a large SFT corpus: 100M tokens
print(f"  pretrain 8B on 15T tokens:      C = {C_PRE:.2e} FLOPs")
print(f"  fine-tune 8B on 100M tokens:    C = {C_FT:.2e} FLOPs")
print(f"  fine-tune is 1/{C_PRE / C_FT:,.0f} of pretraining - "
      f"pretraining share of total: "
      f"{100 * C_PRE / (C_PRE + C_FT):.4f}%")
print("  verdict: pretraining buys the capability; "
      "Days 52-56 just steer it")
