"""Day 46 - "Attention Is All You Need": A Guided Read.

Days 36-45 you built every mechanism in the 2017 paper by
Vaswani et al.: Q/K/V attention (Day 37), scaled dot-product
by hand (Day 38), multi-head (Day 39), the full block (Day 40),
sinusoidal position codes (Day 41), BPE (Day 43), and the
encoder / decoder / encoder-decoder wiring (Day 45). Tonight
is not a new mechanism. Tonight the paper itself is the
artifact: a reading map of what to read, what to skip, and
which of your builds each section is describing - plus the
honest list of what the paper does NOT contain.

Tonight's code does not download the paper and calls no API.
It reproduces two of the paper's own numbers from its stated
configuration, the way you would sanity-check any design doc:

  Part 1  The parameter count. Section 5.3 / Table 3 of the
          paper give the base config: d_model = 512, N = 6
          layers per stack, h = 8 heads, d_ff = 2048, and a
          shared source-target BPE vocabulary of "about
          37,000" tokens (Section 5.1). The embedding matrix
          is shared three ways (encoder in, decoder in,
          pre-softmax out - Section 3.4), the positional
          encodings are fixed sinusoids (zero parameters -
          Day 41), and attention heads split d_model, so
          h = 8 changes no count: each attention block is
          still 4 matrices of d_model x d_model. Add up
          weights, biases, and LayerNorm gains/biases and
          compare with the number the paper reports for the
          base model in Table 2: 65M.
  Part 2  The learning-rate schedule, Section 5.3 verbatim:
              lrate = d_model^-0.5 * min(step^-0.5,
                                         step * warmup^-1.5)
          with warmup_steps = 4000. Print it across the run:
          a linear ramp for 4,000 steps, a peak, then an
          inverse-square-root decay. Base training ran
          100,000 steps (Section 5.3), so the last printed
          step is the end of the paper's own run.

Pure standard-library Python (math only): no torch, no
numpy, no downloads, no paid APIs. Runs identically in the
~/ai-lab venv on the MacBook. Every number is exact
arithmetic - no RNG anywhere in this lesson.

Expected output:

Part 1 - recount the base Transformer from its config (paper, Table 3)
  config: d_model 512, N 6 layers/stack, h 8 heads, d_ff 2048, vocab ~37,000 (shared)
  shared embedding (tied x3, Sec 3.4):    18,944,000
  one encoder layer:                       3,152,384
    x6 encoder layers:                   18,914,304
  one decoder layer:                       4,204,032
    x6 decoder layers:                   25,224,192
  final LayerNorm:                             1,024
  YOUR count:                             63,083,520  = 63.08M
  the paper reports (Table 2): 65M  ->  your count is 97.1% of it
  the ~3% gap is bookkeeping, not architecture: 'about 37,000' tokens
  is doing real work in that sentence (the gap = 3,743 tokens x 512).
Part 2 - the paper's learning-rate schedule (Sec 5.3, warmup 4,000)
  step       1  lr 1.75e-07   warming up (linear ramp)
  step     100  lr 1.75e-05   warming up (linear ramp)
  step   1,000  lr 1.75e-04   warming up (linear ramp)
  step   2,000  lr 3.49e-04   warming up (linear ramp)
  step   4,000  lr 6.99e-04   PEAK - warmup ends, decay takes over
  step  10,000  lr 4.42e-04   decaying as 1/sqrt(step)
  step  40,000  lr 2.21e-04   decaying as 1/sqrt(step)
  step 100,000  lr 1.40e-04   decaying as 1/sqrt(step)
  read: x40 from step 100 to the peak, then the peak only halves
  every time the step count quadruples - a long, gentle tail.
"""
import math

# ---------- Part 1: recount the base Transformer ----------
print("Part 1 - recount the base Transformer from its config "
      "(paper, Table 3)")
d_model = 512          # width of every sub-layer output (Sec 3.1)
n_layers = 6           # N: layers in EACH stack (Sec 3.1)
n_heads = 8            # h: heads split d_model into 8 x 64 (Sec 3.2.2)
d_ff = 2048            # inner width of the feed-forward net (Sec 3.3)
vocab = 37_000         # shared BPE vocab, "about 37,000" (Sec 5.1)

print(f"  config: d_model {d_model}, N {n_layers} layers/stack, "
      f"h {n_heads} heads, d_ff {d_ff}, vocab ~{vocab:,} (shared)")

# One multi-head attention block: Q, K, V and output projections,
# each d_model x d_model with a bias. h = 8 heads does NOT change
# the count - the heads are slices of these same matrices (Day 39).
attn = 4 * d_model * d_model + 4 * d_model
# One position-wise feed-forward net: d_model -> d_ff -> d_model (Day 40).
ffn = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
# One LayerNorm: a gain and a bias per feature (Day 29).
norm = 2 * d_model

# Encoder layer: self-attention + FFN + 2 norms (Sec 3.1).
enc_layer = attn + ffn + 2 * norm
# Decoder layer: masked self-attention + cross-attention + FFN + 3 norms.
dec_layer = 2 * attn + ffn + 3 * norm
# Embeddings: ONE matrix, shared by encoder input, decoder input,
# and the pre-softmax projection (Sec 3.4). Positional encodings
# are fixed sinusoids - zero parameters (Day 41).
embedding = vocab * d_model

enc_total = n_layers * enc_layer
dec_total = n_layers * dec_layer
total = embedding + enc_total + dec_total + norm   # + final norm

print(f"  shared embedding (tied x3, Sec 3.4):  {embedding:>12,}")
print(f"  one encoder layer:                    {enc_layer:>12,}")
print(f"    x{n_layers} encoder layers:                 {enc_total:>12,}")
print(f"  one decoder layer:                    {dec_layer:>12,}")
print(f"    x{n_layers} decoder layers:                 {dec_total:>12,}")
print(f"  final LayerNorm:                      {norm:>12,}")
print(f"  YOUR count:                           {total:>12,}  "
      f"= {total / 1e6:.2f}M")
paper_params = 65_000_000                          # Table 2, base model
print(f"  the paper reports (Table 2): 65M  ->  your count is "
      f"{100 * total / paper_params:.1f}% of it")
gap_tokens = (paper_params - total) // d_model
print("  the ~3% gap is bookkeeping, not architecture: "
      "'about 37,000' tokens")
print(f"  is doing real work in that sentence (the gap = "
      f"{gap_tokens:,} tokens x {d_model}).")

# ---------- Part 2: the paper's learning-rate schedule ----------
print("Part 2 - the paper's learning-rate schedule "
      "(Sec 5.3, warmup 4,000)")
warmup = 4_000


def paper_lr(step):
    """Section 5.3, verbatim: d_model^-0.5 scaled, linear ramp
    during warmup, inverse-square-root decay after it. The min()
    is the whole trick - whichever term is smaller is in charge,
    and the two terms cross exactly at step = warmup."""
    return d_model ** -0.5 * min(step ** -0.5,
                                 step * warmup ** -1.5)


steps = [1, 100, 1_000, 2_000, 4_000, 10_000, 40_000, 100_000]
for s in steps:
    if s < warmup:
        phase = "warming up (linear ramp)"
    elif s == warmup:
        phase = "PEAK - warmup ends, decay takes over"
    else:
        phase = "decaying as 1/sqrt(step)"
    print(f"  step {s:>7,}  lr {paper_lr(s):.2e}   {phase}")
print("  read: x40 from step 100 to the peak, then the peak only "
      "halves")
print("  every time the step count quadruples - a long, gentle tail.")
