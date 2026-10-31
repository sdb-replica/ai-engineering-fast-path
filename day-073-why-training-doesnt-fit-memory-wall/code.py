"""Day 73 - The Memory Wall: Why Training Doesn't Fit on One GPU.

Phase F opens. Everything so far trained on ONE device -
Day 34's biggest net had 113 parameters. Tonight's question
is pure arithmetic, and it explains why the next three days
(parallelism, Days 74-76) exist at all:

    Training memory is NOT the weights. It is weights +
    gradients + optimizer state + activations - and for
    Adam in mixed precision the static bill alone is about
    16 bytes per parameter, before a single activation.

THE RECIPE (state your assumptions - every framework mixes
slightly different dtypes, this is the standard accounting):
  Adam, mixed precision, per parameter:
    fp32 master weights .... 4 bytes  (the copy that updates)
    fp32 gradients ......... 4 bytes  (one grad per param)
    Adam m (1st moment) .... 4 bytes  (Day 18: Adam keeps two)
    Adam v (2nd moment) .... 4 bytes
    TOTAL .................. 16 bytes per parameter
  Some stacks also keep a bf16 compute copy of the weights
  (+2 bytes -> 18). SGD pays 8, SGD+momentum pays 12 - Part 1.
  GB below means 10^9 bytes, the way GPU boxes are sold.

  Part 1  Bytes per parameter: SGD vs SGD+momentum vs Adam,
          component by component. Day 34's 17-parameter XOR
          net costs 272 bytes under Adam. A 70B model costs
          16 bytes x 70 billion. Same recipe, bigger number.
  Part 2  The static bill (weights + grads + optimizer, NO
          activations yet) for 7B / 13B / 70B / 405B models,
          with the inference-only bill (bf16 weights alone,
          2 bytes/param) next to it, and the only column that
          matters: how many 80GB GPUs you need JUST TO STAND
          STILL - before one token is processed.
  Part 3  Activations: the memory autograd saves for the
          backward pass (Day 4's tape, at scale). Stated
          config: 80 layers, d_model 8192, batch 8 sequences
          x 4096 tokens, bf16, and a stated simplification -
          ~16 saved hidden-sized tensors per layer. Gradient
          checkpointing (store only layer boundaries, redo
          one layer's forward in the backward pass) cuts it
          ~14x for ~1/3 extra step time. Checkpointing as a
          fault-tolerance strategy is Day 78 - different job.

Runs anywhere: pure standard library, no torch, no GPU,
no downloads, no APIs. The whole lesson is counting bytes,
so it runs on the MacBook ~/ai-lab venv in milliseconds.

Expected output:

  Part 1 - bytes per parameter: the optimizer is the landlord
    SGD (fp32):              weights 4 + grads 4                  =  8 B/param
    SGD + momentum:          weights 4 + grads 4 + momentum 4     = 12 B/param
    Adam, mixed precision:   master 4 + grads 4 + m 4 + v 4       = 16 B/param
    variant:                 stacks keeping a bf16 compute copy add 2        = 18 B/param
    Day 34's XOR net, 17 params, under Adam: 272 bytes total
  Part 2 - static bill: weights + grads + optimizer (Adam, 16 B/param)
    model  params   static (no acts)  bf16 weights only  80GB GPUs to stand still
       7B      7B           112.0 GB            14.0 GB                         2
      13B     13B           208.0 GB            26.0 GB                         3
      70B     70B          1120.0 GB           140.0 GB                        14
     405B    405B          6480.0 GB           810.0 GB                        81
    a 70B model's STATIC bill is 1.12 TB - and not one token has moved yet
  Part 3 - activations: the tape autograd saves for backward
    config: 80 layers, d_model 8192, batch 8 x seq 4096 = 32768 tokens, bf16
    stated assumption: ~16 saved hidden-sized tensors per layer
    saved per token per layer: 262144 bytes (0.25 MB)
    activations, no checkpointing:     687.19 GB
    gradient checkpointing: boundaries 42.95 GB + one layer recomputed 8.59 GB
    70B step total, no checkpointing: 1120.0 static + 687.19 acts = 1807.19 GB -> 23 GPUs
    70B step total, checkpointing: 1120.0 static + 51.54 acts = 1171.54 GB -> 15 GPUs
    verdict: the wall is arithmetic, not engineering - 16 B/param + activations
    vs 80 GB/card is WHY Days 74-76 (splitting the bill) exist
"""
import math

GB = 1e9                      # gigabytes as sold: 10^9 bytes
GPU_GB = 80                   # one H100/A100-class card

# ---------- Part 1: bytes per parameter ----------
print("Part 1 - bytes per parameter: the optimizer is the landlord")
recipes = [
    ("SGD (fp32):", "weights 4 + grads 4", 4 + 4),
    ("SGD + momentum:", "weights 4 + grads 4 + momentum 4", 4 + 4 + 4),
    ("Adam, mixed precision:", "master 4 + grads 4 + m 4 + v 4",
     4 + 4 + 4 + 4),
]
for name, parts, total in recipes:
    print(f"  {name:<24} {parts:<36} = {total:>2} B/param")
print(f"  {'variant:':<24} stacks keeping a bf16 compute copy "
      f"add 2        = 18 B/param")
print(f"  Day 34's XOR net, 17 params, under Adam: {17 * 16} bytes total")
ADAM_BPP = recipes[-1][2]     # 16 - the recipe Parts 2-3 use

# ---------- Part 2: the static bill ----------
print("Part 2 - static bill: weights + grads + optimizer "
      "(Adam, 16 B/param)")
print(f"  {'model':>5} {'params':>7} {'static (no acts)':>18} "
      f"{'bf16 weights only':>18} {'80GB GPUs to stand still':>25}")
for label, params_b in (("7B", 7), ("13B", 13), ("70B", 70),
                        ("405B", 405)):
    params = params_b * GB
    static_gb = params * ADAM_BPP / GB
    weights_gb = params * 2 / GB          # inference: bf16 weights
    gpus = math.ceil(static_gb / GPU_GB)
    print(f"  {label:>5} {params_b:>6}B {static_gb:>15.1f} GB "
          f"{weights_gb:>15.1f} GB {gpus:>25}")
print("  a 70B model's STATIC bill is 1.12 TB - and not one token "
      "has moved yet")

# ---------- Part 3: activations ----------
print("Part 3 - activations: the tape autograd saves for backward")
LAYERS, D_MODEL = 80, 8192                # a 70B-class transformer
BATCH, SEQ = 8, 4096                      # per-device micro-batch
TOKENS = BATCH * SEQ
SAVED_PER_LAYER = 16      # stated simplification: ~16 saved tensors
BYTES = 2                                 # bf16 activations
per_token_layer = SAVED_PER_LAYER * D_MODEL * BYTES
acts_gb = per_token_layer * LAYERS * TOKENS / GB
bounds_gb = TOKENS * D_MODEL * BYTES * LAYERS / GB   # boundaries only
recompute_gb = per_token_layer * TOKENS / GB         # one layer redone
static70 = 70 * ADAM_BPP                          # 1120.0 GB, Part 2
print(f"  config: {LAYERS} layers, d_model {D_MODEL}, batch {BATCH} "
      f"x seq {SEQ} = {TOKENS} tokens, bf16")
print(f"  stated assumption: ~{SAVED_PER_LAYER} saved hidden-sized "
      "tensors per layer")
print(f"  saved per token per layer: {per_token_layer} bytes (0.25 MB)")
print(f"  activations, no checkpointing:     {acts_gb:.2f} GB")
print(f"  gradient checkpointing: boundaries {bounds_gb:.2f} GB + one "
      f"layer recomputed {recompute_gb:.2f} GB")
ckpt_acts = bounds_gb + recompute_gb
for label, total in (("no checkpointing", static70 + acts_gb),
                     ("checkpointing", static70 + ckpt_acts)):
    acts = total - static70
    print(f"  70B step total, {label}: {static70:.1f} static + "
          f"{acts:.2f} acts = {total:.2f} GB -> "
          f"{math.ceil(total / GPU_GB)} GPUs")
print("  verdict: the wall is arithmetic, not engineering - "
      "16 B/param + activations")
print("  vs 80 GB/card is WHY Days 74-76 (splitting the bill) exist")
