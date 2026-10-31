"""Day 76 - FSDP & DeepSpeed: Shard the State, Gather Just in Time.

Days 73-75 priced the problem three ways. Day 73: training
state costs ~16 bytes per parameter, and plain data parallel
(Day 74) makes EVERY GPU pay all 16 - replicate the model,
replicate the bill. Day 75 split the model itself. Tonight
is the third answer, and it is the one the frameworks -
PyTorch FSDP and Microsoft's DeepSpeed ZeRO - automate:

    Shard the state across the data-parallel workers, and
    all-gather each layer's parameters just in time to
    compute with them, then drop the gathered copy.

THE RECIPE (ZeRO slicing - same 16 total as Day 73, cut
differently; frameworks shuffle which copy wears which
dtype, the 16 and the sharding structure are what matter):
  bf16 params (compute copy) ... 2 bytes  replicated in DP,
                                 ZeRO-1, ZeRO-2; sharded in 3
  bf16 gradients ............... 2 bytes  replicated in DP and
                                 ZeRO-1; sharded from ZeRO-2 on
  optimizer states ............ 12 bytes  fp32 master weights 4
                                 + Adam m 4 + v 4; replicated
                                 only in plain DP, sharded from
                                 ZeRO-1 on
  TOTAL ........................ 16 bytes per parameter

THE LADDER (DeepSpeed ZeRO stages; FSDP's default full
sharding is ZeRO-3's twin):
  ZeRO-1  shard optimizer states:  per GPU 4 + 12/N B/param
  ZeRO-2  + shard gradients:       per GPU 2 + 14/N B/param
  ZeRO-3  + shard parameters:      per GPU 16/N   B/param

  Part 1  Per-GPU static memory for 7B and 70B under plain
          DP vs ZeRO-1/2/3 at N=8 and N=64 - the ladder as
          a table. Activations are NOT in these numbers:
          they are Day 73's separate, per-GPU bill.
  Part 2  The stand-still test, revisited: ZeRO-3 per-GPU
          memory is Day 73's total divided by N - the
          perfect split Day 73 could only hypothesize.
  Part 3  The honest cost, on the wire. Stages 1-2 sync the
          same gradient bytes as DP (the ZeRO paper's own
          accounting). Stage 3 adds two all-gathers of the
          full params per step - forward and backward - for
          1.5x DP's traffic, priced at Day 74's 50 GB/s.

Runs anywhere: pure standard library, no torch, no GPU,
no downloads, no APIs. The whole lesson is counting bytes
and bytes-on-wires, so it runs on the MacBook ~/ai-lab
venv in milliseconds.

Expected output:

  Part 1 - per-GPU static memory: one 16 B/param bill, four holders
    recipe (ZeRO slicing): bf16 params 2 + bf16 grads 2 + states 12
    states = fp32 master 4 + Adam m 4 + v 4; total 16 B/param -
    Day 73's same 16, re-sliced (frameworks shuffle the dtypes)
    7B model - GB resident per GPU (no activations):
      stage          B/param     @ N=8  B/param    @ N=64
      plain DP         16.00     112.0    16.00     112.0
      ZeRO-1            5.50      38.5     4.19      29.3
      ZeRO-2            3.75      26.2     2.22      15.5
      ZeRO-3 / FSDP     2.00      14.0     0.25       1.8
    70B model - GB resident per GPU (no activations):
      stage          B/param     @ N=8  B/param    @ N=64
      plain DP         16.00    1120.0    16.00    1120.0
      ZeRO-1            5.50     385.0     4.19     293.1
      ZeRO-2            3.75     262.5     2.22     155.3
      ZeRO-3 / FSDP     2.00     140.0     0.25      17.5
  Part 2 - the stand-still test (Day 73), answered
    Day 73: 70B static total = 1120 GB = 14 cards' worth IF split perfectly
    plain DP never splits it: every GPU owes all 1120 GB, at any fleet size
    ZeRO-3 @ N=64: 17.5 GB per GPU - Day 73's total / 64 exactly,
      the perfect split achieved; headroom on an 80 GB card: 62.5 GB
      for activations + the one layer gathered at a time
    7B ZeRO-3 @ N=8: 14.0 GB per GPU - Day 73 needed 2 cards to hold it
  Part 3 - sync traffic per step (N=64): the honest cost
    ring factor (N-1)/N = 63/64; effective link 50 GB/s (Day 74)
    7B: full params 14 GB, full grads 14 GB (bf16)
      plain DP / ZeRO-1 / ZeRO-2: all-reduce grads = 27.6 GB -> 0.55 s
      ZeRO-3 / FSDP: gathers 13.8 + 13.8 + reduce-scatter 13.8 = 41.3 GB -> 0.83 s
    70B: full params 140 GB, full grads 140 GB (bf16)
      plain DP / ZeRO-1 / ZeRO-2: all-reduce grads = 275.6 GB -> 5.51 s
      ZeRO-3 / FSDP: gathers 137.8 + 137.8 + reduce-scatter 137.8 = 413.4 GB -> 8.27 s
    ratio: exactly 1.50x DP's traffic, for up to 64x less resident memory
    verdict: memory traded for bandwidth. On NVLink /
    InfiniBand that trade wins, with prefetch hiding the
    gathers; on slow links the same sharding loses.
"""
import math

GB = 1e9                      # gigabytes as sold: 10^9 bytes
GPU_GB = 80                   # one H100/A100-class card
LINK_GBs = 50                 # Day 74's effective interconnect

# The 16 bytes, ZeRO slicing (see docstring).
PARAMS = 2                    # bf16 compute copy
GRADS = 2                     # bf16 gradients
STATES = 12                   # fp32 master 4 + Adam m 4 + v 4
TOTAL = PARAMS + GRADS + STATES          # 16 B/param

STAGES = ("plain DP", "ZeRO-1", "ZeRO-2", "ZeRO-3 / FSDP")


def bpp(stage, n):
    """Bytes per parameter RESIDENT on one GPU, fleet size n."""
    if stage == "plain DP":
        return float(TOTAL)
    if stage == "ZeRO-1":      # shard optimizer states only
        return PARAMS + GRADS + STATES / n
    if stage == "ZeRO-2":      # + shard gradients
        return PARAMS + (GRADS + STATES) / n
    return TOTAL / n           # ZeRO-3 / FSDP: shard everything


# ---------- Part 1: the ladder as a table ----------
print("Part 1 - per-GPU static memory: one 16 B/param bill, "
      "four holders")
print("  recipe (ZeRO slicing): bf16 params 2 + bf16 grads 2 "
      "+ states 12")
print("  states = fp32 master 4 + Adam m 4 + v 4; total 16 "
      "B/param -")
print("  Day 73's same 16, re-sliced (frameworks shuffle the "
      "dtypes)")
for label, pb in (("7B", 7), ("70B", 70)):
    print(f"  {label} model - GB resident per GPU "
          "(no activations):")
    print(f"    {'stage':<14} {'B/param':>7} {'@ N=8':>9} "
          f"{'B/param':>8} {'@ N=64':>9}")
    for stage in STAGES:
        b8, b64 = bpp(stage, 8), bpp(stage, 64)
        print(f"    {stage:<14} {b8:>7.2f} {b8 * pb:>9.1f} "
              f"{b64:>8.2f} {b64 * pb:>9.1f}")

# ---------- Part 2: the stand-still test, answered ----------
print("Part 2 - the stand-still test (Day 73), answered")
total70 = 70 * TOTAL
print(f"  Day 73: 70B static total = {total70:.0f} GB = "
      f"{math.ceil(total70 / GPU_GB)} cards' worth IF split "
      "perfectly")
print(f"  plain DP never splits it: every GPU owes all "
      f"{total70:.0f} GB, at any fleet size")
z3_70 = bpp("ZeRO-3 / FSDP", 64) * 70
print(f"  ZeRO-3 @ N=64: {z3_70:.1f} GB per GPU - Day 73's "
      "total / 64 exactly,")
print(f"    the perfect split achieved; headroom on an 80 GB "
      f"card: {GPU_GB - z3_70:.1f} GB")
print("    for activations + the one layer gathered at a time")
z3_7 = bpp("ZeRO-3 / FSDP", 8) * 7
print(f"  7B ZeRO-3 @ N=8: {z3_7:.1f} GB per GPU - Day 73 "
      "needed 2 cards to hold it")

# ---------- Part 3: the bill moves to the wire ----------
print("Part 3 - sync traffic per step (N=64): the honest cost")
RING = (64 - 1) / 64           # Day 74's ring factor (N-1)/N
print(f"  ring factor (N-1)/N = 63/64; effective link "
      f"{LINK_GBs} GB/s (Day 74)")
for label, pb in (("7B", 7), ("70B", 70)):
    p_gb = pb * PARAMS         # full params, bf16
    g_gb = pb * GRADS          # full grads, bf16
    dp = 2 * RING * g_gb                   # one grad all-reduce
    gather = RING * p_gb                   # one param all-gather
    rs = RING * g_gb                       # grad reduce-scatter
    z3 = 2 * gather + rs          # fwd + bwd gathers + rs
    print(f"  {label}: full params {p_gb:.0f} GB, full grads "
          f"{g_gb:.0f} GB (bf16)")
    print(f"    plain DP / ZeRO-1 / ZeRO-2: all-reduce grads "
          f"= {dp:.1f} GB -> {dp / LINK_GBs:.2f} s")
    print(f"    ZeRO-3 / FSDP: gathers {gather:.1f} + "
          f"{gather:.1f} + reduce-scatter {rs:.1f} = "
          f"{z3:.1f} GB -> {z3 / LINK_GBs:.2f} s")
print("  ratio: exactly 1.50x DP's traffic, for up to 64x "
      "less resident memory")
print("  verdict: memory traded for bandwidth. On NVLink /")
print("  InfiniBand that trade wins, with prefetch hiding the")
print("  gathers; on slow links the same sharding loses.")
