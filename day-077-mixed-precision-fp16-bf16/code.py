"""Day 77 - Mixed Precision: FP16/BF16, and Why It (Mostly) Just Works.

Day 73 priced the mixed-precision recipe at 16 bytes per
parameter without ever saying what the formats ARE. Tonight
is the missing bit-level picture, and it is one idea:

    A float is a BIT BUDGET split between RANGE (exponent
    bits: how big/small a number can be) and PRECISION
    (mantissa bits: how finely it can be told apart from
    its neighbors). fp32 = 1/8/23, fp16 = 1/5/10,
    bf16 = 1/8/7 (sign / exponent / mantissa).

bf16 keeps fp32's 8 exponent bits - the same range, roughly
1e-38 to 3e38 - and pays with precision: only 7 mantissa
bits. fp16 spends 5 bits on exponent and 10 on mantissa:
finer steps, but the range collapses to about 6e-5..65504
for normal numbers. That single trade explains everything
that follows:

  Part 1  Representability probes. Cast everyday values
          into fp16 and bf16 and print what survives. Both
          formats mangle pi and a $1000.10 price; only fp16
          dies at 70000 (overflow to inf). bf16 is coarser
          everywhere - precision is what it sold.
  Part 2  Underflow, and the loss-scaling rescue. Real
          gradients are tiny: 1e-8 in fp16 rounds to EXACTLY
          0.0 - the update silently never happens. bf16's
          range lets 1e-8 survive unscaled. For fp16,
          multiply the loss by S=1024 before backward and
          every gradient comes out 1024x bigger; divide by
          S before the step. The gradient lands back, alive.
  Part 3  Swamping - why the master copy stays fp32.
          w = 1.0, update = 1e-4, 100 steps. In fp16 the
          update is 10x smaller than the gap between
          representable neighbors (2^-10), so w + u rounds
          back to w EVERY time: 100 updates, zero movement.
          The fp32 master copy absorbs all 100. bf16 is
          even coarser near 1.0 (gap 2^-7), so the master
          copy is not optional in either format - bf16
          fixes range, NOT precision.

Runs on the ~/ai-lab venv: numpy only (free), no CUDA, no
paid APIs, no downloads. numpy has float16 built in but NO
bfloat16, so bf16 is emulated here by chopping an fp32's
mantissa to its top 7 bits - real hardware rounds instead
of chopping, so a value can differ by one last-bit step,
but the grid of representable values is identical. On the
Mac, torch has the real thing: torch.tensor(x,
dtype=torch.bfloat16). Every input is a fixed literal -
no RNG anywhere, every digit exact.

Expected output:

  Part 1 - the bit budget: fp32 1/8/23, fp16 1/5/10, bf16 1/8/7
    (sign / exponent = RANGE / mantissa = PRECISION)
             0.1  fp16 ->     0.099976   bf16 ->     0.099609
             1.1  fp16 ->     1.099609   bf16 ->     1.093750
      3.14159265  fp16 ->     3.140625   bf16 ->     3.140625
          1000.1  fp16 ->  1000.000000   bf16 ->  1000.000000
         70000.0  fp16 ->          inf   bf16 -> 69632.000000
    both mangle 0.1, pi, and the $1000.10 price; only fp16
    overflows at 70000 (its max is 65504) - bf16 kept fp32's range
  Part 2 - tiny gradients: fp16 underflow vs loss scaling (S=1024)
    true grad   fp16 direct   fp16 via scale   bf16 direct
        1e-03    1.0004e-03       1.0004e-03    9.9945e-04
        1e-05    1.0014e-05       9.9987e-06    9.9540e-06
        1e-07    1.1921e-07       1.0000e-07    9.9652e-08
        1e-08    0.0000e+00       1.0012e-08    9.9535e-09
    fp16's smallest step is 2^-24 = 5.96e-08, so 1e-8 rounds to 0.0
    scaled by 1024 it casts safely, then /1024 brings it back alive
    bf16 needs no scaling here: its range reaches down to ~1e-38
  Part 3 - swamping: w = 1.0, update = 1e-4, applied 100 times
    after   1 updates:  fp16 w = 1.0000000   fp32 master = 1.0001000
    after  10 updates:  fp16 w = 1.0000000   fp32 master = 1.0010002
    after 100 updates:  fp16 w = 1.0000000   fp32 master = 1.0100017
    total movement: fp16 0.0000000   fp32 0.0100017   (true: 0.0100000)
    fp32's extra 0.0000017 is its own rounding dust - every update
    LANDED; in fp16 the 100 updates never existed at all
    near 1.0 the representable gap is 2^-10 (fp16) / 2^-7 (bf16) /
    2^-23 (fp32): a 1e-4 update falls inside fp16's gap and rounds
    away every single step - the fp32 master is where updates live
    verdict: low precision computes, fp32 remembers - that is the
    whole recipe, and Day 73's 16 bytes/param is its price tag
"""
import warnings

import numpy as np

warnings.filterwarnings("ignore")   # fp16 overflow prints inf on purpose


def to_bf16(x):
    """Emulate bfloat16: keep fp32's sign + 8 exponent bits,
    truncate the mantissa to its top 7 bits. (numpy has no
    bf16 dtype; torch does - see the docstring.)"""
    bits = np.float32(x).view(np.uint32)
    return (bits & np.uint32(0xFFFF0000)).view(np.float32)


# ---------- Part 1: the bit budget - what survives the cast ----------
print("Part 1 - the bit budget: fp32 1/8/23, fp16 1/5/10, bf16 1/8/7")
print("  (sign / exponent = RANGE / mantissa = PRECISION)")
probes = [0.1, 1.1, 3.14159265, 1000.1, 70000.0]
for v in probes:
    f16 = float(np.float16(v))
    bf = float(to_bf16(v))
    print(f"  {v:>12}  fp16 -> {f16:>12.6f}   bf16 -> {bf:>12.6f}")
print("  both mangle 0.1, pi, and the $1000.10 price; only fp16")
print("  overflows at 70000 (its max is 65504) - bf16 kept fp32's range")

# ---------- Part 2: underflow, and the loss-scaling rescue ----------
print("Part 2 - tiny gradients: fp16 underflow vs loss scaling (S=1024)")
S = 1024.0
grads = [1e-3, 1e-5, 1e-7, 1e-8]
print("  true grad   fp16 direct   fp16 via scale   bf16 direct")
for g in grads:
    direct = float(np.float16(g))
    rescued = float(np.float16(g * S)) / S   # scale up, cast, scale back
    bf = float(to_bf16(g))
    print(f"  {g:>9.0e}   {direct:>11.4e}   {rescued:>14.4e}   {bf:>11.4e}")
print("  fp16's smallest step is 2^-24 = 5.96e-08, so 1e-8 rounds to 0.0")
print("  scaled by 1024 it casts safely, then /1024 brings it back alive")
print("  bf16 needs no scaling here: its range reaches down to ~1e-38")

# ---------- Part 3: swamping - why the master copy stays fp32 ----------
print("Part 3 - swamping: w = 1.0, update = 1e-4, applied 100 times")
w16, w32 = np.float16(1.0), np.float32(1.0)
u16, u32 = np.float16(1e-4), np.float32(1e-4)
marks = {}
for step in range(1, 101):
    w16 = np.float16(w16 + u16)      # pure fp16 weight: update + store
    w32 = np.float32(w32 + u32)      # fp32 master copy
    if step in (1, 10, 100):
        marks[step] = (float(w16), float(w32))
for step, (a, b) in marks.items():
    print(f"  after {step:>3} updates:  fp16 w = {a:.7f}   fp32 master = {b:.7f}")
moved16 = float(w16) - 1.0
moved32 = float(w32) - 1.0
print(f"  total movement: fp16 {moved16:.7f}   fp32 {moved32:.7f}"
      "   (true: 0.0100000)")
print("  fp32's extra 0.0000017 is its own rounding dust - every update")
print("  LANDED; in fp16 the 100 updates never existed at all")
print("  near 1.0 the representable gap is 2^-10 (fp16) / 2^-7 (bf16) /")
print("  2^-23 (fp32): a 1e-4 update falls inside fp16's gap and rounds")
print("  away every single step - the fp32 master is where updates live")
print("  verdict: low precision computes, fp32 remembers - that is the")
print("  whole recipe, and Day 73's 16 bytes/param is its price tag")
