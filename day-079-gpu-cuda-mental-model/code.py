"""Day 79 - The GPU: Just Enough CUDA Mental Model.

Phase F. No CUDA code today - you have no GPU on the MacBook,
and you do not need one to get the mental model. The model is
the hardware's SHAPE:

    A GPU is hundreds of small, simple cores grouped into
    Streaming Multiprocessors (SMs). You launch a KERNEL: a
    grid of thread BLOCKS, each block pinned to one SM, the
    threads in a block sharing a small, fast on-chip memory.
    Everything else is a memory hierarchy with four levels -
    registers (per thread) -> shared memory (per block) ->
    HBM (the card's big memory) -> host RAM (over the bus) -
    and a kernel's speed is decided by WHICH level its data
    lives in. The roofline rule prices that in one division:
    an op is compute-bound only if its arithmetic intensity
    (FLOPs per byte moved) beats peak_FLOPS / peak_bandwidth.

  Part 1  The roofline, on spec-sheet numbers (stdlib only).
          H100-class card, stated as assumptions to re-check:
          ~989 TFLOPS bf16, 3.35 TB/s HBM. The ridge point -
          intensity needed to be compute-bound - comes out
          ~295 FLOP/byte. Three ops get classified by their
          intensity: a training matmul (way above the ridge),
          one batch-1 decode step for a 7B in fp16 (Day 65:
          intensity 1.0, memory-bound, uses ~0.3% of the
          card's FLOPs), and an elementwise add (intensity
          ~0.08, memory-bound). Same card, three verdicts -
          the op decides, not the spec sheet.
  Part 2  Feel a hierarchy on hardware you DO have (numpy).
          Read 200,000 values from float64 arrays of growing
          size in a fixed pseudo-random order (deterministic
          hash, no RNG): an 8 KB array lives in L1-class cache,
          a 240 MB array lives in RAM, and the ns-per-read
          ladder between them is the hierarchy, felt. Timings
          are MACHINE-DEPENDENT - printed rounded to ONE
          significant figure; your Mac's digits will differ
          from any printed here, the ladder's shape will not.
          Honest label: a CPU's cache/RAM ladder is an ANALOGY
          for the GPU's registers/shared/HBM ladder - same
          shape, different constants, and on the GPU the first
          step down (off the SM, out to HBM) costs far more.

Runs on the ~/ai-lab venv: numpy only (free), no torch needed,
no CUDA, no paid APIs, no downloads. Part 1 is exact arithmetic
on the stated specs. Part 2's data and probe order are
deterministic; only the TIMINGS vary by machine, by design.

Expected output (Part 2 timings from one VM run - yours differ):

  Part 1 - the roofline: which wall does each op hit?
    assumed card (H100-class spec sheet): 989 TFLOPS bf16, 3.35 TB/s HBM
    ridge point = 989e12 / 3.35e12 = 295 FLOP/byte - above it: compute-bound
      op                               intensity   verdict       time floor
      training matmul 4096^3 bf16         1365.3   compute-bound   0.139 ms
      decode step, 7B fp16, batch 1          1.0   memory-bound    4.179 ms
      elementwise add, 1e8 fp32              0.1   memory-bound    0.358 ms
    decode detail: 14.0 GB of weights read per token -> at most 239 tok/s,
      and it uses 3.35 of 989 TFLOPS (0.3%) - the FLOPs sit idle, waiting on bytes
    verdict: batch-1 decode is a memory problem wearing a math costume (Day 65)
  Part 2 - the same ladder on your CPU (numpy, timings machine-dependent)
       elements    bytes  home        random read (1 sig fig)
           1000     8 KB  L1-sized    ~2 ns   (check sum 9.99e+07)
         100000   800 KB  L2/L3-sized ~2 ns   (check sum 1.00e+10)
        1000000     8 MB  L3-sized    ~4 ns   (check sum 1.00e+11)
       10000000    80 MB  RAM-sized   ~8 ns   (check sum 1.00e+12)
       30000000   240 MB  RAM-sized   ~10 ns   (check sum 3.00e+12)
    read it as: same op, same array values - only the array's HOME
    changed, and the RAM-sized reads run several times slower than
    the cache-sized ones. On a GPU the ladder is registers ->
    shared -> HBM and the steps are bigger still.
"""
import time

import numpy as np

# ---------- Part 1: the roofline (stdlib arithmetic) ----------
print("Part 1 - the roofline: which wall does each op hit?")

# Spec-sheet numbers for an H100-class card. ASSUMPTIONS - re-check
# against the current datasheet before quoting them anywhere; the
# method (one division) is the lesson, the constants will age.
PEAK_FLOPS = 989e12        # bf16 dense TFLOPS, in FLOP/s
PEAK_BW = 3.35e12          # HBM bandwidth, in bytes/s

ridge = PEAK_FLOPS / PEAK_BW
print(f"  assumed card (H100-class spec sheet): 989 TFLOPS bf16, "
      f"3.35 TB/s HBM")
print(f"  ridge point = 989e12 / 3.35e12 = {ridge:.0f} FLOP/byte "
      f"- above it: compute-bound")


def classify(name, flops, nbytes):
    """One op through the roofline: intensity decides the verdict,
    the slower of (compute time, memory time) is the floor."""
    intensity = flops / nbytes
    t_compute = flops / PEAK_FLOPS
    t_memory = nbytes / PEAK_BW
    bound = "compute" if t_compute >= t_memory else "memory"
    return name, intensity, bound, max(t_compute, t_memory) * 1e3


ops = [
    # Training matmul: C = A@B, 4096^3, bf16. FLOPs = 2*M*N*K.
    # Bytes = read A + read B + write C, 2 bytes per element.
    classify("training matmul 4096^3 bf16",
             2 * 4096 ** 3, 3 * 4096 * 4096 * 2),
    # Decode step, Day 65: one token, batch 1, 7B fp16. Every weight
    # is used once: ~2 FLOPs per parameter, every weight byte read.
    classify("decode step, 7B fp16, batch 1",
             2 * 7e9, 7e9 * 2),
    # Elementwise add: C = A + B over 1e8 fp32 values.
    # 1 FLOP per element; bytes = 2 reads + 1 write, 4 bytes each.
    classify("elementwise add, 1e8 fp32",
             1e8, 3 * 1e8 * 4),
]
print(f"    {'op':<32} {'intensity':>9}   {'verdict':<13} {'time floor':>10}")
for name, intensity, bound, t_ms in ops:
    print(f"    {name:<32} {intensity:>9.1f}   {bound + '-bound':<13} "
          f"{t_ms:>7.3f} ms")
tok_s = PEAK_BW / (7e9 * 2)
used = (2 * 7e9) / (7e9 * 2 / PEAK_BW) / 1e12
print(f"  decode detail: 14.0 GB of weights read per token -> at most "
      f"{tok_s:.0f} tok/s,")
print(f"    and it uses {used:.2f} of 989 TFLOPS "
      f"({100 * used / 989:.1f}%) - the FLOPs sit idle, waiting on bytes")
print("  verdict: batch-1 decode is a memory problem wearing a math "
      "costume (Day 65)")

# ---------- Part 2: feel a hierarchy on your own CPU ----------
print("Part 2 - the same ladder on your CPU (numpy, timings "
      "machine-dependent)")


def round1(x):
    """Round to 1 significant figure - honest precision for a timing."""
    from math import floor, log10
    return round(x, -int(floor(log10(x))))


PROBES = 200_000
# Deterministic probe order: Knuth's multiplicative hash - looks
# random to the cache, identical on every run and machine. No RNG.
_i = np.arange(PROBES, dtype=np.int64)
SIZES = [
    (1_000, "L1-sized"),
    (100_000, "L2/L3-sized"),
    (1_000_000, "L3-sized"),
    (10_000_000, "RAM-sized"),
    (30_000_000, "RAM-sized"),
]
print(f"    {'elements':>9} {'bytes':>8}  {'home':<11} "
      f"{'random read (1 sig fig)':>23}")
for n, home in SIZES:
    a = np.arange(n, dtype=np.float64)   # deterministic data, no RNG
    idx = ((_i * 2654435761) % n).astype(np.int64)
    a[idx].sum()                                     # warm the cache
    best = None
    for _ in range(3):
        t0 = time.perf_counter()
        s = a[idx].sum()                             # the timed probes
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    ns = best / PROBES * 1e9
    label = f"{a.nbytes / 1e3:.0f} KB" if a.nbytes < 1e6 \
        else f"{a.nbytes / 1e6:.0f} MB"
    print(f"    {n:>9} {label:>8}  {home:<11} "
          f"~{round1(ns):g} ns   (check sum {s:.2e})")
print("  read it as: same op, same array values - only the array's HOME")
print("  changed, and the RAM-sized reads run several times slower than")
print("  the cache-sized ones. On a GPU the ladder is registers ->")
print("  shared -> HBM and the steps are bigger still.")
