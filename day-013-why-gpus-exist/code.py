"""Day 13 - Linear Algebra Bite 2: Why GPUs Exist.

Tonight's idea: matmul is embarrassingly parallel -- every output cell is an
independent dot product, 2*m*n*k multiply-adds with zero coordination -- so a
GPU's thousands of tiny lanes beat a CPU's handful of big cores. You will:
  1. Run the same 128x128 matmul three ways (Python loops, torch CPU, torch MPS).
  2. Benchmark CPU vs MPS across sizes: the speedup GROWS with size.
  3. See why batched matmul (B, n, n) is the GPU's favorite shape.

Runs on MacBook: ~/ai-lab venv, torch CPU/MPS. No downloads, no paid APIs.
"""
import time
import torch

torch.manual_seed(13)
mps = torch.backends.mps.is_available()
print("torch:", torch.__version__, "| mps available:", mps)

def gflops(m, n, k, seconds):
    """One matmul does 2*m*n*k FLOPs (a multiply AND an add per pair)."""
    return (2 * m * n * k) / max(seconds, 1e-9) / 1e9

# ---- 1. Same 128x128 matmul, three engines ----
n = 128
A = [[float(i - j) / n for j in range(n)] for i in range(n)]
B = [[float(i + j) / n for j in range(n)] for i in range(n)]

t0 = time.perf_counter()
C_loop = [[sum(A[i][k] * B[k][j] for k in range(n))
           for j in range(n)] for i in range(n)]
t_loop = time.perf_counter() - t0

At, Bt = torch.tensor(A), torch.tensor(B)
t0 = time.perf_counter()
Ct = At @ Bt
t_cpu = time.perf_counter() - t0

t_mps = None
if mps:
    Am, Bm = At.to("mps"), Bt.to("mps")
    _ = Am @ Bm                      # warmup: first launch compiles the kernel
    torch.mps.synchronize()
    t0 = time.perf_counter()
    Cm = Am @ Bm
    torch.mps.synchronize()          # wait for the GPU, or the clock lies
    t_mps = time.perf_counter() - t0
    del Am, Bm, Cm

# spot-check: the loop math and torch agree (they must -- same op)
ok = max(abs(C_loop[i][j] - Ct[i, j].item())
         for i in range(0, n, 16) for j in range(0, n, 16)) < 1e-3
print(f"\n128x128 matmul, three engines:")
print(f"  python loops : {t_loop:9.3f}s   one core, no vectorization")
print(f"  torch CPU    : {t_cpu:9.4f}s   vectorized + threaded")
if t_mps is not None:
    print(f"  torch MPS    : {t_mps:9.4f}s   GPU -- slower here! not enough work")
print("loop == torch (spot check):", ok)

# ---- 2. The gap widens with size ----
print("\nsquare matmul, CPU vs MPS (warmup + sync each row):")
print(f"{'size':>6} {'cpu s':>8} {'cpu GF':>8} | {'mps s':>8} {'mps GF':>8} | speedup")
for s in [256, 512, 1024, 2048, 4096]:
    x = torch.randn(s, s)
    reps = 3 if s <= 1024 else 1
    t0 = time.perf_counter()
    for _ in range(reps):
        x @ x
    t_c = (time.perf_counter() - t0) / reps
    line = f"{s:>6} {t_c:8.3f} {gflops(s, s, s, t_c):8.1f} |"
    if mps:
        xm = x.to("mps")
        _ = xm @ xm; torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            xm @ xm
        torch.mps.synchronize()
        t_g = (time.perf_counter() - t0) / reps
        line += f" {t_g:8.3f} {gflops(s, s, s, t_g):8.1f} | {t_c / t_g:6.1f}x"
        del xm
    else:
        line += "   (no MPS on this machine)"
    print(line)
print("Read the speedup column bottom-up: bigger problem -> GPU closer to its roof.")

# ---- 3. Batch is the GPU's favorite shape ----
# (B, n, n): B independent matmuls, one kernel launch -- this is what
# attention and per-token MLPs look like inside a transformer.
B, nb = 64, 512
xb = torch.randn(B, nb, nb)
t0 = time.perf_counter()
_ = xb @ xb
t_bc = time.perf_counter() - t0
total_flops = (2 * B * nb ** 3) / 1e9  # bmm = B separate (nb,nb) matmuls
if mps:
    xbm = xb.to("mps")
    _ = xbm @ xbm; torch.mps.synchronize()
    t0 = time.perf_counter()
    _ = xbm @ xbm
    torch.mps.synchronize()
    t_bm = time.perf_counter() - t0
    print(f"\nbatch matmul ({B},{nb},{nb}) = {B} independent matmuls:")
    print(f"  CPU: {t_bc:.3f}s ({total_flops / t_bc:,.0f} GFLOPS)")
    print(f"  MPS: {t_bm:.3f}s ({total_flops / t_bm:,.0f} GFLOPS)  "
          f"{t_bc / t_bm:.1f}x -- one launch feeds thousands of lanes")
