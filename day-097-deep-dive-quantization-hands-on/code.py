"""Day 97 - Deep Dive: Quantization Hands-On.

Day 65 quantized a formula-built matrix and printed an
error bill for the WEIGHTS. Two things were missing: the
weights were fake, and nobody checked what the error did
to the model's ANSWERS. Today both gaps close.

  Part 1  Train a real (small) classifier in numpy: a
          20 -> 32 -> 4 MLP on synthetic 4-class data,
          hand-rolled backprop, full-batch gradient
          descent. The data has a trap built in (below),
          and the trained first layer comes out with a
          3.0x spread in row ranges - rows are not
          interchangeable, because each hidden unit
          learned a different job.
  Part 2  Day 65's weight-error report, rerun on the
          trained W1 - but now per-TENSOR (one scale for
          all 640 weights) vs per-CHANNEL (one scale per
          row, 32 scales), at int8 and int4.
  Part 3  The report Day 65 never printed: BEHAVIOR.
          Swap BOTH layers for their dequantized versions
          (the way a real deployment ships) and re-run the
          test set: accuracy, mean |logit drift| against
          the fp32 model, and how many predictions flip.
          Weight error is mark-to-model; this is
          mark-to-market.
  Part 4  Bytes for the whole model, scales included,
          then the same math on one 4096x4096 layer -
          where the 32 extra scale numbers stop mattering.

The data's trap: features 0-9 are LOUD (centers at scale
2.0, noise 1.0) and features 10-19 are QUIET (scale 0.20,
noise 0.1). Classes 0 and 1 share the SAME loud center,
as do 2 and 3 - so telling 0 from 1 is impossible without
listening to the quiet features, and a hidden unit that
listens quietly has to turn its weights up. That spread
of row ranges is exactly what per-channel scales exploit
and per-tensor scales waste.

Quantization is Day 65's symmetric scheme (one scale,
w ~= scale * q, codes round(w / scale) clipped to
[-qmax, qmax]; numpy rounds half-to-even, same as torch).

Runs in the ~/ai-lab venv: numpy only, no torch needed,
no CUDA, no paid APIs, no downloads. Deterministic:
one seeded generator, full-batch training, no shuffling.

Expected output:

  Part 1 - a real trained model (numpy, hand-rolled backprop)
    architecture 20 -> 32 (relu) -> 4; 200 epochs, full-batch GD
    fp32 test accuracy: 0.9755 on 2000 samples
    W1 is 32x20 = 640 weights (largest layer); W2 is 4x32 = 128; total 768
    W1 row ranges: smallest row max|w| 0.389, largest 1.163 - a 3.0x spread
    (each row is one hidden unit's incoming weights - different
     units, different jobs, different ranges. Remember this spread.)
  Part 2 - weight error on W1: per-tensor vs per-channel
    int8 per-tensor:   max err 0.004573  rms err 0.002659
    int8 per-channel:  max err 0.004540  rms err 0.001633
    int4 per-tensor:   max err 0.082995  rms err 0.049261
    int4 per-channel:  max err 0.082600  rms err 0.029594
    int4 scales, first 8 rows of W1 (per-channel): 0.0840, 0.1324, 0.1292, 0.0764, 0.0742, 0.0883, 0.1208, 0.1393
    the single per-tensor int4 scale for all 640 weights: 0.1662
    read: max error barely moves (the biggest row sets it under
    both schemes) - the win is in RMS, the typical weight's error.
  Part 3 - behavior: both layers dequantized, same 2000 test samples
    fp32 (reference)   acc 0.9755  mean |dlogit| 0.0000  flips 0
    int8 per-tensor:   acc 0.9755  mean |dlogit| 0.0293  flips 2
    int8 per-channel:  acc 0.9750  mean |dlogit| 0.0170  flips 1
    int4 per-tensor:   acc 0.9080  mean |dlogit| 0.3966  flips 158
    int4 per-channel:  acc 0.9710  mean |dlogit| 0.2924  flips 37
    verdict: int8 is free either way. At int4 the shared scale
    costs ~7 accuracy points; one scale per row buys nearly all
    of it back - same 4 bits per weight, spent where rows differ.
  Part 4 - bytes: the whole model, then one production-size layer
    this net (768 weights, biases kept fp32):
      fp32:           3216 B
      int8 + scales:  1056 B
      int4 + scales:  672 B
    one 4096x4096 layer (16.78M weights), per-channel:
      fp32: 67.11 MB | int8: 16.78 MB | int4: 8.39 MB
      all 4096 row scales together: 16384 B - rounding error in the budget, not a line item.
"""
import numpy as np

rng = np.random.default_rng(97)
D_IN, D_HID, D_OUT = 20, 32, 4
HALF = D_IN // 2
N_TR, N_TE = 1200, 2000

# ---------- the data: loud half, quiet half, paired classes ----------
pairA = rng.normal(0.0, 2.0, size=HALF)
pairB = rng.normal(0.0, 2.0, size=HALF)
centers = np.empty((D_OUT, D_IN))
centers[0, :HALF] = pairA
centers[1, :HALF] = pairA          # class 1: same loud center as 0
centers[2, :HALF] = pairB
centers[3, :HALF] = pairB          # class 3: same loud center as 2
centers[:, HALF:] = rng.normal(0.0, 0.20, size=(D_OUT, HALF))
noise = np.concatenate([np.full(HALF, 1.0), np.full(HALF, 0.1)])


def make(n):
    y = rng.integers(0, D_OUT, n)
    X = centers[y] + rng.normal(0.0, 1.0, size=(n, D_IN)) * noise
    return X, y


Xtr, ytr = make(N_TR)
Xte, yte = make(N_TE)

# ---------- Part 1: train a real little model ----------
W1 = rng.normal(0.0, np.sqrt(2.0 / D_IN), size=(D_HID, D_IN))
b1 = np.zeros(D_HID)
W2 = rng.normal(0.0, np.sqrt(2.0 / D_HID), size=(D_OUT, D_HID))
b2 = np.zeros(D_OUT)


def forward(X, w1, w2):
    H = np.maximum(0.0, X @ w1.T + b1)
    return H @ w2.T + b2, H


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


LR, EPOCHS = 0.05, 200
for _ in range(EPOCHS):
    logits, H = forward(Xtr, W1, W2)
    P = softmax(logits)
    P[np.arange(N_TR), ytr] -= 1.0
    g = P / N_TR                              # d(loss)/d(logits)
    gW2, gb2 = g.T @ H, g.sum(0)
    gH = g @ W2
    gH[H <= 0] = 0.0                          # relu gate
    gW1, gb1 = gH.T @ Xtr, gH.sum(0)
    W1 -= LR * gW1
    b1 -= LR * gb1
    W2 -= LR * gW2
    b2 -= LR * gb2

print("Part 1 - a real trained model (numpy, hand-rolled backprop)")
logits_ref, _ = forward(Xte, W1, W2)
pred_ref = logits_ref.argmax(1)
acc_ref = (pred_ref == yte).mean()
row_max = np.abs(W1).max(axis=1)
print(f"  architecture 20 -> 32 (relu) -> 4; {EPOCHS} epochs, full-batch GD")
print(f"  fp32 test accuracy: {acc_ref:.4f} on {N_TE} samples")
print(f"  W1 is {D_HID}x{D_IN} = {W1.size} weights (largest layer); "
      f"W2 is {D_OUT}x{D_HID} = {W2.size}; total {W1.size + W2.size}")
print(f"  W1 row ranges: smallest row max|w| {row_max.min():.3f}, "
      f"largest {row_max.max():.3f} - a "
      f"{row_max.max() / row_max.min():.1f}x spread")
print("  (each row is one hidden unit's incoming weights - different")
print("   units, different jobs, different ranges. Remember this spread.)")


# ---------- the quantizers (Day 65's scheme, two granularities) ----------
def q_tensor(W, qmax):
    """ONE scale for the whole matrix. Returns (dequantized, scale)."""
    s = np.abs(W).max() / qmax
    codes = np.clip(np.round(W / s), -qmax, qmax)
    return codes * s, s


def q_channel(W, qmax):
    """One scale PER ROW (per output channel). Returns (dequant, scales)."""
    s = np.abs(W).max(axis=1, keepdims=True) / qmax
    codes = np.clip(np.round(W / s), -qmax, qmax)
    return codes * s, s.ravel()


def werr(deq):
    err = deq - W1
    return np.abs(err).max(), np.sqrt((err ** 2).mean())


# ---------- Part 2: the weight-error report, on real weights ----------
print("Part 2 - weight error on W1: per-tensor vs per-channel")
variants = {}
for bits, qmax in ((8, 127), (4, 7)):
    for name, fn in (("per-tensor", q_tensor), ("per-channel", q_channel)):
        deq, _ = fn(W1, qmax)
        variants[(bits, name)] = deq
        mx, rms = werr(deq)
        print(f"  int{bits} {name + ':':<13} max err {mx:.6f}  "
              f"rms err {rms:.6f}")
_, scales4 = q_channel(W1, 7)
_, ts4 = q_tensor(W1, 7)
print("  int4 scales, first 8 rows of W1 (per-channel): "
      + ", ".join(f"{v:.4f}" for v in scales4[:8]))
print(f"  the single per-tensor int4 scale for all 640 weights: {ts4:.4f}")
print("  read: max error barely moves (the biggest row sets it under")
print("  both schemes) - the win is in RMS, the typical weight's error.")

# ---------- Part 3: the behavior report ----------
print("Part 3 - behavior: both layers dequantized, same 2000 test samples")
print(f"  {'fp32 (reference)':<18} acc {acc_ref:.4f}  "
      f"mean |dlogit| 0.0000  flips 0")
for bits, qmax in ((8, 127), (4, 7)):
    for name, fn in (("per-tensor", q_tensor), ("per-channel", q_channel)):
        d1, _ = fn(W1, qmax)
        d2, _ = fn(W2, qmax)
        logits_q, _ = forward(Xte, d1, d2)
        drift = np.abs(logits_q - logits_ref).mean()
        flips = int((logits_q.argmax(1) != pred_ref).sum())
        acc_q = (logits_q.argmax(1) == yte).mean()
        print(f"  int{bits} {name + ':':<13} acc {acc_q:.4f}  "
              f"mean |dlogit| {drift:.4f}  flips {flips}")
print("  verdict: int8 is free either way. At int4 the shared scale")
print("  costs ~7 accuracy points; one scale per row buys nearly all")
print("  of it back - same 4 bits per weight, spent where rows differ.")

# ---------- Part 4: bytes, scales included ----------
print("Part 4 - bytes: the whole model, then one production-size layer")
n_w = W1.size + W2.size
n_scales = D_HID + D_OUT                     # per-channel: rows of W1 + W2
bias_bytes = (b1.size + b2.size) * 4         # biases stay fp32
print(f"  this net ({n_w} weights, biases kept fp32):")
print(f"    fp32:           {n_w * 4 + bias_bytes} B")
print(f"    int8 + scales:  {n_w + n_scales * 4 + bias_bytes} B")
print(f"    int4 + scales:  {n_w // 2 + n_scales * 4 + bias_bytes} B")
big = 4096 * 4096
print(f"  one 4096x4096 layer ({big / 1e6:.2f}M weights), per-channel:")
print(f"    fp32: {big * 4 / 1e6:.2f} MB | int8: {big / 1e6:.2f} MB "
      f"| int4: {big / 2 / 1e6:.2f} MB")
print(f"    all 4096 row scales together: {4096 * 4} B - rounding "
      "error in the budget, not a line item.")
