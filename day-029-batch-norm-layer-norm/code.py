"""Day 29 - Batch Norm & Layer Norm: Stabilizing Training.

Training keeps moving the ground under its own feet. Every step
updates the early layers, so the numbers flowing INTO the later
layers drift in mean and spread -- and a layer downstream is
trying to learn on a target that will not sit still. Stack the
drift deep enough and it compounds geometrically: Part 3 grows
an activation std from 1.0 to 18.9 in five x1.8 layers.

Normalization is an in-network reset, applied on every pass:

  x_hat = (x - mean) / sqrt(var + eps)     # force mean 0, std 1
  out   = gamma * x_hat + beta            # then a LEARNED rescale

gamma and beta are ordinary learned parameters, one pair per
feature. If the layer truly wants the old scale back it can
learn gamma = std, beta = mean and undo the norm exactly -- so
normalization is a tame default, not a straitjacket. eps (1e-5)
only guards against dividing by a zero variance.

The two flavors differ ONLY in which slice feeds the mean/var:

  BatchNorm (nn.BatchNorm1d): per FEATURE, across the batch --
    statistics run down each column of the (batch, features)
    matrix. Because an example's output then depends on its
    batchmates, eval cannot use batch statistics (a batch of
    one has no spread): during training it keeps running
    averages, and eval mode swaps those in. That is Day 25's
    train/eval switch with real teeth.
  LayerNorm (nn.LayerNorm): per EXAMPLE, across its features --
    statistics run across each row. No batch anywhere, no
    running stats, train and eval identical, batch size 1 fine.
    That is why sequence and transformer models use it.

  Part 1  BatchNorm by hand on a 4x3 batch whose features live
          on wildly different scales (~100, ~2, ~9): every
          column lands at mean 0, std 1, matching nn.BatchNorm1d
          exactly; then gamma/beta rescale on purpose.
  Part 2  LayerNorm by hand on the same batch: every ROW lands
          at mean 0, std 1, matching nn.LayerNorm exactly. Then
          the coupling test: change one value in row 0 and watch
          BatchNorm move the OTHER rows (max change 1.1722)
          while LayerNorm leaves them untouched (0.0000).
  Part 3  The mode bill comes due: after ONE training batch the
          running stats have moved 10% of the way (momentum 0.1)
          toward the batch stats, so eval-mode row 0 is nowhere
          near its train-mode value; LayerNorm's train/eval diff
          is 0.000000. Plus the depth demo: five x1.8 layers take
          a std from 1.0 to 18.8956 with no norm, and LayerNorm
          after each layer pins it at 1.0000 every time.

Runs on the ~/ai-lab venv: torch only (free), CPU or MPS, no
CUDA, no paid APIs, no downloads. No randomness anywhere --
every number below is hand-set, so your output should match
the expected output digit for digit (std is always the biased,
divide-by-N flavor: unbiased=False).

Expected output:

  device: mps
  Part 1 - BatchNorm: stats per FEATURE (down each column)
    batch mean per feature: 100.0000, 2.0000, 9.0000
    batch std per feature: 1.5811, 0.7071, 2.2361
    normalized rows:
      [+1.2649, +0.0000, -0.4472]
      [-1.2649, -1.4142, +1.3416]
      [+0.6325, +1.4142, +0.4472]
      [-0.6325, +0.0000, -1.3416]
    largest |column mean| after: 0.0000
    column stds after: 1.0000, 1.0000, 1.0000
    hand vs nn.BatchNorm1d max diff: 0.000000
    gamma [2, 1, 0.5] beta [10, 0, -1] -- row 0: [+12.5298, +0.0000, -1.2236]
    scaled column means (= beta): 10.0000, 0.0000, -1.0000
    scaled column stds (= gamma): 2.0000, 1.0000, 0.5000
  Part 2 - LayerNorm: stats per EXAMPLE (across each row)
    row means before: 37.3333, 37.0000, 38.0000, 35.6667
    normalized rows:
      [+1.4122, -0.7716, -0.6406]
      [+1.4066, -0.8301, -0.5765]
      [+1.4113, -0.7841, -0.6273]
      [+1.4133, -0.7513, -0.6620]
    largest |row mean| after: 0.0000
    row stds after: 1.0000, 1.0000, 1.0000, 1.0000
    hand vs nn.LayerNorm max diff: 0.000000
    change row 0 first value 102 -> 202:
      BatchNorm: other rows move, max change 1.1722
      LayerNorm: other rows move, max change 0.0000
  Part 3 - running stats, eval mode, and depth
    after ONE training batch (momentum 0.1):
      running_mean: 10.0000, 0.2000, 0.9000
      running_var: 1.2333, 0.9667, 1.5667
    BatchNorm row 0 -- train mode: [+1.2649, +0.0000, -0.4472]
    BatchNorm row 0 -- eval mode: [+82.8411, +1.8308, +5.6724]
    LayerNorm train vs eval max diff: 0.000000
    five x1.8 layers, no norm -- feature-1 std: 1.8000, 3.2400, 5.8320, 10.4976, 18.8956
    five x1.8 layers, LayerNorm each -- row std: 1.0000, 1.0000, 1.0000, 1.0000, 1.0000
"""
import torch
import torch.nn as nn

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)
EPS = 1e-5

# One batch, three features on wildly different scales (~100, ~2, ~9)
X = torch.tensor([[102.0, 2.0, 8.0],
                  [98.0, 1.0, 12.0],
                  [101.0, 3.0, 10.0],
                  [99.0, 2.0, 6.0]], device=device)

def fmt(vals):
    return ", ".join(f"{v:.4f}" for v in vals)

def fmt_row(vals):
    return "[" + ", ".join(f"{v:+.4f}" for v in vals) + "]"

# ---------- Part 1: BatchNorm by hand ----------
print("Part 1 - BatchNorm: stats per FEATURE (down each column)")
mean = X.mean(dim=0)                              # one mean per column
var = X.var(dim=0, unbiased=False)                # biased var, as BN uses
Xn = (X - mean) / torch.sqrt(var + EPS)
print("  batch mean per feature:", fmt(mean.tolist()))
print("  batch std per feature:", fmt(torch.sqrt(var + EPS).tolist()))
print("  normalized rows:")
for row in Xn.tolist():
    print("   ", fmt_row(row))
print(f"  largest |column mean| after: {Xn.mean(dim=0).abs().max():.4f}")
print("  column stds after:", fmt(Xn.std(dim=0, unbiased=False).tolist()))
bn = nn.BatchNorm1d(3).to(device)                 # gamma=1, beta=0 to start
bn.train()                                        # train mode: batch stats
with torch.no_grad():
    bn_out = bn(X)                                # also updates running stats
diff = (Xn - bn_out).abs().max().item()
print(f"  hand vs nn.BatchNorm1d max diff: {diff:.6f}")
gamma = torch.tensor([2.0, 1.0, 0.5], device=device)   # what bn.weight learns
beta = torch.tensor([10.0, 0.0, -1.0], device=device)  # what bn.bias learns
scaled = Xn * gamma + beta
print(f"  gamma [2, 1, 0.5] beta [10, 0, -1] -- row 0: {fmt_row(scaled[0].tolist())}")
print("  scaled column means (= beta):", fmt(scaled.mean(dim=0).tolist()))
print("  scaled column stds (= gamma):",
      fmt(scaled.std(dim=0, unbiased=False).tolist()))

# ---------- Part 2: LayerNorm by hand + the coupling test ----------
print("Part 2 - LayerNorm: stats per EXAMPLE (across each row)")
rmean = X.mean(dim=1, keepdim=True)               # one mean per row
rvar = X.var(dim=1, unbiased=False, keepdim=True)
Ln = (X - rmean) / torch.sqrt(rvar + EPS)
print("  row means before:", fmt(X.mean(dim=1).tolist()))
print("  normalized rows:")
for row in Ln.tolist():
    print("   ", fmt_row(row))
print(f"  largest |row mean| after: {Ln.mean(dim=1).abs().max():.4f}")
print("  row stds after:", fmt(Ln.std(dim=1, unbiased=False).tolist()))
ln = nn.LayerNorm(3).to(device)
with torch.no_grad():
    ln_out = ln(X)
diff = (Ln - ln_out).abs().max().item()
print(f"  hand vs nn.LayerNorm max diff: {diff:.6f}")
X2 = X.clone()
X2[0, 0] = 202.0                                  # touch ONE value in row 0
m2 = X2.mean(dim=0)
v2 = X2.var(dim=0, unbiased=False)
Xn2 = (X2 - m2) / torch.sqrt(v2 + EPS)
rm2 = X2.mean(dim=1, keepdim=True)
rv2 = X2.var(dim=1, unbiased=False, keepdim=True)
Ln2 = (X2 - rm2) / torch.sqrt(rv2 + EPS)
print("  change row 0 first value 102 -> 202:")
print(f"    BatchNorm: other rows move, max change "
      f"{(Xn2[1:] - Xn[1:]).abs().max():.4f}")
print(f"    LayerNorm: other rows move, max change "
      f"{(Ln2[1:] - Ln[1:]).abs().max():.4f}")

# ---------- Part 3: running stats, eval mode, and depth ----------
print("Part 3 - running stats, eval mode, and depth")
print("  after ONE training batch (momentum 0.1):")
print("    running_mean:", fmt(bn.running_mean.tolist()))
print("    running_var:", fmt(bn.running_var.tolist()))
bn.eval()                                         # eval mode: running stats
with torch.no_grad():
    eval_out = bn(X)
print(f"  BatchNorm row 0 -- train mode: {fmt_row(bn_out[0].tolist())}")
print(f"  BatchNorm row 0 -- eval mode: {fmt_row(eval_out[0].tolist())}")
ln.train()
with torch.no_grad():
    ln_train = ln(X)
ln.eval()
with torch.no_grad():
    ln_eval = ln(X)
print(f"  LayerNorm train vs eval max diff: "
      f"{(ln_train - ln_eval).abs().max():.6f}")
Y = Xn.clone()                                    # features already std 1
stds = []
for _ in range(5):
    Y = Y * 1.8                                   # one over-eager layer
    stds.append(Y.std(dim=0, unbiased=False)[0].item())
print("  five x1.8 layers, no norm -- feature-1 std:", fmt(stds))
Z = Xn.clone()
ln2 = nn.LayerNorm(3).to(device)                  # gamma=1, beta=0
row_stds = []
with torch.no_grad():
    for _ in range(5):
        Z = ln2(Z * 1.8)                          # same growth, normed each time
        row_stds.append(Z.std(dim=1, unbiased=False).mean().item())
print("  five x1.8 layers, LayerNorm each -- row std:", fmt(row_stds))
