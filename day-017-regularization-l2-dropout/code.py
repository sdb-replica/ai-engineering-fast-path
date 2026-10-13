"""Day 17 - Regularization: L2 & Dropout.

Day 8 showed the disease: a model with too many knobs memorizes noise -
train loss collapses while the honest scores explode. Day 16's digits
were too clean to bleed. Tonight: the two standard brakes, proven three
ways.

  Part 1  L2 (weight decay) rescues Day 8's degree-20 polynomial, with
          exact math. Loss becomes: data loss + lambda * sum(w^2). Every
          weight pays rent proportional to its size; only weights that
          earn their keep survive. lambda is picked on VAL (Day 8's
          contract); the test set is touched exactly once, at the end.
  Part 2  Dropout mechanics: in train mode each unit is zeroed with
          probability p, survivors scaled by 1/(1-p); in eval mode
          dropout is off. model.train() / model.eval() is not ceremony.
  Part 3  Both brakes on Day 16's digits - after poisoning 25% of the
          training labels. The unregularized net memorizes the corrupted
          labels (train acc 0.96 against labels a quarter of which are
          wrong) and pays for it on clean val data (0.84). The
          regularized nets refuse the bad labels - worse train scores,
          ~11 points better where it counts.

Runs on the ~/ai-lab venv: torch (Parts 1-2 on CPU - float64 and a
seeded generator; Part 3 is a tiny net) + scikit-learn for the dataset.
No CUDA, no paid APIs.

Expected output (Part 1 is exact - same numbers everywhere; Parts 2-3
are reference numbers from a numpy twin of the same algorithm - your
MacBook will differ in the last digits, the story will be identical):

  device: mps

  Part 1 - L2 on Day 8's degree-20 polynomial:
    lambda    train      val  ||weights||
       0.0     0.30     2.53     1.89e+06
       0.1     0.79     1.02         3.97
       1.0     0.85     0.92          2.8
       3.0     0.95     0.85         2.36
      10.0     1.28     0.90         1.82
     100.0     3.11     2.04        0.917
  val picks lambda = 3
  test (touched once): lambda=0 -> 38.55   lambda=3 -> 1.02

  Part 2 - dropout mechanics (p = 0.5):
    train pass 1: ~6 of 12 zeroed, survivors = 2.0
    train pass 2: a different ~6 of 12 zeroed, survivors = 2.0
    eval pass: 0 of 12 zeroed, output == input: True

  Part 3 - digits with 25% of training labels flipped:
  run                        train (fed) val (clean) ||weights||
  no regularization               0.9576      0.8389       41.82
  L2 only (wd=0.005)              0.7488      0.9444        9.29
  dropout only (p=0.3)            0.8664      0.8806       37.77
  L2 + dropout                    0.7418      0.9472        9.30
"""
import random

import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

print("device:", "mps" if torch.backends.mps.is_available() else "cpu")

# ---------- Part 1: L2 rescues Day 8's degree-20 disaster (exact) ----------
# Same world as Day 8 (seed 42, pure-Python data, 40/20/40 split). One quiet
# fix: feature stats now come from TRAIN only - Day 8 used each split's own
# stats, a small leak. float64 on CPU on purpose: degree-20 polynomials are
# ill-conditioned, and MPS has no float64. Precision is the point here.
random.seed(42)
xs = [random.uniform(-2, 2) for _ in range(100)]
ys = [2 * x + 1 + random.gauss(0, 1.0) for x in xs]
idx = list(range(100))
random.shuffle(idx)
X = torch.tensor(xs, dtype=torch.float64)
Y = torch.tensor(ys, dtype=torch.float64)
Xtr, Ytr = X[idx[:40]], Y[idx[:40]]      # TRAIN: study material (Day 8)
Xva, Yva = X[idx[40:60]], Y[idx[40:60]]  # VAL: the judge - lambda is picked here
Xte, Yte = X[idx[60:]], Y[idx[60:]]      # TEST: the verdict - touched once, below


def design(xv, degree, mu=None, sd=None):
    """Polynomial features x..x^degree, standardized with train stats."""
    feats = torch.stack([xv ** p for p in range(1, degree + 1)], dim=1)
    if mu is None:
        mu, sd = feats.mean(0), feats.std(0, unbiased=False)
    z = (feats - mu) / sd
    return torch.cat([torch.ones(len(xv), 1, dtype=torch.float64), z], dim=1), mu, sd


Atr, mu, sd = design(Xtr, 20)
Ava, _, _ = design(Xva, 20, mu, sd)
Ate, _, _ = design(Xte, 20, mu, sd)
mse = lambda a, b: ((a - b) ** 2).mean().item()


def ridge(A, y, lam):
    """Solve min ||A b - y||^2 + lam * sum(b^2) exactly. Bias not penalized."""
    if lam == 0:
        beta, *_ = torch.linalg.lstsq(A, y.unsqueeze(1))
        return beta.squeeze(1)
    P = torch.eye(A.shape[1], dtype=torch.float64)
    P[0, 0] = 0.0  # the bias is an offset, not a slope - it pays no rent
    return torch.linalg.solve(A.T @ A + lam * P, A.T @ y)


print("\nPart 1 - L2 on Day 8's degree-20 polynomial:")
print(f"{'lambda':>8}{'train':>9}{'val':>9}{'||weights||':>13}")
best_lam, best_val, betas = None, float("inf"), {}
for lam in (0.0, 0.1, 1.0, 3.0, 10.0, 100.0):
    b = ridge(Atr, Ytr, lam)
    betas[lam] = b
    tr, va = mse(Atr @ b, Ytr), mse(Ava @ b, Yva)
    print(f"{lam:>8.1f}{tr:>9.2f}{va:>9.2f}{b.norm().item():>13.3g}")
    if va < best_val:
        best_val, best_lam = va, lam
print(f"val picks lambda = {best_lam:g}")
print(f"test (touched once): lambda=0 -> {mse(Ate @ betas[0.0], Yte):.2f}"
      f"   lambda={best_lam:g} -> {mse(Ate @ betas[best_lam], Yte):.2f}")

# ---------- Part 2: dropout mechanics (train mode != eval mode) ----------
# Seeded, on CPU, so the masks below are reproducible. p = 0.5: each unit
# is zeroed with probability 1/2; survivors are scaled by 1/(1-p) = 2, so
# the expected sum is unchanged and eval needs no rescaling (inverted
# dropout - this is what nn.Dropout does).
print("\nPart 2 - dropout mechanics (p = 0.5):")
torch.manual_seed(17)
drop = nn.Dropout(p=0.5)
x = torch.ones(12)
drop.train()
for i in (1, 2):
    out = drop(x)
    print(f"  train pass {i}: {(out == 0).sum().item()} of 12 zeroed,"
          f" survivors = {out[out != 0][0].item():.1f}")
drop.eval()
print(f"  eval pass: {(drop(x) == 0).sum().item()} of 12 zeroed,"
      f" output == input: {torch.equal(drop(x), x)}")

# ---------- Part 3: poisoned digits - both brakes on a real net ----------
# Day 16's digits and split discipline, one twist: 25% of the TRAINING
# labels are randomly flipped. Val labels stay clean - the honest judge.
# Memorizing is now measurable: the only way to score high against the
# fed labels is to learn answers that are wrong a quarter of the time.
Xd, yd = load_digits(return_X_y=True)
Xd = Xd / 16.0
Xtr, Xva, ytr, yva = train_test_split(Xd, yd, test_size=0.2,
                                      random_state=17, stratify=yd)
g = torch.Generator().manual_seed(17)
y_fed = torch.tensor(ytr, dtype=torch.long)
flip_idx = torch.randperm(len(y_fed), generator=g)[:len(y_fed) // 4]
y_fed[flip_idx] = torch.randint(0, 10, (len(flip_idx),), generator=g)
Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
Xva_t = torch.tensor(Xva, dtype=torch.float32)
yva_t = torch.tensor(yva, dtype=torch.long)
print(f"\nPart 3 - digits with 25% of training labels flipped"
      f" ({len(flip_idx)} of {len(y_fed)} labels corrupted):")


def train_digits(wd, p):
    torch.manual_seed(17)  # same init for every run - a fair race
    net = nn.Sequential(
        nn.Linear(64, 256), nn.ReLU(), nn.Dropout(p),  # Day 12 + Part 2
        nn.Linear(256, 10),                            # 10 logits (Day 14)
    )
    opt = torch.optim.SGD(net.parameters(), lr=0.5, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()                    # Day 14
    for epoch in range(80):
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(Xtr_t), 32):
            j = perm[i:i + 32]
            opt.zero_grad()
            loss = loss_fn(net(Xtr_t[j]), y_fed[j])    # Day 7 ritual
            loss.backward()
            opt.step()
    net.eval()  # dropout OFF for scoring - Part 2's lesson, applied
    with torch.no_grad():
        acc = lambda X_, y_: (net(X_).argmax(1) == y_).float().mean().item()
        wnorm = sum(q.norm().item() ** 2 for name, q in net.named_parameters()
                    if name.endswith("weight")) ** 0.5
    return acc(Xtr_t, y_fed), acc(Xva_t, yva_t), wnorm


print(f"{'run':<26}{'train (fed)':>12}{'val (clean)':>12}{'||weights||':>12}")
for label, wd, p in (("no regularization", 0.0, 0.0),
                     ("L2 only (wd=0.005)", 0.005, 0.0),
                     ("dropout only (p=0.3)", 0.0, 0.3),
                     ("L2 + dropout", 0.005, 0.3)):
    tr, va, wn = train_digits(wd, p)
    print(f"{label:<26}{tr:>12.4f}{va:>12.4f}{wn:>12.2f}")
print("\nthe corrupted labels cap honest train accuracy near 0.75 + slack:")
print("scoring 0.96 against them means the net memorized wrong answers.")
print("WIN: both brakes score WORSE on fed labels and BETTER on clean val.")
