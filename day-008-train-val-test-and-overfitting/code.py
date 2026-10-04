"""Day 8 — Train/Val/Test & Overfitting: the generalization contract.

One idea: training loss is a grade the student gives themselves — rigged.
You need three piles: TRAIN (study material), VAL (the judge you tune
against), TEST (the verdict — locked away, touched once). Tonight we fit
three models of growing flexibility on the same toy world as Days 6/7 and
watch the overfit model's train loss collapse while its honest loss explodes.

Runs on a MacBook with the ~/ai-lab venv + torch (MPS or CPU). No CUDA, no paid APIs.
"""

import random

import torch

random.seed(42)  # same toy world as Days 6/7, noisier this time

# --- Data: y = 2x + 1 + noise (100 points), then the honest 40/20/40 split ---
xs = [random.uniform(-2, 2) for _ in range(100)]
ys = [2 * x + 1 + random.gauss(0, 1.0) for x in xs]
idx = list(range(100))
random.shuffle(idx)  # shuffle BEFORE splitting — order is information

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {device}")
X = torch.tensor(xs, dtype=torch.float32, device=device)
Y = torch.tensor(ys, dtype=torch.float32, device=device)

Xtr, Ytr = X[idx[:40]], Y[idx[:40]]      # TRAIN: the study material
Xva, Yva = X[idx[40:60]], Y[idx[40:60]]  # VAL: the judge (pick the model here)
Xte, Yte = X[idx[60:]], Y[idx[60:]]      # TEST: the verdict — touched once, below


def design(xv, degree):
    """Polynomial features x, x^2, ..., x^degree + bias, standardized."""
    feats = torch.stack([xv ** p for p in range(1, degree + 1)], dim=1)
    z = (feats - feats.mean(0)) / feats.std(0, unbiased=False)
    return torch.cat([torch.ones(len(xv), 1, device=device), z], dim=1)


def fit_and_score(degree):
    beta, *_ = torch.linalg.lstsq(design(Xtr, degree), Ytr.unsqueeze(1))
    beta = beta.squeeze(1)  # exact least squares — no gradient descent
    mse = lambda a, b: ((a - b) ** 2).mean().item()
    return (mse(design(Xtr, degree) @ beta, Ytr),
            mse(design(Xva, degree) @ beta, Yva),
            mse(design(Xte, degree) @ beta, Yte))


print(f"\n{'model':<12}{'train':>8}{'val':>8}{'test':>8}")
results = {}
for deg in (1, 8, 20):
    tr, va, te = fit_and_score(deg)
    results[deg] = (tr, va, te)
    print(f"degree {deg:<4}{tr:>8.2f}{va:>8.2f}{te:>8.2f}")

best = min(results, key=lambda d: results[d][1])
print(f"\nval picks degree {best}  (val MSE {results[best][1]:.2f} — the honest score)")
print(f"test verdict: degree {best} MSE {results[best][2]:.2f} "
      f"vs degree 20 MSE {results[20][2]:.2f}  <-- overfitting, quantified")
print("\nthe test set was touched exactly once: right here, at the end.")