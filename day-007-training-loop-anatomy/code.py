"""Day 7 — The Training Loop: forward → loss → backward → step.

One idea: every training loop in PyTorch (and in deep learning, period) is
four moves: forward (make predictions) → loss (score them) → backward (get
gradients) → step (nudge the knobs downhill). Today we replay Day 6's tiny
regression in torch, and let autograd replace the hand-derived gradients.

Runs on a MacBook with the ~/ai-lab venv + torch (MPS or CPU). No CUDA, no paid APIs.
"""

import random

import torch

random.seed(42)  # same toy data as Day 6

# --- Data: y = 2x + 1 plus noise (100 points) ---
xs = [random.uniform(-2, 2) for _ in range(100)]
ys = [2 * x + 1 + random.gauss(0, 0.5) for x in xs]
n = len(xs)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {device}")
X = torch.tensor(xs, dtype=torch.float32, device=device)
Y = torch.tensor(ys, dtype=torch.float32, device=device)

# --- The knobs, but this time torch watches them (Day 4's tape recorder) ---
w = torch.tensor(0.0, requires_grad=True, device=device)
b = torch.tensor(0.0, requires_grad=True, device=device)
lr = 0.1

# --- Side by side at epoch 0: yesterday's formula vs autograd ---
preds0 = [w.item() * x + b.item() for x in xs]  # w=0, b=0
loss0 = sum((p - y) ** 2 for p, y in zip(preds0, ys)) / n
dw_hand = sum(2 * (p - y) * x for p, y, x in zip(preds0, ys, xs)) / n
db_hand = sum(2 * (p - y) for p, y in zip(preds0, ys)) / n

preds_t = w * X + b
torch_loss0 = ((preds_t - Y) ** 2).mean()
torch_loss0.backward()
print(f"hand-derived grads: dw={dw_hand:.5f}  db={db_hand:.5f}")
print(f"autograd grads:     dw={w.grad.item():.5f}  db={b.grad.item():.5f}")
w.grad.zero_()
b.grad.zero_()

# --- The loop: forward → loss → backward → step ---
print("\nepoch   loss     w       b")
for epoch in range(100):
    preds = w * X + b                      # forward: predictions
    loss = ((preds - Y) ** 2).mean()       # loss: one number
    loss.backward()                        # backward: fills w.grad, b.grad
    with torch.no_grad():                  # step: bookkeeping, don't record it
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()                         # reset the tape for next epoch
    b.grad.zero_()
    if epoch % 20 == 0:
        print(f"{epoch:3d}   {loss.item():.4f}  {w.item():.3f}  {b.item():.3f}")

print(f"\nlearned: w={w.item():.3f}  b={b.item():.3f}   (truth: 2.000 / 1.000)")
print("Day 6's pure-Python loop converged to w=2.047, b=1.091. Same loop. Same math.")
