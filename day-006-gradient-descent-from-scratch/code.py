"""Day 6 — Gradient Descent From Scratch: tiny linear regression, no framework.

One idea: gradient descent = feel the slope of the loss, step downhill, repeat.
No torch.autograd today — we hand-derive dL/dw and dL/db for MSE and write the
loop ourselves. This loop (forward -> loss -> backward -> step) is the ancestor
of every training loop in deep learning.

Runs on any MacBook Python 3.11+; no torch, no numpy, no paid APIs.
"""

import random

# --- Toy data: y = 2x + 1 plus noise (Day 5's loss on Day 6's data) ---
random.seed(42)
xs = [random.uniform(-2, 2) for _ in range(100)]
ys = [2 * x + 1 + random.gauss(0, 0.5) for x in xs]
n = len(xs)


def mse(w, b):
    return sum((w * x + b - y) ** 2 for x, y in zip(xs, ys)) / n


# --- Gradient descent, hand-rolled ---
w, b = 0.0, 0.0   # the knobs, initialized at zero
lr = 0.1          # learning rate: step size

print("=== Gradient descent, lr = 0.1 ===")
for epoch in range(100):
    # forward: predictions + loss (Day 5)
    preds = [w * x + b for x in xs]
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys)) / n
    # backward: hand-derived gradients of MSE. Note the shape:
    #   dw = mean(2 * error * x)   ->  "error x input"
    #   db = mean(2 * error)
    dw = sum(2 * (p - y) * x for p, y, x in zip(preds, ys, xs)) / n
    db = sum(2 * (p - y) for p, y in zip(preds, ys)) / n
    # step: move opposite the slope (downhill)
    w -= lr * dw
    b -= lr * db
    if epoch % 20 == 0:
        print(f"epoch {epoch:3d}  loss={loss:.4f}  w={w:.3f}  b={b:.3f}")

print(f"\nlearned: w={w:.3f} b={b:.3f}   (truth: w=2.000 b=1.000)")
print(f"final MSE: {mse(w, b):.4f}")

# --- The same loop with a learning rate that's way too big ---
print("\n=== Gradient descent, lr = 1.5 (too big) ===")
w, b = 0.0, 0.0
for epoch in range(8):
    preds = [w * x + b for x in xs]
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys)) / n
    dw = sum(2 * (p - y) * x for p, y, x in zip(preds, ys, xs)) / n
    db = sum(2 * (p - y) for p, y in zip(preds, ys)) / n
    w -= 1.5 * dw
    b -= 1.5 * db
    print(f"epoch {epoch}: loss={loss:15.2f}  w={w:12.2f}  b={b:12.2f}")
print("\nOvershoots the bowl, oscillates, and blows up. "
      "Step size is the one knob you must respect.")
