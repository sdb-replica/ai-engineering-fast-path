"""Day 12 - Linear Algebra Bite 1: Vectors, Dot Products, Matrix Multiply.

Tonight's idea: a neuron is a dot product. A layer is a matrix multiply.
  1. Vector + dot product: a weighted sum. w . x + b is the Day-6 neuron.
  2. Matrix @ vector: every row of W dot-products x in one shot -- one layer.
  3. Matrix @ matrix: add a batch dim, score a whole watchlist at once.
  4. The contract: shapes (m,k) @ (k,n) = (m,n) are the type system.

Runs on MacBook: ~/ai-lab venv, torch CPU/MPS. No downloads, no paid APIs.
"""
import torch

torch.manual_seed(12)

# ---- 1. A neuron is a dot product ----
# Day-6's line y = 2.0*x + 1.0, rewritten: w . x + b with w = [2.0], b = 1.0.
# Trading toy: a next-day score from 4 features
#   [ret_1d, ret_5d, log(volume/volume_20d), RSI-50]
x = torch.tensor([0.004, -0.011, 0.18, 0.12])   # one stock-day's features
w = torch.tensor([0.9, 0.4, 0.2, 0.1])          # weights = opinions on features
b = torch.tensor(0.05)                          # bias = base rate
hand = (w * x).sum() + b                        # multiply pairs, add, shift
print("features x :", [round(v, 4) for v in x.tolist()])
print("weights  w :", [round(v, 4) for v in w.tolist()])
print("by hand (w*x).sum() + b :", round(hand.item(), 6))
print("torch.dot(w, x) + b      :", round((torch.dot(w, x) + b).item(), 6))
print("match:", bool(torch.allclose(hand, torch.dot(w, x) + b)))

# ---- 2. A layer is a matrix multiply ----
# Three neurons = three opinions on the same features. Stack their weight
# rows into W -- one matmul computes all three dot products at once.
W = torch.stack([w, w * 0.5, w * -0.3])         # shape (3, 4)
b2 = torch.tensor([0.05, -0.02, 0.0])           # one bias per neuron
y_loop = torch.stack([torch.dot(row, x) + b2[i] for i, row in enumerate(W)])
y_mat = W @ x + b2                              # (3,4) @ (4,) -> (3,)
print("\nW.shape:", tuple(W.shape), "x.shape:", tuple(x.shape),
      "y.shape:", tuple(y_mat.shape))
print("looped dot products:", [round(v, 6) for v in y_loop.tolist()])
print("W @ x + b          :", [round(v, 6) for v in y_mat.tolist()])
print("match:", bool(torch.allclose(y_loop, y_mat)))

# ---- 3. The batch: score a whole watchlist in one matmul ----
# X rows = examples, columns = features: (5,4) @ (4,3) -> (5,3). Zero loops.
X = torch.stack([x, x * 1.5, x * -0.8, x * 2.0, x * 0.3])
scores_loop = torch.stack([W @ row + b2 for row in X])
scores_mat = X @ W.T + b2                       # (5,4) @ (4,3) -> (5,3)
print("\nX.shape:", tuple(X.shape), "W.T.shape:", tuple(W.T.shape),
      "scores.shape:", tuple(scores_mat.shape))
print("batch loop == one matmul:", bool(torch.allclose(scores_loop, scores_mat)))
print("row 0 (base)      :", [round(v, 6) for v in scores_mat[0].tolist()])
print("row 3 (2x leverage):", [round(v, 6) for v in scores_mat[3].tolist()])
lin = scores_mat[3] - b2                        # strip biases, leverage should hold
print("2x leverage on features == 2x on scores (b aside):",
      bool(torch.allclose(lin, 2 * (scores_mat[0] - b2))))
n_mult = X.shape[0] * W.shape[0] * X.shape[1]
print("multiplies in one matmul:", n_mult, "(zero Python loops)")

# ---- 4. The contract: shapes are the type system ----
# torch.nn.functional.linear(x, W, b) == x @ W.T + b. Same numbers, cleaner API.
import torch.nn.functional as F
y_f = F.linear(X, W, b2)                        # (5,4) @ (4,3) -> (5,3)
print("\nF.linear(X, W, b) == X @ W.T + b:", bool(torch.allclose(y_f, scores_mat)))
try:
    X @ W                                       # (5,4) @ (3,4): inner dims mismatch
    print("X @ W: no error (unexpected)")
except RuntimeError as e:
    print("X @ W fails:", str(e).split("\n")[0][:90])
