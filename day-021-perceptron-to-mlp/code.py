"""Day 21 - Perceptron to MLP: Stacking Layers With Non-Linearities.

Phase B opens with the smallest neural network there is. A perceptron
is one neuron: pred = w.x + b, a weighted vote whose decision boundary
is a single straight line. Stacking neurons into a hidden layer is
what turns that line into a curve - but ONLY if a non-linearity sits
between the layers.

  Part 1  Stack two linear layers with no bend and they collapse into
          one: W2(W1x + b1) + b2 = (W2W1)x + (W2b1 + b2). The two-layer
          outputs match the single effective layer exactly (diff 0).
  Part 2  One neuron vs XOR, the classic inseparable problem: the
          targets are 1 on the off-diagonal corners, 0 on the diagonal.
          Training converges to predicting 0.5 for everything - the
          loss parks at 0.2500, the honest "I cannot draw this line."
  Part 3  The same data, an MLP (2 -> 4 with ReLU -> 1), weights set
          by hand so every digit reproduces: loss 0.070625 -> 0.000005,
          all four corners called correctly. The hidden layer did not
          add a fancier line; it bent the boundary.

Runs on the ~/ai-lab venv: torch only, CPU on purpose (tiny data, no
RNG, every number exact). No CUDA, no paid APIs.

Expected output:

  Part 1 - two linear layers collapse into one
    effective W = [3.0000, 5.0000], b = 2.5000
    two layers : 5.5000, 7.5000, 10.5000, 3.5000
    one layer  : 5.5000, 7.5000, 10.5000, 3.5000
    max difference: 0.000000

  Part 2 - one neuron vs XOR (SGD, lr=0.5, 300 epochs)
    loss 0.4650 -> 0.2500
    predictions 0.5000, 0.5000, 0.5000, 0.5000  (correct: 2/4)

  Part 3 - MLP with a ReLU hidden layer vs XOR (SGD, lr=0.5, 100 epochs)
    loss 0.070625 -> 0.000005
    predictions 0.0032, 1.0000, 0.9999, -0.0029  (correct: 4/4)
"""
import torch
import torch.nn as nn

# ---------- Part 1: two linear layers collapse into one ----------
# Layer 1 maps 2 -> 3, layer 2 maps 3 -> 1. No non-linearity anywhere.
print("Part 1 - two linear layers collapse into one")
lin1 = nn.Linear(2, 3)
lin2 = nn.Linear(3, 1)
with torch.no_grad():
    lin1.weight.copy_(torch.tensor([[1.0, 2.0], [-1.0, 0.5], [0.0, 3.0]]))
    lin1.bias.copy_(torch.tensor([0.5, -1.0, 2.0]))
    lin2.weight.copy_(torch.tensor([[2.0, -1.0, 0.5]]))
    lin2.bias.copy_(torch.tensor([-0.5]))
X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]])
with torch.no_grad():
    two = lin2(lin1(X))                                # through both layers
    W_eff = lin2.weight @ lin1.weight                  # the collapse, in math
    b_eff = lin2.weight @ lin1.bias + lin2.bias
    one = X @ W_eff.T + b_eff                          # one single layer
print(f"  effective W = [{W_eff[0, 0]:.4f}, {W_eff[0, 1]:.4f}], "
      f"b = {b_eff.item():.4f}")
fmt = lambda t: ", ".join(f"{v:.4f}" for v in t.flatten().tolist())
print(f"  two layers : {fmt(two)}")
print(f"  one layer  : {fmt(one)}")
print(f"  max difference: {(two - one).abs().max().item():.6f}")

# ---------- Part 2: one neuron vs XOR ----------
# XOR targets: 1 when exactly one input is on. No straight line splits
# the 1-corners from the 0-corners (see the diagram), so the best a
# single neuron can do is shrug: predict 0.5 for every corner.
print("\nPart 2 - one neuron vs XOR (SGD, lr=0.5, 300 epochs)")
Xx = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
neuron = nn.Linear(2, 1)
with torch.no_grad():
    neuron.weight.copy_(torch.tensor([[0.2, -0.1]]))
    neuron.bias.copy_(torch.tensor([0.0]))
opt = torch.optim.SGD(neuron.parameters(), lr=0.5)
first = None
for _ in range(300):
    opt.zero_grad()
    loss = ((neuron(Xx) - y) ** 2).mean()
    if first is None:
        first = loss.item()
    loss.backward()
    opt.step()
with torch.no_grad():
    pred = neuron(Xx)
    final = ((pred - y) ** 2).mean().item()
    correct = int(((pred > 0.5) == (y > 0.5)).sum())
print(f"  loss {first:.4f} -> {final:.4f}")
print(f"  predictions {fmt(pred)}  (correct: {correct}/4)")

# ---------- Part 3: an MLP bends the boundary ----------
# Same XOR data, one new ingredient: a hidden layer of 4 neurons with
# ReLU between the two linear maps. Each hidden neuron still draws
# its own line, but ReLU folds its output at zero, and the output
# neuron adds up folded pieces - a bent, piecewise boundary.
# (Weights are set by hand instead of drawn at random, so the run is
# RNG-free and every digit below reproduces exactly.)
print("\nPart 3 - MLP with a ReLU hidden layer vs XOR (SGD, lr=0.5, 100 epochs)")
mlp = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
with torch.no_grad():
    mlp[0].weight.copy_(torch.tensor([[1.0, -1.0], [-1.0, 1.0],
                                      [0.5, 0.5], [-0.5, 0.8]]))
    mlp[0].bias.copy_(torch.tensor([0.0, 0.0, -0.5, 0.0]))
    mlp[2].weight.copy_(torch.tensor([[1.0, 1.0, 1.0, -0.5]]))
    mlp[2].bias.copy_(torch.tensor([0.0]))
opt = torch.optim.SGD(mlp.parameters(), lr=0.5)
first = None
for _ in range(100):
    opt.zero_grad()
    loss = ((mlp(Xx) - y) ** 2).mean()
    if first is None:
        first = loss.item()
    loss.backward()
    opt.step()
with torch.no_grad():
    pred = mlp(Xx)
    final = ((pred - y) ** 2).mean().item()
    correct = int(((pred > 0.5) == (y > 0.5)).sum())
print(f"  loss {first:.6f} -> {final:.6f}")
print(f"  predictions {fmt(pred)}  (correct: {correct}/4)")
