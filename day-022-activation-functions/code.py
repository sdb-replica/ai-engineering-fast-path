"""Day 22 - Activation Functions: ReLU, Sigmoid & Tanh.

Day 21 bent the boundary with ReLU and moved on. Today we shop the
bend itself. An activation function is the elementwise function
applied right after z = w.x + b, before the value passes on:

  Part 1  The three classic shapes on the same inputs, with their
          derivatives read off by autograd. Sigmoid squashes to
          (0, 1) and its derivative never tops 0.25. Tanh squashes
          to (-1, 1), zero-centered, derivative up to 1.0. ReLU is
          max(0, z): derivative exactly 1 when on, 0 when off.
  Part 2  The chain-rule tax. A gradient flowing backward is
          MULTIPLIED by the activation's derivative at every neuron
          it passes (Day 4). Through 10 neurons sitting at z = 2:
          sigmoid shrinks it to 1.6279e-10, tanh to 3.0986e-12,
          ReLU passes it through untouched at 1.0. Even at its
          best point (z = 0), sigmoid taxes every hop 4x.
  Part 3  The same XOR network from Day 21 - same hand-set starting
          weights, same optimizer - raced with each activation in
          the hidden layer. All three call the corners 4/4 after
          100 epochs, but the losses tell the real story: ReLU
          0.000005, tanh 0.000020, sigmoid 0.069563. Sigmoid got
          the direction right and the confidence wrong, because
          every gradient it learned from arrived taxed.

Runs on the ~/ai-lab venv: torch only, CPU on purpose (tiny data, no
RNG - starting weights are set by hand, so every digit reproduces
exactly). No CUDA, no paid APIs.

Expected output:

  Part 1 - three shapes on the same inputs, derivatives via autograd
    z       : -4.0000, -2.0000, -1.0000, 0.0000, 1.0000, 2.0000, 4.0000
    sigmoid : 0.0180, 0.1192, 0.2689, 0.5000, 0.7311, 0.8808, 0.9820
    sigmoid': 0.0177, 0.1050, 0.1966, 0.2500, 0.1966, 0.1050, 0.0177
    tanh    : -0.9993, -0.9640, -0.7616, 0.0000, 0.7616, 0.9640, 0.9993
    tanh'   : 0.0013, 0.0707, 0.4200, 1.0000, 0.4200, 0.0707, 0.0013
    relu    : 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 2.0000, 4.0000
    relu'   : 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 1.0000

  Part 2 - the chain-rule tax: gradient through neurons at z = 2
    sigmoid  grad 0.1050 -> after 1: 1.0499e-01, after 5: 1.2759e-05, after 10: 1.6279e-10
    tanh     grad 0.0707 -> after 1: 7.0651e-02, after 5: 1.7603e-06, after 10: 3.0986e-12
    relu     grad 1.0000 -> after 1: 1.0000e+00, after 5: 1.0000e+00, after 10: 1.0000e+00
    sigmoid at its best (z = 0, grad 0.2500): after 10: 9.5367e-07

  Part 3 - same XOR network, three activations (SGD, lr=0.5, 100 epochs)
    relu     loss 0.070625 -> 0.000005  predictions 0.0032, 1.0000, 0.9999, -0.0029  (correct: 4/4)
    tanh     loss 0.669811 -> 0.000020  predictions -0.0039, 0.9957, 0.9955, -0.0050  (correct: 4/4)
    sigmoid  loss 0.793775 -> 0.069563  predictions 0.3106, 0.7527, 0.7340, 0.2233  (correct: 4/4)
"""
import torch
import torch.nn as nn

fmt = lambda t: ", ".join(f"{v:.4f}" for v in t.flatten().tolist())

# ---------- Part 1: three shapes, derivatives via autograd ----------
# Same seven inputs through each function. The derivative rows come
# from autograd itself: sum the outputs, call backward(), read z.grad
# - Day 4's tape recorder doing the calculus for us.
print("Part 1 - three shapes on the same inputs, derivatives via autograd")
z = torch.tensor([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0],
                 requires_grad=True)
print(f"  {'z':8s}: {fmt(z)}")
for name, fn in [("sigmoid", torch.sigmoid), ("tanh", torch.tanh),
                 ("relu", torch.relu)]:
    if z.grad is not None:
        z.grad = None
    out = fn(z)
    out.sum().backward()
    print(f"  {name:8s}: {fmt(out)}")
    print(f"  {name + chr(39):8s}: {fmt(z.grad)}")

# ---------- Part 2: the chain-rule tax ----------
# Backprop multiplies the incoming gradient by the activation's
# derivative at each neuron (the chain rule from Day 4). If a signal
# passes through N neurons all sitting at the same z, it comes out
# scaled by derivative ** N. Watch what 10 hops do at z = 2.
print("\nPart 2 - the chain-rule tax: gradient through neurons at z = 2")
for name, fn in [("sigmoid", torch.sigmoid), ("tanh", torch.tanh),
                 ("relu", torch.relu)]:
    zz = torch.tensor([2.0], requires_grad=True)
    fn(zz).backward()
    d = zz.grad.item()
    print(f"  {name:8s} grad {d:.4f} -> after 1: {d:.4e}, "
          f"after 5: {d ** 5:.4e}, after 10: {d ** 10:.4e}")
print(f"  sigmoid at its best (z = 0, grad 0.2500): "
      f"after 10: {0.25 ** 10:.4e}")

# ---------- Part 3: same network, three activations ----------
# Day 21's XOR race, rerun as a controlled experiment: identical
# hand-set starting weights, data, optimizer and epoch count - the
# ONLY thing that changes is the hidden layer's activation.
print("\nPart 3 - same XOR network, three activations "
      "(SGD, lr=0.5, 100 epochs)")
Xx = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])


def train(act):
    mlp = nn.Sequential(nn.Linear(2, 4), act, nn.Linear(4, 1))
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
    return first, final, pred, correct


for name, act in [("relu", nn.ReLU()), ("tanh", nn.Tanh()),
                  ("sigmoid", nn.Sigmoid())]:
    first, final, pred, correct = train(act)
    print(f"  {name:8s} loss {first:.6f} -> {final:.6f}  "
          f"predictions {fmt(pred)}  (correct: {correct}/4)")
