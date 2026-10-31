"""Day 34 - What a Neural Net Is: A Composed Function + an Optimizer.

Phase B recap. Fourteen days of new machinery - perceptrons,
activations, backprop, Modules, CNNs, RNNs, norms, init - and
tonight it all compresses into one sentence:

    A neural net is ONE composed differentiable function
    f_theta(x), plus an optimizer that rewrites theta from
    the function's own gradients.

Composed: the output layer eats the hidden layer's output,
which ate the input - f(x) = L2(relu(L1(x))). Differentiable:
every piece in the chain has a derivative, so Day 23's blame
signal can flow from the loss back to every number in theta.
Theta IS the function: the architecture is the shape of the
composition, the 17 numbers in theta are the function itself.

  Part 1  Unroll the composition by hand on Day 21's XOR net.
          For input [0, 1]: z1, the ReLU gate, and the final
          sum, digit by digit - then the same formula on all
          four corners matches the nn.Module exactly. Theta
          is 8 + 4 + 4 + 1 = 17 numbers, nothing else.
  Part 2  One generic train() loop, two different functions.
          Swap only the hidden activation - ReLU vs tanh,
          Day 22's race - and the loop does not change by one
          character: it only ever calls model(X), reads a
          scalar loss, calls backward(), and steps. The
          optimizer serves the composition, any composition.
  Part 3  The two halves, separated. Run the function 100
          times with NO optimizer step and the loss is frozen
          at 0.070625 - a function alone never learns. Train
          it, then restore the saved theta snapshot and the
          loss returns to exactly 0.070625: change theta,
          change the function; restore theta, restore it.
          The inventory underneath shows every Phase B
          architecture - MLP, CNN, RNN, tabular MLP - is
          the same object: an nn.Module with a theta you
          can count.

Runs on the ~/ai-lab venv: torch only (free), CPU on
purpose (tiny data), no CUDA, no paid APIs, no downloads.
Every weight is set by hand, so every number is exact -
no RNG anywhere in this lesson.

Expected output:

  Part 1 - unroll f_theta: L2(relu(L1(x))), by hand
    input [0.0, 1.0]: z1 = -1.0000, 1.0000, 0.0000, 0.8000
      after ReLU: h = 0.0000, 1.0000, 0.0000, 0.8000  -> pred 0.6000 (target 1)
    all four preds, by hand: 0.0000, 0.6000, 1.0000, 0.3500  loss 0.070625
    nn.Module vs by-hand formula: max diff 0.000000
    theta = W1 8 + b1 4 + W2 4 + b2 1 = 17 numbers - they ARE the function
  Part 2 - one generic loop, two different functions (SGD, lr=0.5, 100 epochs)
    relu  loss 0.070625 -> 0.000005  predictions 0.0032, 1.0000, 0.9999, -0.0029  (correct: 4/4)
    tanh  loss 0.669811 -> 0.000020  predictions -0.0039, 0.9957, 0.9955, -0.0050  (correct: 4/4)
    the loop changed by zero lines - only the piece inside f changed
  Part 3 - no optimizer, no learning; theta restored, function restored
    100 forwards, no step: loss 0.070625 -> 0.070625  (changed: False)
    after 100 optimizer steps: loss 0.000005
    snapshot restored: loss 0.070625  preds vs original max diff 0.000000
    every Phase B architecture is the same object - f_theta with a countable theta:
      MLP 2->4->1 (Days 21-25):        17 params
      Conv2d 1->8, k=3 (Days 26-27):   80 params
      RNN cell 1->2 (Day 28):          10 params
      TabMLP 5->16->1 (Day 32):        113 params
    verdict: architecture = shape of the composition, theta = the function, optimizer = the editor
"""
import torch
import torch.nn as nn

# Day 21's hand-set init - the same starting function all Phase B.
W1_INIT = torch.tensor([[1.0, -1.0], [-1.0, 1.0],
                        [0.5, 0.5], [-0.5, 0.8]])
B1_INIT = torch.tensor([0.0, 0.0, -0.5, 0.0])
W2_INIT = torch.tensor([[1.0, 1.0, 1.0, -0.5]])
B2_INIT = torch.tensor([0.0])
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])          # XOR (Day 21)

fmt = lambda t: ", ".join(f"{v:.4f}" for v in t.flatten().tolist())


def build(act):
    net = nn.Sequential(nn.Linear(2, 4), act, nn.Linear(4, 1))
    with torch.no_grad():
        net[0].weight.copy_(W1_INIT)
        net[0].bias.copy_(B1_INIT)
        net[2].weight.copy_(W2_INIT)
        net[2].bias.copy_(B2_INIT)
    return net


# ---------- Part 1: unroll the composition by hand ----------
print("Part 1 - unroll f_theta: L2(relu(L1(x))), by hand")
x_one = torch.tensor([0.0, 1.0])
with torch.no_grad():
    z1 = x_one @ W1_INIT.T + B1_INIT                   # L1: linear piece
    h = torch.relu(z1)                                 # the bend (Day 22)
    pred_one = (h @ W2_INIT.T + B2_INIT).item()        # L2: linear piece
print(f"  input [0.0, 1.0]: z1 = {fmt(z1)}")
print(f"    after ReLU: h = {fmt(h)}  -> pred {pred_one:.4f} (target 1)")
with torch.no_grad():
    manual = torch.relu(X @ W1_INIT.T + B1_INIT) @ W2_INIT.T + B2_INIT
    loss0 = ((manual - y) ** 2).mean().item()
    module_pred = build(nn.ReLU())(X)
    diff = (manual - module_pred).abs().max().item()
print(f"  all four preds, by hand: {fmt(manual)}  loss {loss0:.6f}")
print(f"  nn.Module vs by-hand formula: max diff {diff:.6f}")
n_theta = W1_INIT.numel() + B1_INIT.numel() + W2_INIT.numel() \
    + B2_INIT.numel()
print(f"  theta = W1 {W1_INIT.numel()} + b1 {B1_INIT.numel()} + "
      f"W2 {W2_INIT.numel()} + b2 {B2_INIT.numel()} = {n_theta} numbers"
      " - they ARE the function")

# ---------- Part 2: one generic loop, two different functions ----------
print("Part 2 - one generic loop, two different functions "
      "(SGD, lr=0.5, 100 epochs)")


def train(model):
    """The Phase B loop (Days 7/25). It knows nothing about the
    model except: model(X) is differentiable, loss is a scalar."""
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    first = None
    for _ in range(100):
        opt.zero_grad()
        loss = ((model(X) - y) ** 2).mean()
        if first is None:
            first = loss.item()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(X)
        final = ((pred - y) ** 2).mean().item()
        correct = int(((pred > 0.5) == (y > 0.5)).sum().item())
    return first, final, pred, correct


for name, act in (("relu", nn.ReLU()), ("tanh", nn.Tanh())):
    first, final, pred, correct = train(build(act))
    print(f"  {name:<5} loss {first:.6f} -> {final:.6f}  "
          f"predictions {fmt(pred)}  (correct: {correct}/4)")
print("  the loop changed by zero lines - only the piece inside f changed")

# ---------- Part 3: separate the two halves ----------
print("Part 3 - no optimizer, no learning; theta restored, "
      "function restored")
net = build(nn.ReLU())
snapshot = {k: v.clone() for k, v in net.state_dict().items()}
with torch.no_grad():
    before = ((net(X) - y) ** 2).mean().item()
    for _ in range(100):                               # forwards only
        _ = net(X)
    frozen = ((net(X) - y) ** 2).mean().item()
print(f"  100 forwards, no step: loss {before:.6f} -> {frozen:.6f}"
      f"  (changed: {bool(before != frozen)})")
_, trained_final, _, _ = train(net)
print(f"  after 100 optimizer steps: loss {trained_final:.6f}")
net.load_state_dict(snapshot)
with torch.no_grad():
    restored = ((net(X) - y) ** 2).mean().item()
    orig = torch.relu(X @ W1_INIT.T + B1_INIT) @ W2_INIT.T + B2_INIT
    rdiff = (net(X) - orig).abs().max().item()
print(f"  snapshot restored: loss {restored:.6f}  "
      f"preds vs original max diff {rdiff:.6f}")
print("  every Phase B architecture is the same object - "
      "f_theta with a countable theta:")
inventory = (
    ("MLP 2->4->1 (Days 21-25):", build(nn.ReLU())),
    ("Conv2d 1->8, k=3 (Days 26-27):", nn.Conv2d(1, 8, 3, padding=1)),
    ("RNN cell 1->2 (Day 28):", nn.RNN(1, 2, batch_first=True)),
    ("TabMLP 5->16->1 (Day 32):",
     nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 1))),
)
for label, module in inventory:
    count = sum(p.numel() for p in module.parameters())
    print(f"    {label:<32} {count} params")
print("  verdict: architecture = shape of the composition, "
      "theta = the function, optimizer = the editor")
