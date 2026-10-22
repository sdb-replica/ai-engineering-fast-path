"""Day 25 - Training Loop, Done Right: Device, Eval Mode & no_grad.

Day 24 gave you a clean model class. Real code also needs a clean
LOOP around it, and that loop has three independent switches that
beginners constantly conflate:

  device      WHERE tensors live. Pick it once:
              "mps" if torch.backends.mps.is_available() else
              "cpu" - then move the model AND every batch with
              .to(device). Model on MPS + batch on CPU is the
              classic RuntimeError: compute and data must
              co-locate, always.
  train/eval  HOW layers behave. model.train() / model.eval() flip
              a mode flag: dropout (Day 17) drops units in train
              mode and is the identity in eval mode. But eval()
              does NOT stop gradient recording - Part 1 proves a
              forward pass in eval mode still requires_grad.
  no_grad     WHETHER the autograd tape (Day 4) records at all.
              Under torch.no_grad() nothing is taped: no grads
              are populated, memory stays flat, and a measurement
              pass cannot pollute training. Measuring needs BOTH
              switches: eval() for behaviour, no_grad() for tape.

  Part 1  A DropoutNet (Day 24's XORNet + one nn.Dropout) with
          hand-set weights. In eval mode two passes are identical
          (max diff 0.000000, preds 0.0000/0.6000/1.0000/0.3500);
          in train mode the mask is redrawn and passes vary. And
          eval without no_grad still records the tape.
  Part 2  evaluate(): the reusable measurement helper - eval() +
          no_grad(), then it RESTORES train mode if the model was
          training. On the untrained net it reports the familiar
          Day 21 loss 0.070625 and leaves every .grad at None,
          while one train-mode backward() populates them all.
  Part 3  The full loop, done right, on the selected device:
          co-location check, model.train() every epoch, and
          evaluate() for the honest before/after numbers. Same
          weights and math as Day 24, so the CPU reference result
          is byte-identical: 0.070625 -> 0.000005, 4/4 corners.
          (On MPS the final digits may differ in the last place;
          the arc and the 4/4 are the guarantee.)

Runs on the ~/ai-lab venv: torch only, hand-set weights - no RNG
in any printed number except Part 1's train-mode dropout masks,
where only the structural claim (passes vary) is printed. No CUDA,
no paid APIs.

Expected output:

  Part 1 - train vs eval: dropout is a mode, not a layer you remove
    eval preds: 0.0000, 0.6000, 1.0000, 0.3500
    two eval passes differ by: 0.000000
    eval pass (no no_grad) requires_grad: True
    train-mode passes vary across calls: True

  Part 2 - evaluate(): eval + no_grad, then train mode restored
    evaluate() loss on the untrained net: 0.070625
    prediction under no_grad requires_grad: False
    grads after evaluate(): all None = True
    grads after one train backward(): all populated = True

  Part 3 - the full loop on one device, done right
    device selected: mps
    model and data co-located: True
    initial eval loss: 0.070625
    training flag restored after evaluate(): True
    final eval loss: 0.000005
    predictions 0.0032, 1.0000, 0.9999, -0.0029  (correct: 4/4)
"""
import torch
import torch.nn as nn

# Day 21's hand-set weights, reused for every net below: no RNG in
# any printed number, and every result is comparable across days.
W1 = [[1.0, -1.0], [-1.0, 1.0], [0.5, 0.5], [-0.5, 0.8]]
B1 = [0.0, 0.0, -0.5, 0.0]
W2 = [[1.0, 1.0, 1.0, -0.5]]
B2 = [0.0]
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
fmt = lambda t: ", ".join(f"{v:.4f}" for v in t.flatten().tolist())


def load_hand_weights(model):
    with torch.no_grad():
        model.fc1.weight.copy_(torch.tensor(W1))
        model.fc1.bias.copy_(torch.tensor(B1))
        model.fc2.weight.copy_(torch.tensor(W2))
        model.fc2.bias.copy_(torch.tensor(B2))
    return model


class XORNet(nn.Module):
    """Day 24's class, unchanged."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class DropoutNet(nn.Module):
    """XORNet + one dropout layer, to make the mode flag visible."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        return self.fc2(self.drop(torch.relu(self.fc1(x))))


def evaluate(model, Xb, yb):
    """Measure without touching training: eval + no_grad, then
    put the mode flag back exactly where you found it."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        pred = model(Xb)
        loss = ((pred - yb) ** 2).mean().item()
        correct = int(((pred > 0.5) == (yb > 0.5)).sum())
    if was_training:
        model.train()
    return loss, correct


# ---------- Part 1: train vs eval mode ----------
print("Part 1 - train vs eval: dropout is a mode, not a layer you remove")
dnet = load_hand_weights(DropoutNet())
dnet.eval()                       # dropout becomes the identity
with torch.no_grad():
    p1, p2 = dnet(X), dnet(X)
print(f"  eval preds: {fmt(p1)}")
print(f"  two eval passes differ by: {(p1 - p2).abs().max().item():.6f}")
still_taped = dnet(X)             # eval() alone, no no_grad wrapper
print(f"  eval pass (no no_grad) requires_grad: {still_taped.requires_grad}")
dnet.train()                      # masks are redrawn every pass now
torch.manual_seed(25)
seen = {tuple(dnet(X).flatten().round(decimals=4).tolist())
        for _ in range(16)}
print(f"  train-mode passes vary across calls: {len(seen) > 1}")

# ---------- Part 2: the evaluate() helper ----------
print("\nPart 2 - evaluate(): eval + no_grad, then train mode restored")
dnet.train()                      # pretend we are mid-training
loss0, _ = evaluate(dnet, X, y)
print(f"  evaluate() loss on the untrained net: {loss0:.6f}")
with torch.no_grad():
    png = dnet(X)
print(f"  prediction under no_grad requires_grad: {png.requires_grad}")
print(f"  grads after evaluate(): all None = "
      f"{all(p.grad is None for p in dnet.parameters())}")
loss_t = ((dnet(X) - y) ** 2).mean()   # dnet is back in train mode
loss_t.backward()
print(f"  grads after one train backward(): all populated = "
      f"{all(p.grad is not None for p in dnet.parameters())}")

# ---------- Part 3: the full loop, done right ----------
print("\nPart 3 - the full loop on one device, done right")
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cpu")
model = load_hand_weights(XORNet()).to(device)   # the whole Module
Xd, yd = X.to(device), y.to(device)              # ... and the data
print(f"  device selected: {device}")
print(f"  model and data co-located: "
      f"{all(p.device == Xd.device for p in model.parameters())}")
opt = torch.optim.SGD(model.parameters(), lr=0.5)
model.train()
init_loss, _ = evaluate(model, Xd, yd)
print(f"  initial eval loss: {init_loss:.6f}")
print(f"  training flag restored after evaluate(): {model.training}")
for _ in range(100):
    model.train()                 # cheap insurance, every epoch
    opt.zero_grad()
    loss = ((model(Xd) - yd) ** 2).mean()
    loss.backward()
    opt.step()
final_loss, correct = evaluate(model, Xd, yd)
with torch.no_grad():
    pred = model(Xd)
print(f"  final eval loss: {final_loss:.6f}")
print(f"  predictions {fmt(pred)}  (correct: {correct}/4)")
