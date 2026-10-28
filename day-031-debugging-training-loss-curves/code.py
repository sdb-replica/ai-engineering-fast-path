"""Day 31 - Debugging Training: Loss Curves as Observability.

A single loss value tells you almost nothing. The SHAPE of the
loss over time is the signal -- the loss curve is the primary
dashboard of a training run, the way latency and error-rate
graphs are the dashboard of a service. Tonight five shapes,
each with a different likely cause and a different first fix:

  HEALTHY    steady fall, flattening near a floor. Train and
             val fall together. Leave it alone.
  FLATLINE   barely moves (Part 1: down 17.6% in 30 epochs).
             Almost no signal is reaching the weights: stride
             far too small, gradients vanishing (Days 22/28),
             dead units, or a loss that is not connected to
             the weights at all (Day 20: round()/detach()
             gave gradients of exactly zero).
  DIVERGING  rises, faster and faster, on its way to NaN.
             Stride too big (Day 30), exploding gradients, or
             unscaled data. Kill the run -- waiting never
             fixes a divergence.
  SPIKE      one-step jump, then a return to trend (Part 2:
             0.1155 -> 236.4101 -> 1.7963). Suspect ONE bad
             batch -- a corrupt example or a data glitch --
             before you suspect the model. Recovery is real
             but not free: the bad step still moved the
             weights, and the healed run ends worse.
  OVERFIT    train keeps falling while val bottoms and rises
  FORK       (Part 3). The model has started memorizing
             (Days 8/17). The cure is stopping at the val
             minimum -- early stopping -- not more epochs.

Two reading habits matter as much as the shapes. Judge the
TREND over a window, never one point to the next: a small
batch makes the curve jagged even when it is healthy
(Day 30's gradient noise). And always read train and val
together -- the fork is invisible in the train curve alone.

The diagnose() function below is the lesson in code: crude
threshold rules that turn a loss history into a verdict.
Real debugging is the same shape -- hypothesis from the
curve, ONE intervention, shortest run that can confirm it.

  Part 1  Three runs on Day 7's line (truth w=2.5, b=-1),
          same init w=b=0: lr 0.05 healthy, lr 0.001 flat,
          lr 0.65 diverging to 227.3 inside 10 epochs.
  Part 2  The healthy run with one corrupted target
          (50.0, a bad print) fed at epoch 15 only: loss
          spikes 2046x for one epoch, then recovers to
          0.0333 by epoch 30 vs the healthy 0.0020.
  Part 3  A degree-9 polynomial (features x..x^9) on 10
          noisy points of the true line y=x, validated on
          41 clean points: train falls the whole 400
          epochs, val bottoms at epoch 13 (0.0083) and
          ends at 0.0265 -- 3.2x worse.

Runs on the ~/ai-lab venv: torch only (free), CPU or MPS, no
CUDA, no paid APIs, no downloads. No randomness -- every run
starts from a hand-set init, so your numbers should match
the expected output. (The diverging losses are printed to
1 decimal: a diverging run amplifies last-bit float
differences, so its trailing digits are not stable.)

Expected output:

  device: mps
  Part 1 - three runs, same line, same init: read the shape
    healthy   lr 0.050: loss 11.4167 -> 2.0310 (ep 5) -> 0.3933 (ep 10) -> 0.0020 (ep 30)  w 2.4895  b -0.9576  -> HEALTHY
    flatline  lr 0.001: loss 11.4167 -> 11.0548 (ep 5) -> 10.7045 (ep 10) -> 9.4124 (ep 30)  w 0.2383  b -0.0583  -> FLATLINE
    diverging lr 0.650: loss 11.4 -> 48.7 (ep 5) -> 227.3 (ep 10)  -> DIVERGING
  Part 2 - one bad batch: a single corrupted target at epoch 15
    ep 14 loss 0.1155   ep 15 loss 236.4101   ep 16 loss 1.7963   ep 20 loss 0.5137   ep 30 loss 0.0333  -> SPIKE (recovered)
    spike size: 2046x the epoch-14 loss; final w 2.5691  b -0.8407 (healthy run: w 2.4895  b -0.9576)
  Part 3 - the overfit fork: train falls, val bottoms, then rises
    ep   0: train 0.4824  val 0.3500
    ep  10: train 0.1448  val 0.0099
    ep  50: train 0.1222  val 0.0164
    ep 200: train 0.1128  val 0.0202
    ep 400: train 0.1086  val 0.0265  -> OVERFIT FORK
    best val 0.0083 at epoch 13 - stopping there beats epoch 400 by 3.2x on val
"""
import math

import torch
import torch.nn as nn

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)


def diagnose(train_losses, val_losses=None):
    """Crude shape rules: a loss history in, a verdict out."""
    first, last = train_losses[0], train_losses[-1]
    if not all(math.isfinite(v) for v in train_losses) or last > first:
        return "DIVERGING"
    jumps = [train_losses[i] / max(train_losses[i - 1], 1e-12)
             for i in range(1, len(train_losses))]
    if max(jumps) > 20 and last < 0.01 * max(train_losses):
        return "SPIKE (recovered)"
    if (val_losses is not None and last < first
            and val_losses[-1] > 1.5 * min(val_losses)):
        return "OVERFIT FORK"
    if last > 0.75 * first:
        return "FLATLINE"
    return "HEALTHY"


# ---------- Part 1: three runs, three shapes ----------
print("Part 1 - three runs, same line, same init: read the shape")
X = torch.tensor([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
                 device=device)
Y = 2.5 * X - 1.0                                 # truth: w = 2.5, b = -1


def train_line(lr, epochs, spike_epoch=None):
    w = torch.tensor(0.0, device=device, requires_grad=True)
    b = torch.tensor(0.0, device=device, requires_grad=True)
    opt = torch.optim.SGD([w, b], lr=lr)
    losses = []
    for epoch in range(epochs + 1):
        targets = Y
        if epoch == spike_epoch:                  # one bad print, once
            targets = Y.clone()
            targets[8] = 50.0
        loss = ((w * X + b - targets) ** 2).mean()
        losses.append(loss.item())
        if epoch == epochs:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
    return losses, w.item(), b.item()


healthy, hw, hb = train_line(0.05, 30)
print(f"  healthy   lr 0.050: loss {healthy[0]:.4f} -> {healthy[5]:.4f} (ep 5)"
      f" -> {healthy[10]:.4f} (ep 10) -> {healthy[30]:.4f} (ep 30)"
      f"  w {hw:.4f}  b {hb:.4f}  -> {diagnose(healthy)}")
flat, fw, fb = train_line(0.001, 30)
print(f"  flatline  lr 0.001: loss {flat[0]:.4f} -> {flat[5]:.4f} (ep 5)"
      f" -> {flat[10]:.4f} (ep 10) -> {flat[30]:.4f} (ep 30)"
      f"  w {fw:.4f}  b {fb:.4f}  -> {diagnose(flat)}")
div, _, _ = train_line(0.65, 10)
print(f"  diverging lr 0.650: loss {div[0]:.1f} -> {div[5]:.1f} (ep 5)"
      f" -> {div[10]:.1f} (ep 10)  -> {diagnose(div)}")

# ---------- Part 2: a spike is a data story first ----------
print("Part 2 - one bad batch: a single corrupted target at epoch 15")
spiked, sw, sb = train_line(0.05, 30, spike_epoch=15)
print(f"  ep 14 loss {spiked[14]:.4f}   ep 15 loss {spiked[15]:.4f}"
      f"   ep 16 loss {spiked[16]:.4f}   ep 20 loss {spiked[20]:.4f}"
      f"   ep 30 loss {spiked[30]:.4f}  -> {diagnose(spiked)}")
print(f"  spike size: {spiked[15] / spiked[14]:.0f}x the epoch-14 loss;"
      f" final w {sw:.4f}  b {sb:.4f} (healthy run: w {hw:.4f}  b {hb:.4f})")

# ---------- Part 3: the overfit fork, train vs val ----------
print("Part 3 - the overfit fork: train falls, val bottoms, then rises")
tx = torch.linspace(-1.0, 1.0, 10, device=device)
noise = torch.tensor([0.50, -0.40, 0.45, -0.35, 0.30,
                      -0.50, 0.40, -0.30, 0.48, -0.42], device=device)
ty = tx + noise                                   # true line y = x, noised
vx = torch.linspace(-1.0, 1.0, 41, device=device)
vy = vx.clone()                                   # clean validation line


def poly_features(x):
    return torch.stack([x ** k for k in range(1, 10)], dim=1)  # x .. x^9


model = nn.Linear(9, 1, bias=False).to(device)
with torch.no_grad():
    model.weight.zero_()                          # hand-set init, as always
opt = torch.optim.SGD(model.parameters(), lr=0.5)
train_hist, val_hist, rows = [], [], []
for epoch in range(401):
    train_loss = ((model(poly_features(tx)).squeeze(1) - ty) ** 2).mean()
    with torch.no_grad():                         # Day 25: eval discipline
        val_loss = ((model(poly_features(vx)).squeeze(1) - vy) ** 2).mean()
    train_hist.append(train_loss.item())
    val_hist.append(val_loss.item())
    if epoch in (0, 10, 50, 200, 400):
        rows.append((epoch, train_loss.item(), val_loss.item()))
    if epoch == 400:
        break
    opt.zero_grad()
    train_loss.backward()
    opt.step()
for epoch, tl, vl in rows:
    suffix = f"  -> {diagnose(train_hist, val_hist)}" if epoch == 400 else ""
    print(f"  ep {epoch:3d}: train {tl:.4f}  val {vl:.4f}{suffix}")
best_epoch = min(range(len(val_hist)), key=lambda i: val_hist[i])
print(f"  best val {val_hist[best_epoch]:.4f} at epoch {best_epoch}"
      f" - stopping there beats epoch 400 by"
      f" {val_hist[-1] / val_hist[best_epoch]:.1f}x on val")
