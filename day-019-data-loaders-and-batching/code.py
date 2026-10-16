"""Day 19 - Data Loaders & Batching.

Until today every training loop sliced the data by hand. The DataLoader
is the pipeline that replaces the hand-slicing: a Dataset stores the
examples, a Sampler picks the order for this epoch, and the loader
hands the training loop one stacked batch tensor at a time.

  Part 1  Anatomy. 250 examples, batch_size=64: 4 batches, sizes
          [64, 64, 64, 58] - the tail batch is short. drop_last=True
          throws the tail away: 3 batches, 192 seen, 58 skipped.
  Part 2  Order matters. A dataset sorted by label (128 zeros, then
          128 ones) with shuffle=False feeds the model pure single-class
          batches - half the epoch pushes one way, half the other.
          shuffle=True deals a fresh permutation every epoch: same 256
          examples, each exactly once, in an order that changes.
  Part 3  The Day 7 loop, loader-fed. 256 examples, batch 32 -> 8
          steps per epoch, 240 steps over 30 epochs. Loss walks
          3.8638 -> 0.0097 (the noise floor); w, b land on the truth.

Runs on the ~/ai-lab venv: torch only, CPU on purpose (tiny data,
seeded, deterministic). No CUDA, no paid APIs. On your Mac the same
script also works unchanged on MPS - the loader doesn't care which
device the tensors move to afterwards.

Expected output:

  Part 1 - anatomy: 250 examples, batch_size=64
  dataset: 250 examples
  drop_last=False -> 4 batches, sizes [64, 64, 64, 58]
  first batch: X shape (64, 1), rows 0..63
  drop_last=True  -> 3 batches, 192 examples seen, 58 skipped per epoch

  Part 2 - sorted-by-label data, batch_size=32
  shuffle=False: batch 1 classes {0: 32, 1: 0}, batch 5 classes {0: 0, 1: 32}
  shuffle=True : every epoch covers all 256 examples exactly once: True
  shuffle=True : epoch 1 order == epoch 2 order: False (fresh deal each epoch)

  Part 3 - Day 7 loop on loader batches (256 examples, batch 32, SGD lr=0.05)
  steps per epoch: 8, total steps over 30 epochs: 240
    epoch   1  loss 3.8638
    epoch   2  loss 0.7389
    epoch   3  loss 0.1480
    epoch   5  loss 0.0147
    epoch  10  loss 0.0097
    epoch  30  loss 0.0097
  learned w = 2.4890, b = -1.0076  (truth: w = 2.5, b = -1, noise floor MSE ~ 0.01)
"""
import random

import torch
from torch.utils.data import DataLoader, TensorDataset

# ---------- Part 1: anatomy - dataset, batcher, the short tail ----------
# X[i] = i on purpose: you can watch exactly which rows land where.
print("Part 1 - anatomy: 250 examples, batch_size=64")
X = torch.arange(250, dtype=torch.float32).reshape(250, 1)
y = torch.arange(250) % 2
ds = TensorDataset(X, y)
print(f"dataset: {len(ds)} examples")

loader = DataLoader(ds, batch_size=64)                 # drop_last=False
sizes = [xb.shape[0] for xb, _ in loader]
print(f"drop_last=False -> {len(loader)} batches, sizes {sizes}")
xb0, _ = next(iter(loader))
print(f"first batch: X shape {tuple(xb0.shape)}, "
      f"rows {int(xb0[0, 0])}..{int(xb0[-1, 0])}")

loader_dl = DataLoader(ds, batch_size=64, drop_last=True)
seen = sum(xb.shape[0] for xb, _ in loader_dl)
print(f"drop_last=True  -> {len(loader_dl)} batches, {seen} examples seen, "
      f"{250 - seen} skipped per epoch")

# ---------- Part 2: order matters - the sorted-by-label trap ----------
# Labels sorted like a dataset dumped in date order: 128 down-days,
# then 128 up-days. With shuffle=False the batches are pure.
print("\nPart 2 - sorted-by-label data, batch_size=32")
labels = torch.cat([torch.zeros(128), torch.ones(128)]).long()
ds2 = TensorDataset(torch.zeros(256, 1), labels)
plain = DataLoader(ds2, batch_size=32, shuffle=False)
batches = list(plain)
c1 = torch.bincount(batches[0][1], minlength=2)
c5 = torch.bincount(batches[4][1], minlength=2)
print(f"shuffle=False: batch 1 classes {{0: {int(c1[0])}, 1: {int(c1[1])}}}, "
      f"batch 5 classes {{0: {int(c5[0])}, 1: {int(c5[1])}}}")

g = torch.Generator().manual_seed(19)
idx = torch.arange(256).reshape(256, 1).float()
ds_idx = TensorDataset(idx, labels)
shuffled_idx = DataLoader(ds_idx, batch_size=32, shuffle=True, generator=g)
e1 = torch.cat([xb[:, 0] for xb, _ in shuffled_idx]).long()
e2 = torch.cat([xb[:, 0] for xb, _ in shuffled_idx]).long()
print(f"shuffle=True : every epoch covers all 256 examples exactly once: "
      f"{bool((e1.sort().values == torch.arange(256)).all())}")
print(f"shuffle=True : epoch 1 order == epoch 2 order: "
      f"{bool((e1 == e2).all())} (fresh deal each epoch)")

# ---------- Part 3: the Day 7 loop, fed by a loader ----------
# Same ritual as Day 7 - forward, loss, backward, step - but the inner
# loop is now `for xb, yb in loader`. Data from the pure-Python RNG
# (seed 19) so these numbers reproduce exactly.
print("\nPart 3 - Day 7 loop on loader batches (256 examples, batch 32, "
      "SGD lr=0.05)")
random.seed(19)
n = 256
xs = [random.gauss(0, 1) for _ in range(n)]
noise = [random.gauss(0, 0.1) for _ in range(n)]
X3 = torch.tensor(xs, dtype=torch.float32).reshape(n, 1)
y3 = torch.tensor([2.5 * a - 1 + e for a, e in zip(xs, noise)],
                  dtype=torch.float32)
train_loader = DataLoader(TensorDataset(X3, y3), batch_size=32,
                          shuffle=False)
w = torch.zeros((), requires_grad=True)
b = torch.zeros((), requires_grad=True)
opt = torch.optim.SGD([w, b], lr=0.05)
steps = 0
for epoch in range(1, 31):
    epoch_loss = 0.0
    for xb, yb in train_loader:
        opt.zero_grad()                                # Day 7 ritual
        loss = ((xb[:, 0] * w + b - yb) ** 2).mean()   # forward + loss
        loss.backward()                                # backward
        opt.step()                                     # step
        epoch_loss += loss.item() * len(xb)
        steps += 1
    if epoch in (1, 2, 3, 5, 10, 30):
        print(f"  epoch {epoch:>3}  loss {epoch_loss / n:.4f}")
print(f"steps per epoch: {len(train_loader)}, "
      f"total steps over 30 epochs: {steps}")
print(f"learned w = {w.item():.4f}, b = {b.item():.4f}  "
      f"(truth: w = 2.5, b = -1, noise floor MSE ~ 0.01)")
