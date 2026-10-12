"""Day 16 - Tiny Classifier on Real Data.

Assemble every piece from Days 2-15 into one working machine: a neural net
that reads handwritten digits (sklearn's digits dataset: 1,797 8x8 images,
64 pixel features, 10 digit classes).

The cast, by day:
  Day  2  tensors hold the data (1797, 64)
  Day  8  80/20 train/val split keeps the score honest
  Day 12  Linear layers = stacked dot products (64 -> 32 -> 10)
  Day 14  softmax + cross-entropy scores 10 logits against the truth
  Day 4/6/7  autograd + SGD + the training-loop ritual
  Day  5  the loss tells the model how wrong it is

One loan from the future: ReLU (max(0, x)) bends the line so stacked layers
can learn curves instead of just planes. Full treatment lands Day 22; tonight
it is one line.

Runs on the ~/ai-lab venv: torch CPU or MPS, no paid APIs. sklearn is free
(pip install scikit-learn).

Expected output (reference numbers -- your MacBook may differ in the last
digits since its RNG stream differs, but the story is the same):
  device: mps
  dataset: 1797 samples, 64 features (8x8 pixels), 10 classes
  train: 1437  val: 360
  epoch   1  train loss 0.8442  val acc 0.6889
  epoch   5  train loss 0.1742  val acc 0.9306
  epoch  10  train loss 0.1188  val acc 0.9250
  epoch  15  train loss 0.0915  val acc 0.9444
  epoch  20  train loss 0.0703  val acc 0.9556
  sample predictions (true -> pred): 6 -> 6, 8 -> 8, 3 -> 3, 1 -> 1, 8 -> 8, 9 -> 9
"""
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(16)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)

# 1. REAL DATA - 1,797 handwritten digits as 8x8 pixel images
X, y = load_digits(return_X_y=True)
print(f"dataset: {X.shape[0]} samples, {X.shape[1]} features (8x8 pixels), "
      f"{len(set(y.tolist()))} classes")
X = X / 16.0                                    # pixels 0..16 -> 0..1
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=16)
print(f"train: {len(Xtr)}  val: {len(Xva)}")
Xtr = torch.tensor(Xtr, dtype=torch.float32).to(device)
Xva = torch.tensor(Xva, dtype=torch.float32).to(device)
ytr = torch.tensor(ytr, dtype=torch.long).to(device)
yva = torch.tensor(yva, dtype=torch.long).to(device)

# 2. THE MACHINE - everything from Days 2-15, assembled
model = nn.Sequential(
    nn.Linear(64, 32),   # Day 12: 32 stacked dot products
    nn.ReLU(),           # one-line loan from Day 22: bends the line
    nn.Linear(32, 10),   # 10 logits, one per digit
).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.5)   # Day 6
loss_fn = nn.CrossEntropyLoss()                     # Day 14

# 3. THE LOOP - Day 7's ritual, mini-batch SGD (64 shuffled digits at a time)
EPOCHS, B = 20, 64
for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(Xtr), device=device)
    for i in range(0, len(Xtr), B):
        idx = perm[i:i + B]
        opt.zero_grad()                                 # Day 7: no free accumulation
        loss = loss_fn(model(Xtr[idx]), ytr[idx])       # forward -> loss
        loss.backward()                                 # backward (Day 4)
        opt.step()                                      # step (Day 6)
    if epoch in (1, 5, 10, 15, 20):
        model.eval()
        with torch.no_grad():
            tr_loss = loss_fn(model(Xtr), ytr).item()
            vacc = (model(Xva).argmax(1) == yva).float().mean().item()
        print(f"epoch {epoch:3d}  train loss {tr_loss:.4f}  val acc {vacc:.4f}")

# 4. RECEIPTS - predictions on digits it never trained on
model.eval()
with torch.no_grad():
    pred = model(Xva).argmax(1)
print("\nsample predictions (true -> pred):")
for i in range(6):
    print(f"  {yva[i].item()} -> {pred[i].item()}")
acc = (pred == yva).float().mean().item()
print(f"\nWIN: a neural net reading handwriting - val accuracy {acc:.4f}")
