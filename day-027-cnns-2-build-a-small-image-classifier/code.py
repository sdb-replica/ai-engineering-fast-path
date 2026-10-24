"""Day 27 - CNNs 2: Build a Small Image Classifier.

Day 26 slid ONE hand-set kernel over a toy grid. Tonight the kernels
are learned, there are stacks of them, and the stack reads real
handwriting: the same sklearn digits from Day 16 (1,797 images) --
but this time kept AS images, (N, 1, 8, 8), never flattened first.

The recipe is two halves:

  feature extractor  Learns WHAT to look at. Conv block 1: eight
      3x3 kernels (Day 26) + ReLU (Day 22), then max pooling --
      keep only the strongest response in each 2x2 patch, halving
      the map. Conv block 2: sixteen kernels over those 8 maps,
      ReLU, pool again. Early kernels learn edges; later kernels
      learn combinations of edges -- shapes. Each pool trades
      exact position for a smaller, sturdier map.
  head  Decides the answer. Flatten (16, 2, 2) to 64 numbers and
      one Linear layer to 10 logits -- Day 16's classifier,
      unchanged. Only the extractor is new.

Shape is the contract at every joint, and Part 1 prints it:
  (B,1,8,8) -> conv1 (B,8,8,8) -> pool (B,8,4,4)
            -> conv2 (B,16,4,4) -> pool (B,16,2,2)
            -> flatten 64 -> linear (B,10)
Padding 1 keeps a 3x3 conv's output the same size as its input
(Day 26 formula: (8 + 2 - 3)/1 + 1 = 8), so only pooling shrinks
the map -- one shrinker to reason about, not two.

The training is Day 25's loop done right -- model on the device,
batches moved to it, eval under eval() + no_grad -- fed by Day 19's
DataLoader. The optimizer is Adam (Day 18) at lr 0.01: conv nets
have features at very different scales, Adam's home turf.

Parameter bill: conv1 80 + conv2 1,168 + head 650 = 1,898 total,
FEWER than Day 16's dense model (2,410) -- weight sharing (Day 26)
pays again. Watch whether fewer params + image structure also
beats Day 16's val accuracy of 0.9556.

  Part 1  The architecture on paper: stage-by-stage shapes from a
          dummy batch, and the per-layer parameter count.
  Part 2  Train 15 epochs, batch 64, Adam. Print the loss/acc arc
          at epochs 1, 5, 10, 15.
  Part 3  Receipts: sample predictions on held-out digits and the
          final val accuracy, next to Day 16's number.

Runs on the ~/ai-lab venv: torch + scikit-learn (free), CPU or MPS,
no CUDA, no paid APIs, no downloads -- digits ships with sklearn.

Expected output (reference numbers -- your MacBook's RNG stream
differs from the reference run, so last digits will vary; the arc
and the win are the same):

  device: mps
  Part 1 - the architecture: shapes are the contract
    input            (2, 1, 8, 8)
    after conv1+ReLU (2, 8, 8, 8)
    after pool1      (2, 8, 4, 4)
    after conv2+ReLU (2, 16, 4, 4)
    after pool2      (2, 16, 2, 2)
    after flatten    (2, 64)
    logits           (2, 10)
    params: conv1 80  conv2 1168  head 650  total 1898
    Day 16 dense model params: 2410
  Part 2 - train: Day 25's loop, Day 19's loader, Adam
    dataset: 1797 samples, images (1, 8, 8), 10 classes
    train: 1437  val: 360
    epoch   1  train loss 0.8715  val acc 0.7722
    epoch   5  train loss 0.1058  val acc 0.9417
    epoch  10  train loss 0.0468  val acc 0.9722
    epoch  15  train loss 0.0150  val acc 0.9833
  Part 3 - receipts
    sample predictions (true -> pred):
      6 -> 6
      8 -> 8
      3 -> 3
      1 -> 1
      8 -> 8
      9 -> 9
    WIN: CNN val accuracy 0.9833 vs Day 16 MLP 0.9556
"""
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(27)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)


class SmallCNN(nn.Module):
    """Day 24's class discipline, Day 26's layers.

    Extractor: conv -> ReLU -> pool, twice. Head: flatten -> linear.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Linear(16 * 2 * 2, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))   # (B, 8, 4, 4)
        x = self.pool(torch.relu(self.conv2(x)))   # (B, 16, 2, 2)
        x = torch.flatten(x, 1)                    # (B, 64)
        return self.head(x)                        # (B, 10) logits


model = SmallCNN().to(device)

# ---------- Part 1: the architecture, shapes are the contract ----------
print("Part 1 - the architecture: shapes are the contract")
dummy = torch.zeros(2, 1, 8, 8, device=device)
with torch.no_grad():
    a = torch.relu(model.conv1(dummy))
    b = model.pool(a)
    c = torch.relu(model.conv2(b))
    d = model.pool(c)
    flat = torch.flatten(d, 1)
    logits = model.head(flat)
print(f"  input            {tuple(dummy.shape)}")
print(f"  after conv1+ReLU {tuple(a.shape)}")
print(f"  after pool1      {tuple(b.shape)}")
print(f"  after conv2+ReLU {tuple(c.shape)}")
print(f"  after pool2      {tuple(d.shape)}")
print(f"  after flatten    {tuple(flat.shape)}")
print(f"  logits           {tuple(logits.shape)}")
n1 = sum(p.numel() for p in model.conv1.parameters())
n2 = sum(p.numel() for p in model.conv2.parameters())
n3 = sum(p.numel() for p in model.head.parameters())
print(f"  params: conv1 {n1}  conv2 {n2}  head {n3}  total {n1 + n2 + n3}")
print("  Day 16 dense model params: 2410")

# ---------- Part 2: train ----------
print("Part 2 - train: Day 25's loop, Day 19's loader, Adam")
X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype("float32").reshape(-1, 1, 8, 8)  # keep the image!
print(f"  dataset: {X.shape[0]} samples, images (1, 8, 8), 10 classes")
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=16)
print(f"  train: {len(Xtr)}  val: {len(Xva)}")
Xtr = torch.tensor(Xtr).to(device)
Xva = torch.tensor(Xva).to(device)
ytr = torch.tensor(ytr, dtype=torch.long).to(device)
yva = torch.tensor(yva, dtype=torch.long).to(device)
loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
opt = torch.optim.Adam(model.parameters(), lr=0.01)   # Day 18
loss_fn = nn.CrossEntropyLoss()                       # Day 14


def evaluate():
    """Day 25's helper: eval mode + no_grad, restore training after."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        acc = (model(Xva).argmax(1) == yva).float().mean().item()
        tr_loss = loss_fn(model(Xtr), ytr).item()
    if was_training:
        model.train()
    return tr_loss, acc


for epoch in range(1, 16):
    model.train()
    for xb, yb in loader:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)   # forward -> loss
        loss.backward()                 # backward (Day 4)
        opt.step()                      # step (Day 6)
    if epoch in (1, 5, 10, 15):
        tr_loss, vacc = evaluate()
        print(f"  epoch {epoch:3d}  train loss {tr_loss:.4f}  val acc {vacc:.4f}")

# ---------- Part 3: receipts ----------
print("Part 3 - receipts")
model.eval()
with torch.no_grad():
    pred = model(Xva).argmax(1)
print("  sample predictions (true -> pred):")
for i in range(6):
    print(f"    {yva[i].item()} -> {pred[i].item()}")
acc = (pred == yva).float().mean().item()
print(f"  WIN: CNN val accuracy {acc:.4f} vs Day 16 MLP 0.9556")
