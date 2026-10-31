"""Day 35 - Phase B Project: The End-to-End Mini-Project.

The capstone. Day 32 built a direction predictor as an
experiment: features, a split, two nets, some numbers in a
scrollback. Tonight the same build becomes a PROJECT - one
clean, reproducible pipeline you could hand to another
engineer (or an interviewer) and get the same result:

    data -> features -> split -> train -> artifact -> report

Nothing inside the stages is new. Features are Day 32's five,
the split is Day 9's chronological rule, the model is Day 24's
nn.Module, evaluation is Day 25's discipline, the fork it
avoids is Day 31's. What is new is the ASSEMBLY:

  One CONFIG, one seed. Every choice lives in a single dict
  at the top; torch.manual_seed is called once per training
  run, before the model exists. Same config + same seed +
  same code = the same report card, proved in Part 3 by
  running the whole training a second time.

  Stage checks. Each stage's output is verified before the
  next stage trusts it: shapes, split sizes, and the no-peek
  rule re-derived by hand for row 0 (the label really is
  return[20], the features really stop at day 19).

  Early stopping. Day 31 showed val bottoming and rising -
  tonight that shape becomes a RULE: score val after every
  epoch, snapshot the best weights (Day 24's state_dict),
  stop after 25 epochs with no improvement, and restore the
  snapshot before test is touched. The best epoch here is 8;
  a fixed 200 epochs would have shipped a worse model.

  An artifact + a report card. The trained weights are saved
  to day35_direction_mlp.pt and reloaded into a fresh model
  (max diff 0.000000 - the artifact IS the model). The run
  ends with a printed report card: task, model, val/test,
  baselines, and an honest verdict - beats the majority
  baseline by 17.5 points, lands within a point of the
  one-line momentum rule. Same honest low-signal finance
  result as Day 32; now it is packaged like engineering.

Runs on the ~/ai-lab venv: torch only (free), MPS if
available else CPU, no CUDA, no paid APIs, no downloads -
the data is formula-built exactly as in Day 32. Part 1's
numbers are exact on any machine. Parts 2-3 are REFERENCE
numbers from a verified float32 numpy twin of this exact
algorithm: your MacBook's init/shuffle stream will differ
in the last digits, but the contract is the pattern - early
stop fires, the restored epoch is early, test beats the
majority baseline and sits next to the rule, and a rerun
with the same seed reproduces your card on your machine.

Expected output:

  device: mps
  config: seed 42  hidden [16]  lr 0.01  batch 32  max epochs 200  patience 25
  Part 1 - stages 1-3: data -> features -> split, with stage checks
    returns: 620 days -> 600 samples, X shape (600, 5), y shape (600,)
    features: ret_1d mom_5d mom_20d vol_5d up_frac_14d
    first row: [0.00356, 0.00410, 0.00535, 0.00174, 0.92857]  label 1 (up)
    splits (chronological, no shuffle): train 360  val 120  test 120
    train: up 0.5222  majority baseline 0.5222  mom_5d rule 0.6972
    val:   up 0.5167  majority baseline 0.5167  mom_5d rule 0.6417
    test:  up 0.5000  majority baseline 0.5000  mom_5d rule 0.6833
    normalize with train stats only: mean [0.00014, 0.00016, 0.00038, 0.00461, 0.53353]
                                     std  [0.00631, 0.00461, 0.00340, 0.00141, 0.26490]
    first row normalized: [0.542, 0.856, 1.460, -2.034, 1.491]
    stage checks passed: shapes (600, 5)/(600,), splits sum to 600, row 0 label = sign of return[20] - no peeking
  Part 2 - stage 4: train with early stopping + best-checkpoint restore
    model: TabMLP 5 -> 16 -> 1, 113 params (0.31 per train row), Adam lr 0.01
    ep   1: train loss 0.6409  train acc 0.6750  val loss 0.6448  val acc 0.6833
    ep  10: train loss 0.5441  train acc 0.7194  val loss 0.5740  val acc 0.7000
    ep  20: train loss 0.5332  train acc 0.7278  val loss 0.5775  val acc 0.6917
    ep  30: train loss 0.5211  train acc 0.7389  val loss 0.5826  val acc 0.6833
    early stop at epoch 33: no val improvement for 25 epochs
    restored best checkpoint: epoch 8, val loss 0.5716, val acc 0.7333
  Part 3 - stages 5-6: save the artifact, prove reproducibility, report card
    test (touched once): acc 0.6750  (majority 0.5000, mom_5d rule 0.6833)
    saved day35_direction_mlp.pt -> reloaded into a fresh model: max diff 0.000000, test acc 0.6750
    rerun with the same seed: best epoch 8, test acc 0.6750 - identical report card: True
    REPORT CARD - Phase B project (the deliverable)
      task:  next-day direction from 5 price features, 600 days, chronological split
      model: TabMLP 5 -> 16 -> 1 (113 params), early stop - best epoch 8 of 33 run
      val acc 0.7333  |  test acc 0.6750 (touched once)
      test baselines: majority 0.5000  |  mom_5d rule 0.6833
      verdict: beats majority by 17.5 pts, within a point of the one-line rule - the
               honest low-signal result, and the same card prints on every rerun
"""
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------- the whole run, in one place ----------
CONFIG = {
    "seed": 42,
    "hidden": [16],
    "lr": 0.01,
    "batch_size": 32,
    "max_epochs": 200,
    "patience": 25,
}
ARTIFACT = "day35_direction_mlp.pt"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)
print(f"config: seed {CONFIG['seed']}  hidden {CONFIG['hidden']}  "
      f"lr {CONFIG['lr']}  batch {CONFIG['batch_size']}  "
      f"max epochs {CONFIG['max_epochs']}  patience {CONFIG['patience']}")

# ---------- Part 1: stages 1-3 - data, features, split ----------
print("Part 1 - stages 1-3: data -> features -> split, with stage checks")
ret_list = []
for i in range(620):
    n1 = math.sin(i * 12.9898) * 43758.5453
    n1 = n1 - math.floor(n1) - 0.5               # seeded hash noise
    n2 = math.sin(i * 78.233 + 5.0) * 12543.123
    n2 = n2 - math.floor(n2) - 0.5
    wave = 0.005 * math.sin(i / 6.0) + 0.003 * math.sin(i / 17.0 + 1.3)
    ret_list.append(wave + (n1 + n2) * 0.012)    # momentum wave + noise
rets = torch.tensor(ret_list, dtype=torch.float32)

rows, labels = [], []
for t in range(19, 619):                         # 600 labeled days
    w5, w20, w14 = rets[t - 4:t + 1], rets[t - 19:t + 1], rets[t - 13:t + 1]
    rows.append(torch.stack([
        rets[t],                                 # ret_1d
        w5.mean(),                               # mom_5d
        w20.mean(),                              # mom_20d
        w5.std(),                                # vol_5d
        (w14 > 0).float().mean(),                # up_frac_14d
    ]))
    labels.append(1.0 if rets[t + 1] > 0 else 0.0)   # tomorrow's sign
X = torch.stack(rows)
y = torch.tensor(labels)
print(f"  returns: 620 days -> 600 samples, X shape {tuple(X.shape)},"
      f" y shape {tuple(y.shape)}")
print("  features: ret_1d mom_5d mom_20d vol_5d up_frac_14d")
first = ", ".join(f"{v:.5f}" for v in X[0].tolist())
print(f"  first row: [{first}]  label {int(y[0])} (up)")

Xtr, ytr = X[:360], y[:360]                      # chronological split:
Xva, yva = X[360:480], y[360:480]                # no shuffling time
Xte, yte = X[480:], y[480:]
print("  splits (chronological, no shuffle): train 360  val 120  test 120")
mom_pred = (X[:, 1] > 0).float()                 # one-line rule: mom_5d sign
base = {}
for name, Xs, ys in (("train", Xtr, ytr), ("val", Xva, yva),
                     ("test", Xte, yte)):
    up = ys.mean().item()
    maj = max(up, 1.0 - up)
    sl = {"train": slice(0, 360), "val": slice(360, 480),
          "test": slice(480, 600)}[name]
    rule = (mom_pred[sl] == ys).float().mean().item()
    base[name] = (maj, rule)
    print(f"  {name + ':':<7}up {up:.4f}  majority baseline {maj:.4f}"
          f"  mom_5d rule {rule:.4f}")

mu, sd = Xtr.mean(dim=0), Xtr.std(dim=0)         # train stats ONLY
print("  normalize with train stats only: mean ["
      + ", ".join(f"{v:.5f}" for v in mu.tolist()) + "]")
print("                                   std  ["
      + ", ".join(f"{v:.5f}" for v in sd.tolist()) + "]")
Xtr_n, Xva_n, Xte_n = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
print("  first row normalized: ["
      + ", ".join(f"{v:.3f}" for v in Xtr_n[0].tolist()) + "]")

# stage checks: verify the contracts before stage 4 trusts them
assert X.shape == (600, 5) and y.shape == (600,)
assert len(Xtr) + len(Xva) + len(Xte) == 600
row0 = torch.stack([rets[19], rets[15:20].mean(), rets[0:20].mean(),
                    rets[15:20].std(), (rets[6:20] > 0).float().mean()])
assert torch.allclose(row0, X[0])                # features stop at day 19
assert y[0].item() == float(rets[20] > 0)        # label is day 20's sign
print("  stage checks passed: shapes (600, 5)/(600,), splits sum to 600,"
      " row 0 label = sign of return[20] - no peeking")
Xtr_n, Xva_n, Xte_n = Xtr_n.to(device), Xva_n.to(device), Xte_n.to(device)
ytr_d, yva_d, yte_d = ytr.to(device), yva.to(device), yte.to(device)

# ---------- the model + Day 25 evaluation discipline ----------
crit = nn.BCEWithLogitsLoss()


class TabMLP(nn.Module):
    """Rows in, one logit out: P(up) = sigmoid(logit). (Day 32's net.)"""

    def __init__(self, hidden):
        super().__init__()
        layers, d = [], 5
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def evaluate(model, Xs, ys):
    model.eval()                                 # Day 25: eval + no_grad
    with torch.no_grad():
        logits = model(Xs)
        acc = ((logits > 0).float() == ys).float().mean()
        return crit(logits, ys).item(), acc.item()


def train_model(verbose=True):
    """One reproducible run: seed first, then build, then train.
    Early stopping (the new mechanism): score val every epoch,
    snapshot the best weights, stop after `patience` stale
    epochs, and restore the snapshot before returning."""
    torch.manual_seed(CONFIG["seed"])
    model = TabMLP(CONFIG["hidden"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    loader = DataLoader(TensorDataset(Xtr_n, ytr_d),
                        batch_size=CONFIG["batch_size"], shuffle=True)
    best_val, best_epoch, best_state, wait = float("inf"), -1, None, 0
    epoch = 0
    for epoch in range(1, CONFIG["max_epochs"] + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        tl, ta = evaluate(model, Xtr_n, ytr_d)
        vl, va = evaluate(model, Xva_n, yva_d)
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"  ep {epoch:3d}: train loss {tl:.4f}  train acc {ta:.4f}"
                  f"  val loss {vl:.4f}  val acc {va:.4f}")
        if vl < best_val:
            best_val, best_epoch, wait = vl, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= CONFIG["patience"]:
                if verbose:
                    print(f"  early stop at epoch {epoch}: no val"
                          f" improvement for {CONFIG['patience']} epochs")
                break
    model.load_state_dict(best_state)            # ship the best, not the last
    return model, best_epoch, epoch


def count_params(hidden):
    dims = [5] + hidden + [1]
    return sum((a + 1) * b for a, b in zip(dims[:-1], dims[1:]))


# ---------- Part 2: stage 4 - train by rule ----------
print("Part 2 - stage 4: train with early stopping + best-checkpoint restore")
n_params = count_params(CONFIG["hidden"])
print(f"  model: TabMLP 5 -> 16 -> 1, {n_params} params"
      f" ({n_params / 360:.2f} per train row), Adam lr {CONFIG['lr']}")
model, best_epoch, epochs_run = train_model()
assert n_params == sum(p.numel() for p in model.parameters())
val_loss, val_acc = evaluate(model, Xva_n, yva_d)
print(f"  restored best checkpoint: epoch {best_epoch},"
      f" val loss {val_loss:.4f}, val acc {val_acc:.4f}")

# ---------- Part 3: stages 5-6 - artifact + report card ----------
print("Part 3 - stages 5-6: save the artifact, prove reproducibility,"
      " report card")
_, test_acc = evaluate(model, Xte_n, yte_d)
maj_te, rule_te = base["test"]
print(f"  test (touched once): acc {test_acc:.4f}"
      f"  (majority {maj_te:.4f}, mom_5d rule {rule_te:.4f})")
torch.save(model.state_dict(), ARTIFACT)
fresh = TabMLP(CONFIG["hidden"]).to(device)
fresh.load_state_dict(torch.load(ARTIFACT, map_location=device))
diff = max((a - b).abs().max().item() for a, b in
           zip(model.state_dict().values(), fresh.state_dict().values()))
_, fresh_test = evaluate(fresh, Xte_n, yte_d)
print(f"  saved {ARTIFACT} -> reloaded into a fresh model:"
      f" max diff {diff:.6f}, test acc {fresh_test:.4f}")
model2, best_epoch2, _ = train_model(verbose=False)   # the rerun proof
_, test_acc2 = evaluate(model2, Xte_n, yte_d)
same = best_epoch2 == best_epoch and test_acc2 == test_acc
print(f"  rerun with the same seed: best epoch {best_epoch2},"
      f" test acc {test_acc2:.4f} - identical report card: {same}")
print("  REPORT CARD - Phase B project (the deliverable)")
print("    task:  next-day direction from 5 price features, 600 days,"
      " chronological split")
print(f"    model: TabMLP 5 -> 16 -> 1 ({n_params} params), early stop -"
      f" best epoch {best_epoch} of {epochs_run} run")
print(f"    val acc {val_acc:.4f}  |  test acc {test_acc:.4f} (touched once)")
print(f"    test baselines: majority {maj_te:.4f}  |  mom_5d rule {rule_te:.4f}")
print(f"    verdict: beats majority by {(test_acc - maj_te) * 100:.1f} pts,"
      " within a point of the one-line rule - the")
print("             honest low-signal result, and the same card prints"
      " on every rerun")
