"""Day 67 - Experiment Tracking: The 10-Minute Setup.

Day 31 gave you loss curves as observability - for ONE run, live,
on your screen. Tonight is what happens on run three: three
terminal scrollbacks, three sets of numbers, and no way to say
which config produced the good val loss. Print statements are
logs you throw away. Tonight the logs get a store.

ONE idea: an experiment tracker is just a structured log -
(run id, params, metrics-over-time, artifacts) in an append-only
store you can QUERY. The discipline that makes it worth anything:
log the params, the final metrics, and a hash of the config for
EVERY run - the diverged ones too - so results are comparable
and reproducible, not remembered.

  Part 1  A tiny Tracker in pure stdlib: start_run writes the
          run's config.json artifact and appends a start event;
          log_metric appends one JSON line per measurement;
          end_run writes result.json and appends the final
          numbers. compare() is just a read of that log, sorted.
  Part 2  Day 7's regression (truth w=2.5, b=-1) under the
          Day 31 trio - lr 0.001 (flatline), 0.05 (healthy),
          0.65 (diverging) - same data, same 30 epochs, only
          lr changes. Every epoch of every run is logged,
          including the run that blows up.
  Part 3  compare() prints the runs table sorted by final val
          loss and the best run's full config; then the best
          config is re-run untouched - and lands on the same
          config hash and the identical final val loss.
          That is reproducibility, demonstrated not promised.

Pure Python standard library - no torch needed, no installs,
no W&B account, no MLflow server, no downloads. This file
layout IS what the real tools keep: MLflow's local store is
a directory of run folders with params/metrics files, and a
W&B run object carries the same four fields. Learn the shape
here and the tools are just bigger versions of it.

The demo writes ./runs/ next to where you run it (and resets
that folder first, so your output matches this page every
time). Run it in ~/ai-lab, then open runs/ and look around.

Expected output:

  Part 1 - the tracker: (run id, params, metrics, artifacts) in an append-only store
    store = runs/index.jsonl (events) + runs/<run-id>/ (config.json, result.json)
  Part 2 - Day 7's line under Day 31's trio (same data, same epochs, only lr moves)
    run-001  lr=0.001  final train 9.4124  val 8.6657  w=0.2383 b=-0.0583  hash 37c2d268  (flatline)
    run-002  lr=0.05   final train 0.0020  val 0.0022  w=2.4895 b=-0.9576  hash 7d8d997b  (healthy)
    run-003  lr=0.65   final train 108276.9826  val 85268.1238  w=-252.3847 b=-1.0000  hash bb50b172  (diverging)
    the diverged run is IN the table - a run you don't log is a result you can't rule out
  Part 3 - compare(): the question scrollback can't answer
    run           lr epochs  final train     final val  hash
    run-002     0.05     30       0.0020        0.0022  7d8d997b
    run-001    0.001     30       9.4124        8.6657  37c2d268
    run-003     0.65     30  108276.9826    85268.1238  bb50b172
    best run: run-002 - full config from its artifact:
      {"data": "day7-line", "epochs": 30, "lr": 0.05, "seed": 42}  (config_hash 7d8d997b)
    re-run as run-004: hash matches True, final val identical True (0.0022)
    store on disk: runs/index.jsonl = 194 events, 4 run folders each holding config.json + result.json
    verdict: log params + final metrics + config hash for EVERY run, and 'which config was that?' is a query, not an archaeology dig
"""
import hashlib
import json
import shutil
from pathlib import Path

# ---------- the data: Day 7's line, truth w=2.5, b=-1 ----------
X_TRAIN = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
Y_TRAIN = [2.5 * x - 1.0 for x in X_TRAIN]
X_VAL = [-1.75, -0.75, 0.25, 1.25]          # never trained on
Y_VAL = [2.5 * x - 1.0 for x in X_VAL]


def mse(w, b, xs, ys):
    return sum((w * x + b - y) ** 2 for x, y in zip(xs, ys)) / len(xs)


def train_line(lr, epochs):
    """Full-batch gradient descent (Day 6's hand math).
    Returns the (train_loss, val_loss) history, epoch 0..epochs."""
    w = b = 0.0
    history = [(mse(w, b, X_TRAIN, Y_TRAIN), mse(w, b, X_VAL, Y_VAL))]
    for _ in range(epochs):
        n = len(X_TRAIN)
        dw = sum(2 * (w * x + b - y) * x
                 for x, y in zip(X_TRAIN, Y_TRAIN)) / n
        db = sum(2 * (w * x + b - y)
                 for x, y in zip(X_TRAIN, Y_TRAIN)) / n
        w -= lr * dw
        b -= lr * db
        history.append((mse(w, b, X_TRAIN, Y_TRAIN),
                        mse(w, b, X_VAL, Y_VAL)))
    return w, b, history


# ---------- Part 1: the tracker (the whole tool) ----------
def config_hash(params):
    blob = json.dumps(params, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


class Run:
    def __init__(self, tracker, run_id, params):
        self.tracker = tracker
        self.run_id = run_id
        self.params = params
        self.hash = config_hash(params)
        self.final = {}

    def log_metric(self, name, value, step):
        self.tracker.append({"event": "metric", "run": self.run_id,
                             "name": name, "value": value,
                             "step": step})

    def end_run(self, final_metrics):
        self.final = final_metrics
        out = {"run_id": self.run_id, "config_hash": self.hash,
               "params": self.params, "final": final_metrics}
        path = self.tracker.root / self.run_id / "result.json"
        path.write_text(json.dumps(out, indent=2))
        self.tracker.append({"event": "end", "run": self.run_id,
                             "config_hash": self.hash,
                             "final": final_metrics})


class Tracker:
    """Append-only JSONL event log + one folder per run."""

    def __init__(self, root="runs"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.root / "index.jsonl"

    def append(self, event):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def start_run(self, name, params):
        done = 0
        if self.log_path.exists():
            for line in open(self.log_path):
                if json.loads(line)["event"] == "start":
                    done += 1
        run_id = f"run-{done + 1:03d}"
        run = Run(self, run_id, params)
        folder = self.root / run_id
        folder.mkdir(exist_ok=True)
        cfg = {"run_id": run_id, "name": name, "params": params,
               "config_hash": run.hash}
        (folder / "config.json").write_text(json.dumps(cfg, indent=2))
        self.append({"event": "start", "run": run_id, "name": name,
                     "params": params, "config_hash": run.hash})
        return run


def compare(root="runs"):
    """The query: read the log, one row per finished run,
    sorted by final val loss. The scrollback can't do this."""
    rows = {}
    for line in open(Path(root) / "index.jsonl"):
        ev = json.loads(line)
        if ev["event"] == "start":
            rows[ev["run"]] = {"params": ev["params"],
                               "hash": ev["config_hash"]}
        elif ev["event"] == "end":
            rows[ev["run"]]["final"] = ev["final"]
    done = {r: v for r, v in rows.items() if "final" in v}
    ranked = sorted(done.items(),
                    key=lambda kv: kv[1]["final"]["val_loss"])
    print(f"  {'run':<9} {'lr':>6} {'epochs':>6} "
          f"{'final train':>12} {'final val':>13}  hash")
    for run_id, v in ranked:
        f = v["final"]
        print(f"  {run_id:<9} {v['params']['lr']:>6} "
              f"{v['params']['epochs']:>6} "
              f"{f['train_loss']:>12.4f} {f['val_loss']:>13.4f}"
              f"  {v['hash']}")
    return ranked[0]


# ---------- Part 2: three runs, one variable changed ----------
print("Part 1 - the tracker: (run id, params, metrics, artifacts)"
      " in an append-only store")
print("  store = runs/index.jsonl (events) + runs/<run-id>/"
      " (config.json, result.json)")

print("Part 2 - Day 7's line under Day 31's trio "
      "(same data, same epochs, only lr moves)")
ROOT = Path("runs")
if ROOT.exists():                       # demo resets its own store
    shutil.rmtree(ROOT)
tracker = Tracker(ROOT)
for lr, note in ((0.001, "flatline"), (0.05, "healthy"),
                 (0.65, "diverging")):
    params = {"lr": lr, "epochs": 30, "seed": 42,
              "data": "day7-line"}
    run = tracker.start_run(f"lr-sweep-{note}", params)
    w, b, history = train_line(lr, params["epochs"])
    for step, (tr, va) in enumerate(history):
        run.log_metric("train_loss", tr, step)
        run.log_metric("val_loss", va, step)
    run.end_run({"train_loss": history[-1][0],
                 "val_loss": history[-1][1], "w": w, "b": b})
    print(f"  {run.run_id}  lr={lr:<6} final train "
          f"{history[-1][0]:.4f}  val {history[-1][1]:.4f}"
          f"  w={w:.4f} b={b:.4f}  hash {run.hash}  ({note})")
print("  the diverged run is IN the table - a run you don't log"
      " is a result you can't rule out")

# ---------- Part 3: query it, then prove reproducibility ----------
print("Part 3 - compare(): the question scrollback can't answer")
best_id, best = compare(ROOT)
cfg = json.loads((ROOT / best_id / "config.json").read_text())
print(f"  best run: {best_id} - full config from its artifact:")
print(f"    {json.dumps(cfg['params'], sort_keys=True)}"
      f"  (config_hash {cfg['config_hash']})")
again = tracker.start_run("lr-sweep-healthy-rerun", cfg["params"])
w, b, history = train_line(cfg["params"]["lr"],
                           cfg["params"]["epochs"])
again.end_run({"train_loss": history[-1][0],
               "val_loss": history[-1][1], "w": w, "b": b})
print(f"  re-run as {again.run_id}: hash matches "
      f"{again.hash == best['hash']}, final val identical "
      f"{history[-1][1] == best['final']['val_loss']}"
      f" ({history[-1][1]:.4f})")
n_events = sum(1 for _ in open(ROOT / "index.jsonl"))
print(f"  store on disk: runs/index.jsonl = {n_events} events, "
      f"4 run folders each holding config.json + result.json")
print("  verdict: log params + final metrics + config hash for"
      " EVERY run, and 'which config was that?' is a query,"
      " not an archaeology dig")
