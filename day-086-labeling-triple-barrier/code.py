"""Day 86 - Labeling: The Triple-Barrier Method.

Phase G, finance ML capstone. Days 83-85 gave you the
features; tonight the LABELS - the answer key the model
trains against. The label IS the question you are asking,
and the naive question is the wrong one:

    Fixed-horizon label:  "is the price higher exactly
    10 bars from now?"  It ignores the PATH (the trade
    could have hit your stop at bar 2 and recovered),
    ignores your exits, and calls +0.1% and +5% the
    same thing: "up".

López de Prado's fix (Advances in Financial Machine
Learning): give every sample THREE barriers, exactly
like the bracket order you already trade with -

    upper barrier:  profit-take at +pt%   (tonight +2%)
    lower barrier:  stop-loss  at -sl%    (tonight -1%)
    vertical barrier: a time limit        (tonight 10 bars)

The label is whichever barrier the path touches FIRST:
+1 take-profit, -1 stop, 0 time-out. Path-dependent,
asymmetric if you want it, and honest about the trades
that resolve neither way.

  Part 1  Three hand-designed paths - one per outcome -
          labeled by triple_barrier(), with asserts.
          The wanderer is the lesson: fixed-horizon
          calls it a WIN (+0.60% at bar 10) but the
          trade never reached target and timed out.
  Part 2  A 620-bar synthetic series (uptrend, then
          downtrend, then chop; seeded Generator, no
          downloads) labeled at 204 entries: the class
          balance, the mean realized return per label,
          and the teaching number - how often the
          fixed-horizon label DISAGREES.
  Part 3  Barriers are a choice: flip the asymmetry
          (+1% / -2%) and the same entries relabel -
          the dataset encodes your exit policy.

Runs on the ~/ai-lab venv: numpy only (free), no
downloads, no paid APIs. All three Part 1 paths are
hand-set offsets, every digit exact; Part 2's series
comes from np.random.default_rng(86), stated seed,
so the run reproduces bar for bar.

Expected output:

  Part 1 - three designed paths, barriers +2.0% / -1.0%, time 10 bars
    A (climbs) : first touch +2.30% at bar 6  -> label +1  (fixed-horizon: +1)
    B (slides) : first touch -1.40% at bar 5  -> label -1  (fixed-horizon: -1)
    C (wanders): no barrier in 10 bars, ends +0.60%  -> label +0  (fixed-horizon: +1)
    asserts passed: the 3 paths give the 3 distinct labels; fixed-horizon disagrees on 1 of 3
  Part 2 - 620-bar synthetic series (uptrend / downtrend / chop, seed 86)
    204 entries (every 3rd bar), same barriers
    class balance:  +1: 33 (16.2%)   -1: 99 (48.5%)   +0: 72 (35.3%)
    mean realized return by label:  +1: +2.31%   -1: -1.31%   +0: +0.55%
    fixed-horizon vs triple-barrier: disagree on 87 of 204 (42.6%)
  Part 3 - same entries, flipped bracket (+1.0% / -2.0%): labels are policy
    class balance:  +1: 85 (41.7%)   -1: 50 (24.5%)   +0: 69 (33.8%)
    92 of 204 entries changed label - same prices, same features, different question
    verdict: the label is the exit policy written as data - choose the barriers
             your bracket would actually trade, or the model learns someone else's trade
"""
import numpy as np

P0 = 100.0
PT, SL, MAX_HOLD = 0.02, 0.01, 10   # +2% take, -1% stop, 10-bar clock


def triple_barrier(prices, entry, pt=PT, sl=SL, max_hold=MAX_HOLD):
    """Label one sample: which barrier does the path touch FIRST?

    Returns (label, exit_index): +1 profit-take, -1 stop-loss,
    0 time barrier. Returns are measured from the entry price.
    """
    last = min(entry + max_hold, len(prices) - 1)
    for j in range(entry + 1, last + 1):
        r = prices[j] / prices[entry] - 1.0
        if r >= pt:
            return 1, j
        if r <= -sl:
            return -1, j
    return 0, last


def fixed_horizon(prices, entry, horizon=MAX_HOLD):
    """The old way: sign of the return exactly `horizon` bars later."""
    last = min(entry + horizon, len(prices) - 1)
    r = prices[last] / prices[entry] - 1.0
    return int(r > 0) - int(r < 0)


def path_from_offsets(offsets_pct):
    """A price path from hand-set % offsets off the entry price."""
    return P0 * (1.0 + np.asarray(offsets_pct, dtype=float) / 100.0)


# ---------- Part 1: three designed paths, one per outcome ----------
print("Part 1 - three designed paths, barriers +2.0% / -1.0%, "
      "time 10 bars")
DESIGNED = (
    ("A (climbs) ", [0, 0.4, 0.9, 0.7, 1.3, 1.8, 2.3], 1),
    ("B (slides) ", [0, -0.3, -0.6, -0.4, -0.9, -1.4], -1),
    ("C (wanders)", [0, 0.5, -0.5, 0.8, -0.2, 0.3, -0.6, 0.4,
                      0.1, -0.3, 0.6], 0),
)
n_disagree = 0
for name, offsets, want in DESIGNED:
    path = path_from_offsets(offsets)
    lab, ex = triple_barrier(path, 0)
    assert lab == want, (name, lab, want)
    fh = fixed_horizon(path, 0)
    n_disagree += int(fh != lab)
    ret = path[ex] / path[0] - 1.0
    if lab == 0:
        what = f"no barrier in {ex} bars, ends {ret:+.2%}"
    else:
        what = f"first touch {ret:+.2%} at bar {ex}"
    print(f"  {name}: {what}  -> label {lab:+d}  "
          f"(fixed-horizon: {fh:+d})")
print(f"  asserts passed: the 3 paths give the 3 distinct labels; "
      f"fixed-horizon disagrees on {n_disagree} of 3")

# ---------- Part 2: label a whole series, count the balance ----------
print("Part 2 - 620-bar synthetic series (uptrend / downtrend / "
      "chop, seed 86)")
rng = np.random.default_rng(86)
rets = np.concatenate([
    rng.normal(0.0012, 0.006, 207),    # trending up
    rng.normal(-0.0012, 0.006, 207),   # trending down
    rng.normal(0.0000, 0.004, 206),    # chop
])
prices = P0 * np.cumprod(1.0 + rets)
entries = range(0, len(prices) - MAX_HOLD, 3)
tb_labels, tb_rets, fh_labels = [], [], []
for e in entries:
    lab, ex = triple_barrier(prices, e)
    tb_labels.append(lab)
    tb_rets.append(prices[ex] / prices[e] - 1.0)
    fh_labels.append(fixed_horizon(prices, e))
tb_labels = np.array(tb_labels)
tb_rets = np.array(tb_rets)
fh_labels = np.array(fh_labels)
n = len(tb_labels)
bal = "   ".join(
    f"{s:+d}: {(tb_labels == s).sum()} ({(tb_labels == s).mean():.1%})"
    for s in (1, -1, 0))
print(f"  {n} entries (every 3rd bar), same barriers")
print(f"  class balance:  {bal}")
means = "   ".join(
    f"{s:+d}: {tb_rets[tb_labels == s].mean():+.2%}"
    for s in (1, -1, 0))
print(f"  mean realized return by label:  {means}")
dis = int((tb_labels != fh_labels).sum())
print(f"  fixed-horizon vs triple-barrier: disagree on {dis} of {n} "
      f"({dis / n:.1%})")

# ---------- Part 3: flip the asymmetry, the dataset changes ----------
print("Part 3 - same entries, flipped bracket (+1.0% / -2.0%): "
      "labels are policy")
flip = np.array([triple_barrier(prices, e, pt=0.01, sl=0.02)[0]
                 for e in entries])
bal2 = "   ".join(
    f"{s:+d}: {(flip == s).sum()} ({(flip == s).mean():.1%})"
    for s in (1, -1, 0))
print(f"  class balance:  {bal2}")
changed = int((flip != tb_labels).sum())
print(f"  {changed} of {n} entries changed label - same prices, "
      f"same features, different question")
print("  verdict: the label is the exit policy written as data - "
      "choose the barriers")
print("           your bracket would actually trade, or the model "
      "learns someone else's trade")
