"""Day 70 - Monitoring Models in Prod: Drift.

A deployed model does not crash when it goes stale. It keeps
answering, confidently, while the world it was trained on
quietly moves out from under it. That silent decay is DRIFT,
and catching it is a monitoring problem - the ML version of
your Prometheus alerts.

Two kinds, one paragraph each:
  Data (input) drift    - the inputs changed: P(x) moved.
                          New user mix, a client update, a
                          sensor swapped. You can SEE this
                          one - inputs arrive with every
                          request, no labels required.
  Concept drift         - the relationship changed: P(y|x)
                          moved. Same inputs, different right
                          answers (fraudsters adapt, tastes
                          change). Invisible until labels
                          land - which is why it is the
                          dangerous one.

Today's detector is the Population Stability Index (PSI):
bin the live inputs using the TRAINING reference's decile
bins, then

    PSI = sum over bins of (live% - ref%) * ln(live% / ref%)

Identical distributions score 0; every bin that gained or
lost share adds a positive term. Industry rule-of-thumb
bands (rules of thumb, not laws - calibrate per feature):
    PSI < 0.10          stable
    0.10 <= PSI < 0.25  watch - investigate
    PSI >= 0.25         act - retrain / roll back / page

Empty-bin guard: a live bin can hold 0 requests, and
ln(0) is -inf, so every share is floored at eps = 1e-4
before the ratio. Small enough to barely move a real bin,
big enough to keep the math finite.

  Part 1  The reference: one input feature as it looked at
          training time - formula-built from a fixed
          quantile grid of N(50, 10), no RNG anywhere, so
          every digit below is exact and reproducible.
  Part 2  Three live windows vs the reference: no drift,
          a +0.5-sigma mean shift, and a variance blow-up
          (std x2, mean untouched - the case a mean-only
          alert sleeps through). PSI + verdict per window.
  Part 3  Prediction drift, the cheap always-on signal:
          a toy model flags feature > 65 as "large". Watch
          its predicted flag rate move with the inputs -
          computable the second predictions are served,
          long before any label exists.

The workflow this plugs into: drift alerts are the EARLY
WARNING. Confirmation is re-running your eval (Day 57) on
fresh labels when they finally arrive.

Runs on the ~/ai-lab venv: numpy only (free), no torch
needed, no APIs, no downloads, no RNG.

Expected output:

  Part 1 - the reference: what "normal" looked like at training time
    reference feature: n=2000  mean 50.00  std 10.00  (formula-built, no RNG)
    10 bins cut at the reference deciles - each bin holds 10.0% by construction
    PSI = sum over bins of (live% - ref%) * ln(live% / ref%)
    empty-bin guard: every share is floored at eps=0.0001 before the log
  Part 2 - three live windows vs the reference
    no drift          PSI 0.0009   mean shift +0.03 sigma  -> STABLE
    mean shift        PSI 0.2376   mean shift +0.50 sigma  -> WATCH
    variance blow-up  PSI 0.4915   mean shift +0.00 sigma  -> ACT
    read: the variance case never moved its mean - a mean-only alert sleeps through it
  Part 3 - prediction drift: the cheap always-on signal (no labels needed)
    toy model flags any request with feature > 65 as 'large'
    training-time flag rate: 0.067
    no drift          predicted flag rate 0.071
    mean shift        predicted flag rate 0.159
    variance blow-up  predicted flag rate 0.227
    verdict: inputs moved first, predictions followed - accuracy can only confirm it when labels land
"""
import numpy as np
from statistics import NormalDist

ND = NormalDist()          # standard normal, for its inverse CDF
EPS = 1e-4                 # empty-bin guard (see docstring)


def sample(mu, sigma, n):
    """Deterministic 'sample' of N(mu, sigma): evaluate the
    inverse CDF on a fixed midpoint grid. No RNG - the same
    numbers on every machine, every run."""
    p = (np.arange(n) + 0.5) / n
    z = np.array([ND.inv_cdf(float(pi)) for pi in p])
    return mu + sigma * z


def shares(x, edges):
    """Fraction of x in each bin, floored at EPS so ln() in
    the PSI sum never sees a zero."""
    counts, _ = np.histogram(x, bins=edges)
    return np.clip(counts / counts.sum(), EPS, None)


def psi(live_shares, ref_shares):
    return float(np.sum((live_shares - ref_shares)
                        * np.log(live_shares / ref_shares)))


def verdict(value):
    if value < 0.10:
        return "STABLE"
    if value < 0.25:
        return "WATCH"
    return "ACT"


# ---------- Part 1: the reference ----------
print("Part 1 - the reference: what \"normal\" looked like "
      "at training time")
ref = sample(50.0, 10.0, 2000)
deciles = [50.0 + 10.0 * ND.inv_cdf(q / 10) for q in range(1, 10)]
edges = np.array([-np.inf] + deciles + [np.inf])
ref_shares = shares(ref, edges)
print(f"  reference feature: n={len(ref)}  mean {ref.mean():.2f}  "
      f"std {ref.std():.2f}  (formula-built, no RNG)")
print("  10 bins cut at the reference deciles - each bin holds "
      "10.0% by construction")
print("  PSI = sum over bins of (live% - ref%) * ln(live% / ref%)")
print(f"  empty-bin guard: every share is floored at eps={EPS} "
      "before the log")

# ---------- Part 2: three live windows ----------
print("Part 2 - three live windows vs the reference")
scenarios = (
    ("no drift", sample(50.3, 10.0, 1731)),       # a normal Tuesday
    ("mean shift", sample(55.0, 10.0, 2000)),      # +0.5 sigma
    ("variance blow-up", sample(50.0, 20.0, 2000)),  # std x2
)
live_windows = {}
for name, live in scenarios:
    live_windows[name] = live
    value = psi(shares(live, edges), ref_shares)
    shift = (live.mean() - ref.mean()) / ref.std()
    print(f"  {name:<17} PSI {value:.4f}   mean shift "
          f"{shift:+.2f} sigma  -> {verdict(value)}")
print("  read: the variance case never moved its mean - a "
      "mean-only alert sleeps through it")

# ---------- Part 3: prediction drift ----------
print("Part 3 - prediction drift: the cheap always-on signal "
      "(no labels needed)")
print("  toy model flags any request with feature > 65 as 'large'")
print(f"  training-time flag rate: {(ref > 65).mean():.3f}")
for name, live in scenarios:
    print(f"  {name:<17} predicted flag rate "
          f"{(live > 65).mean():.3f}")
print("  verdict: inputs moved first, predictions followed - "
      "accuracy can only confirm it when labels land")
