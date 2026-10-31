"""Day 69 - Data Pipelines for ML: What "MLOps" Actually Means Day-to-Day.

Phase E. Strip the buzzword and MLOps day-to-day is mostly
plumbing around the model - and the #1 production ML failure
is not the model at all. It is the data path, specifically:

    TRAINING-SERVING SKEW - the feature computed one way in
    training (offline batch job, full history on disk) and
    another way at serving (online job, partial live data),
    so the two silently disagree. The model was validated on
    one set of numbers and is fed a different set in prod.
    No exception, no crash - just quiet degradation.

The nastiest subspecies is a point-in-time violation: the
offline job computes a feature for a prediction at time t
using data that did not exist at t. That is Day 32's
no-peeking rule promoted from a notebook habit to a systems
law: a feature for time t may only use events with ts < t.

The fix is a contract, not a tool: ONE feature definition in
ONE place, imported by both paths, with the point-in-time
rule asserted inside it - plus freshness and quality checks
on the log it reads. Feature stores, schedulers and stream
processors are interchangeable brands of plumbing; the
contract is the part that cannot be bought.

  Part 1  The bug. One event log of price ticks, one spec
          ("mean of last 5 closes BEFORE t"), and two buggy
          copies: the training job uses a centered window
          that peeks at ticks after t; the serving job uses
          the last 3 closes. Print the skew table - the two
          paths disagree by whole points, silently.
  Part 2  The fix. Both paths import the SAME function.
          Features come out identical (max diff 0.0), and a
          freshness check reports how stale the newest
          usable event is.
  Part 3  The guard. The shared function asserts
          point-in-time correctness. Feed it a poisoned
          window with the tick AT t smuggled in and it fails
          loudly - the exact failure Part 1 let through.

Pure standard library - no torch needed tonight, this is a
data-path lesson. Runs in the ~/ai-lab venv on the MacBook:
no CUDA, no paid APIs, no downloads, fully deterministic.

Expected output:

  Part 1 - two copies of "the same" feature: the skew
    spec: mean of last 5 closes BEFORE t
      t      spec     train     serve     skew
      6  103.8000  108.0000  105.3333   2.6667
      8  107.0000  111.4000  109.0000   2.4000
      9  108.0000  113.6000  109.6667   3.9333
    max skew 3.9333 - and note train != spec either: the peek inflated it
    no exception, no log line - the model just eats different numbers in prod
  Part 2 - ONE shared definition, imported by both paths
      t     train     serve     diff
      6  103.8000  103.8000   0.0000
      8  107.0000  107.0000   0.0000
      9  108.0000  108.0000   0.0000
    max diff 0.0000 - same code, same window, same numbers, by construction
    freshness check: newest event usable at t=9 is ts=8 (1 tick old) - OK
  Part 3 - the point-in-time guard fails loudly
    a poisoned window for t=8 smuggles in the tick AT ts=8 (the one being predicted)
    AssertionError: point-in-time violation: event ts=8 is not < t=8
    in Part 1's training job that same event passed silently - the silence is the bug class
    verdict: one definition, shared by both paths, point-in-time asserted - that IS the MLOps job
"""

# One source of truth: an append-only event log of price ticks.
# (ts, close) - in prod this is your Kafka topic / S3 partition;
# the shape is identical: timestamped facts, arriving in order.
EVENTS = [
    (0, 100.0), (1, 102.0), (2, 101.0), (3, 105.0),
    (4, 103.0), (5, 108.0), (6, 107.0), (7, 112.0),
    (8, 110.0), (9, 115.0), (10, 113.0), (11, 118.0),
]

# The spec, in one sentence: for a prediction at time t, the
# feature is the mean of the last 5 closes that existed BEFORE
# t. Everything below is either obeying this or violating it.


# ---------- Part 1: two private copies of the feature ----------
def buggy_train_feature(events, t):
    """Offline batch job. The engineer had the whole table, so
    they grabbed a 'centered' window around t - which peeks at
    ticks at t+1 and t+2 that did not exist at prediction time."""
    window = [c for ts, c in events if t - 2 <= ts <= t + 2]
    return sum(window) / len(window)


def buggy_serve_feature(events, t):
    """Online job. Someone 'optimized' it to the last 3 closes
    before t. Point-in-time legal - but it is a DIFFERENT
    feature wearing the same name."""
    window = [c for ts, c in events if ts < t][-3:]
    return sum(window) / len(window)


def spec_feature(events, t):
    """What the spec actually says - ground truth for the table."""
    window = [c for ts, c in events if ts < t][-5:]
    return sum(window) / len(window)


print('Part 1 - two copies of "the same" feature: the skew')
print("  spec: mean of last 5 closes BEFORE t")
print(f"  {'t':>3} {'spec':>9} {'train':>9} {'serve':>9} {'skew':>8}")
max_skew = 0.0
for t in (6, 8, 9):
    spec = spec_feature(EVENTS, t)
    tr = buggy_train_feature(EVENTS, t)
    sv = buggy_serve_feature(EVENTS, t)
    skew = abs(tr - sv)
    max_skew = max(max_skew, skew)
    print(f"  {t:>3} {spec:>9.4f} {tr:>9.4f} {sv:>9.4f} {skew:>8.4f}")
print(f"  max skew {max_skew:.4f} - and note train != spec either: "
      "the peek inflated it")
print("  no exception, no log line - the model just eats "
      "different numbers in prod")

# ---------- Part 2: one shared definition, both paths ----------
def assert_point_in_time(window_events, t):
    """The systems law, enforced: every event a feature uses
    must have existed strictly before the prediction time."""
    for ts, _ in window_events:
        assert ts < t, (f"point-in-time violation: event ts={ts} "
                        f"is not < t={t}")


def mean_last5_before(events, t):
    """THE feature definition - the only one. The training job
    and the serving job both import this exact function."""
    window = [(ts, c) for ts, c in events if ts < t][-5:]
    assert_point_in_time(window, t)
    return sum(c for _, c in window) / len(window)


print("Part 2 - ONE shared definition, imported by both paths")
print(f"  {'t':>3} {'train':>9} {'serve':>9} {'diff':>8}")
max_diff = 0.0
for t in (6, 8, 9):
    tr = mean_last5_before(EVENTS, t)   # training path calls it
    sv = mean_last5_before(EVENTS, t)   # serving path calls it
    max_diff = max(max_diff, abs(tr - sv))
    print(f"  {t:>3} {tr:>9.4f} {sv:>9.4f} {abs(tr - sv):>8.4f}")
print(f"  max diff {max_diff:.4f} - same code, same window, "
      "same numbers, by construction")
newest = max(ts for ts, _ in EVENTS if ts < 9)
print(f"  freshness check: newest event usable at t=9 is ts={newest} "
      f"({9 - newest} tick old) - OK")

# ---------- Part 3: the guard fails loudly ----------
print("Part 3 - the point-in-time guard fails loudly")
poisoned = [(ts, c) for ts, c in EVENTS if ts < 8][-4:] + [(8, 110.0)]
print("  a poisoned window for t=8 smuggles in the tick AT ts=8 "
      "(the one being predicted)")
try:
    assert_point_in_time(poisoned, 8)
    print("  NOT REACHED - the guard let a future event through")
except AssertionError as e:
    print(f"  AssertionError: {e}")
print("  in Part 1's training job that same event passed silently - "
      "the silence is the bug class")
print("  verdict: one definition, shared by both paths, point-in-time "
      "asserted - that IS the MLOps job")
