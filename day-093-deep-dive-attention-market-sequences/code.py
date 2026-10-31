"""Day 93 - Deep Dive: Attention Over Market Sequences.

Days 36-42 built attention over words. Days 83-92 built
the honest finance pipeline. Tonight they collide: the
SAME scaled dot-product attention from Day 38, run over
market days instead of tokens. A day becomes a token by
embedding it - Day 85's five features, z-scored against
pre-window history, pushed through one fixed projection.

The machinery transfers unchanged, because a sequence
is a sequence: scores = QK^T / sqrt(d_k), softmax the
rows, blend the values (Day 38), causal mask on top -
which in finance is not a trick but the law: the mask
IS Day 69's point-in-time rule, enforced by the
architecture. Day t literally cannot read day t+1.

What is DIFFERENT from Day 38: the Q/K projections are
hand-designed, not learned. They are built to ask one
question - "which past days were shocks?" - by keying
on the volume z-score and realized-vol features, so the
two planted reversal-shock days (53 and 57) should light
up in the map. That is a demonstration of how to READ a
map, not a discovery. In a real model those projections
are learned by gradient descent on a prediction task
(Days 42/44), and whatever lights up is the start of an
investigation - it still has to survive Day 87's
walk-forward and Day 84's trial-counting before it is
allowed to mean anything. Attention weights are a
mixing coefficient, not a cause: a day can hold a huge
weight and contribute almost nothing if its value
vector is tiny, and a low-weight day can matter through
later layers. The map shows where the model LOOKED.
It does not prove why the output moved.

  Part 1  A synthetic 60-day series (seed 93, 1%/day
          vol) with two planted reversal shocks, and
          the Day 85 feature factory z-scored on days
          20-49 only. The last 10 days become tokens.
  Part 2  Day 38's attention, by hand in numpy, causal
          mask on: the full 10x10 weight map, each
          query's top-attended day, and how much of each
          late row's budget the two shock days capture.
  Part 3  The receipts: day 59's output is a blend -
          print it against its own embedding - and an
          ablation: cut the volume channel out of the
          keys and watch the shock days go dark. The
          map follows the mechanism. That is all a map
          ever does.

Runs on the ~/ai-lab venv: numpy only (free), no CUDA,
no paid APIs, no downloads. Fixed seed, so every number
is reproducible. (Day 38 used torch; the formula is the
same - numpy keeps tonight self-contained.)

Expected output:

Part 1 - 60 synthetic days, shocks planted at 53 and 57; last 10 days are the tokens
  close: day 50 105.62   day 53 103.99   day 57 107.39   day 59 108.18
   day   ret %   z-scored features (fit on days 20-49)    ret_5d  rvol_20  rsi_14   vwap   vol_z
    50   -0.01        -1.46    -2.29   -2.67  -2.62   -0.88
    51    0.95         0.09    -1.91   -1.68  -1.76    1.25
    52    2.10         1.40    -0.52    0.01  -0.05    3.24
    53   -4.60        -1.93     4.33   -3.13  -3.40    3.89  <- SHOCK
    54    0.33        -0.87     4.38   -2.90  -3.03   -0.28
    55    0.49        -0.53     4.43   -2.56  -2.50   -0.22
    56   -1.90        -2.47     4.96   -3.50  -3.75    1.70
    57    4.30        -0.97     8.24   -1.04  -0.14    2.44  <- SHOCK
    58    0.90         2.77     8.36   -0.63   0.60    0.19
    59   -0.17         2.43     8.37   -0.73   0.44   -0.63
  read vol_z: day 53 tops the column (3.89); day 57's 2.44 is muted - its trailing
  window already holds day 53's spike (z means 'unusual for lately' - Day 85, clause 3).
  and rvol_20 stays hot for days AFTER a shock (its 20-day window remembers) -
  watch that smear spread attention onto days 58-59 in the map.
Part 2 - the attention map: rows are queries (who is asking), columns are keys (who gets read)
       K   50K   51K   52K   53K   54K   55K   56K   57K   58K   59
  Q 50 1.00    .    .    .    .    .    .    .    .    .
  Q 51 0.31 0.69    .    .    .    .    .    .    .    .
  Q 52 0.07 0.23 0.70    .    .    .    .    .    .    .
  Q 53 0.01 0.04 0.13 0.82    .    .    .    .    .    .
  Q 54 0.02 0.05 0.15 0.69 0.09    .    .    .    .    .
  Q 55 0.02 0.04 0.15 0.63 0.08 0.08    .    .    .    .
  Q 56 0.01 0.02 0.07 0.55 0.06 0.06 0.24    .    .    .
  Q 57 0.00 0.00 0.03 0.31 0.02 0.02 0.10 0.51    .    .
  Q 58 0.00 0.00 0.04 0.21 0.02 0.02 0.05 0.44 0.22    .
  Q 59 0.00 0.00 0.04 0.19 0.02 0.02 0.05 0.39 0.19 0.11
  '.' = masked future - the causal mask is Day 69's point-in-time law, in the architecture
  every row sums to 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000 - a budget (Day 14)
  day 50 looks hardest at day 50 (weight 1.00)
  day 51 looks hardest at day 51 (weight 0.69)
  day 52 looks hardest at day 52 (weight 0.70)
  day 53 looks hardest at day 53 (weight 0.82)
  day 54 looks hardest at day 53 (weight 0.69)
  day 55 looks hardest at day 53 (weight 0.63)
  day 56 looks hardest at day 53 (weight 0.55)
  day 57 looks hardest at day 57 (weight 0.51)
  day 58 looks hardest at day 57 (weight 0.44)
  day 59 looks hardest at day 57 (weight 0.39)
  day 58: the two shock days hold 0.65 of its budget (uniform over its 9 legal keys would give them 0.22)
  day 59: the two shock days hold 0.58 of its budget (uniform over its 10 legal keys would give them 0.20)
  attention received per legal query (column averages):
    d50 0.14  d51 0.12  d52 0.16  d53 0.49  d54 0.05  d55 0.04  d56 0.11  d57 0.45  d58 0.21  d59 0.11
Part 3 - what the map DOES, and what it does not do
  day 59's own embedding : [ 2.43  8.37 -0.73  0.44 -0.63  3.05]
  day 59's output (blend): [-0.06  6.9  -1.47 -0.82  1.85  2.61]
  the blend is pulled toward the shock days' embeddings - 0.58 of it IS days 53+57, by weight
  ablation - volume channel cut from the keys: day 59's shock share falls 0.58 -> 0.19
  the map followed the mechanism both times. A bright cell says 'this value got mixed in',
  never 'this day caused the prediction' - weights are not explanations (Jain & Wallace).
verdict: same formula, new tokens. Fluency is reading the map; honesty is knowing it proves nothing
         until walk-forward (Day 87) and trial-counting (Day 84) say otherwise.
"""
import numpy as np

SEED = 93
N = 60
SHOCKS = (53, 57)          # planted reversal-shock days
WIN = list(range(50, 60))  # the attention window: days 50..59

# ---------- Part 1: series + Day 85 features ----------
rng = np.random.default_rng(SEED)
z = rng.standard_normal(N)
logret = 0.0002 + 0.010 * z          # calm tape: 1%/day vol
logret[0] = 0.0
logret[52] = 0.021                   # run-up into the first shock
logret[53] = -0.046                  # SHOCK 1: -4.6% reversal day
logret[56] = -0.019                  # drift down into the second
logret[57] = 0.043                   # SHOCK 2: +4.3% reversal day
price = 100.0 * np.exp(np.cumsum(logret))
volume = (1_000_000 * (1.0 + 1.5 * np.abs(logret) / 0.010)
          * np.exp(0.15 * rng.standard_normal(N))).round().astype(int)


def rsi_wilder(close, period=14):
    """Day 85's Wilder RSI, unchanged: causal by construction."""
    n = len(close)
    out = np.full(n, np.nan)
    d = np.diff(close)
    gains = np.where(d > 0, d, 0.0)
    losses = np.where(d < 0, -d, 0.0)
    ag, al = gains[:period].mean(), losses[:period].mean()
    for t in range(period, n):
        if t > period:
            ag = (ag * (period - 1) + gains[t - 1]) / period
            al = (al * (period - 1) + losses[t - 1]) / period
        out[t] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out


def compute_features(close, vol):
    """Day 85's five features - every one trailing-only."""
    n = len(close)
    lr = np.zeros(n)
    lr[1:] = np.log(close[1:] / close[:-1])
    f = {k: np.full(n, np.nan)
         for k in ("ret_5d", "rvol_20", "rsi_14", "vwap_dev", "vol_z")}
    f["rsi_14"] = rsi_wilder(close)
    for t in range(n):
        if t >= 5:
            f["ret_5d"][t] = 100.0 * lr[t - 4:t + 1].sum()
        if t >= 20:
            f["rvol_20"][t] = 100.0 * lr[t - 19:t + 1].std() \
                * np.sqrt(252)
            vw = (close[t - 19:t + 1] * vol[t - 19:t + 1]).sum() \
                / vol[t - 19:t + 1].sum()
            f["vwap_dev"][t] = 100.0 * (close[t] / vw - 1.0)
            w = vol[t - 19:t + 1].astype(float)
            f["vol_z"][t] = (vol[t] - w.mean()) / w.std()
    return f


FEATS = compute_features(price, volume)
NAMES = ("ret_5d", "rvol_20", "rsi_14", "vwap_dev", "vol_z")
RAW = np.stack([FEATS[k] for k in NAMES], axis=1)      # (60, 5)
# z-score with PRE-WINDOW history only (days 20-49): the scaler
# is fit on the past, never on the window it will describe.
mu, sd = RAW[20:50].mean(axis=0), RAW[20:50].std(axis=0)
Z = (RAW - mu) / sd
F = Z[WIN]                                             # (10, 5) tokens

print("Part 1 - 60 synthetic days, shocks planted at 53 and 57; "
      "last 10 days are the tokens")
print(f"  close: day 50 {price[50]:.2f}   day 53 {price[53]:.2f}   "
      f"day 57 {price[57]:.2f}   day 59 {price[59]:.2f}")
print(f"  {'day':>4}{'ret %':>8}   z-scored features (fit on days 20-49)"
      f"{'':>2}{'ret_5d':>8}{'rvol_20':>9}{'rsi_14':>8}{'vwap':>7}"
      f"{'vol_z':>8}")
for i, t in enumerate(WIN):
    mark = "  <- SHOCK" if t in SHOCKS else ""
    print(f"  {t:>4}{100 * logret[t]:>8.2f}   "
          f"{'':>2}{F[i, 0]:>8.2f}{F[i, 1]:>9.2f}{F[i, 2]:>8.2f}"
          f"{F[i, 3]:>7.2f}{F[i, 4]:>8.2f}{mark}")
print("  read vol_z: day 53 tops the column (3.89); day 57's 2.44 is "
      "muted - its trailing")
print("  window already holds day 53's spike (z means 'unusual for "
      "lately' - Day 85, clause 3).")
print("  and rvol_20 stays hot for days AFTER a shock (its 20-day "
      "window remembers) -")
print("  watch that smear spread attention onto days 58-59 in the map.")

# ---------- Part 2: Day 38 attention, causal, by hand ----------
# Embedding: fixed stated projection, 5 features -> d_model 6.
# Columns 0-4 carry the features; column 5 is a mix (ret+vol+rvol).
P = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.3],   # ret_5d
              [0.0, 1.0, 0.0, 0.0, 0.0, 0.3],   # rvol_20
              [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],   # rsi_14
              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],   # vwap_dev
              [0.0, 0.0, 0.0, 0.0, 1.0, 0.3]])  # vol_z
E = F @ P                                              # (10, 6)

# Hand-designed projections (NOT learned - see the docstring).
# Every query asks roughly the same question ("which past days
# were shocks?"): q0 is mostly the bias, with a nudge from the
# mix column; q1 scales with the query's own vol regime.
d_k = 3
Wq = np.zeros((6, d_k))
Wq[5, 0] = 0.2        # mix -> q0
Wq[1, 1] = 0.2        # own rvol -> q1
Wq[0, 2] = 0.3        # own ret_5d -> q2
bq = np.array([0.9, 0.0, 0.0])
# Keys advertise: k0 = volume shock, k1 = vol regime, k2 = move.
Wk = np.zeros((6, d_k))
Wk[4, 0] = 0.75       # vol_z -> k0
Wk[1, 1] = 0.4        # rvol_20 -> k1
Wk[0, 2] = 0.45       # ret_5d -> k2
Q = E @ Wq + bq                                        # (10, 3)
K = E @ Wk                                             # (10, 3)
V = E.copy()                        # Wv = identity: values are
                                    # the embeddings themselves


def causal_attention(Q, K, V):
    """Day 38's five moves, numpy edition: scores, scale,
    causal mask (future -> -inf BEFORE the softmax), softmax
    rows, blend the values."""
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    future = np.triu(np.ones(scores.shape, dtype=bool), k=1)
    scores = np.where(future, -np.inf, scores)
    m = scores.max(axis=1, keepdims=True)
    e = np.exp(scores - m)
    w = e / e.sum(axis=1, keepdims=True)
    return w, w @ V


W, OUT = causal_attention(Q, K, V)
shock_cols = [WIN.index(s) for s in SHOCKS]

print("Part 2 - the attention map: rows are queries (who is "
      "asking), columns are keys (who gets read)")
print("       " + "".join(f"K{d:>5}" for d in WIN))
for i, t in enumerate(WIN):
    cells = "".join("    ." if j > i else f"{W[i, j]:5.2f}"
                    for j in range(10))
    print(f"  Q{t:>3}{cells}")
print("  '.' = masked future - the causal mask is Day 69's "
      "point-in-time law, in the architecture")
sums = ", ".join(f"{v:.4f}" for v in W.sum(axis=1))
print(f"  every row sums to {sums} - a budget (Day 14)")
for i, t in enumerate(WIN):
    j = int(np.argmax(W[i]))
    print(f"  day {t} looks hardest at day {WIN[j]} "
          f"(weight {W[i, j]:.2f})")
for i in (8, 9):
    share = W[i, shock_cols].sum()
    print(f"  day {WIN[i]}: the two shock days hold {share:.2f} of "
          f"its budget (uniform over its {i + 1} legal keys would "
          f"give them {2 / (i + 1):.2f})")
recv = W.sum(axis=0) / (10 - np.arange(10))
print("  attention received per legal query (column averages):")
print("    " + "  ".join(f"d{d} {recv[j]:.2f}"
                         for j, d in enumerate(WIN)))

# ---------- Part 3: the output blend + the ablation ----------
print("Part 3 - what the map DOES, and what it does not do")
print(f"  day 59's own embedding : "
      + np.array2string(E[9], precision=2, suppress_small=True))
print(f"  day 59's output (blend): "
      + np.array2string(OUT[9], precision=2, suppress_small=True))
print(f"  the blend is pulled toward the shock days' embeddings - "
      f"{W[9, shock_cols].sum():.2f} of it IS days 53+57, by weight")
Wk_cut = Wk.copy()
Wk_cut[4, 0] = 0.0            # keys stop advertising volume
W2, _ = causal_attention(Q, E @ Wk_cut, V)
print(f"  ablation - volume channel cut from the keys: day 59's "
      f"shock share falls {W[9, shock_cols].sum():.2f} -> "
      f"{W2[9, shock_cols].sum():.2f}")
print("  the map followed the mechanism both times. A bright cell "
      "says 'this value got mixed in',")
print("  never 'this day caused the prediction' - weights are "
      "not explanations (Jain & Wallace).")
print("verdict: same formula, new tokens. Fluency is reading "
      "the map; honesty is knowing it proves nothing")
print("         until walk-forward (Day 87) and trial-counting "
      "(Day 84) say otherwise.")

# ---------- self-checks ----------
assert np.allclose(W.sum(axis=1), 1.0)
assert np.all(W[np.triu_indices(10, k=1)] == 0.0)
assert WIN[int(np.argmax(W[9]))] in SHOCKS
assert W[9, shock_cols].sum() > 0.50
