"""Day 90 - Position Sizing & Risk: Kelly Intuition, Volatility
Targeting.

Phase G has spent seven days on the prediction half: why
finance is hard (Day 83), honest labels and validation
(Days 84-89). Tonight is the other half - the half that
decides whether a real edge becomes real growth:

    Size. A bet's growth rate is set not by its expected
    value alone but by the fraction of bankroll you stake.

  Part 1  The Kelly criterion. A bet that wins with
          p = 0.55 at even money (b = 1): the growth-optimal
          stake is f* = p - (1-p)/b = 0.10. First the exact
          arithmetic - expected log growth per bet at five
          stakes - then ONE shared 500-bet path at 1/4, 1/2,
          1x, 1.5x and 2x Kelly. Same bets, same order, only
          the stake changes. 2x Kelly has the same positive
          edge as everyone else and still finishes UNDER
          water: past f*, extra stake buys variance, not
          growth, and variance compounds against you.
  Part 2  Volatility targeting. Kelly answers "how much of
          my roll on this bet"; vol targeting answers "how
          much risk per position": stake in inverse
          proportion to realized volatility. One asset, two
          regimes (calm 1% daily sd, storm 3%). Fixed
          fraction rides the regime change; vol-targeted
          sizing holds the daily-return sd nearly flat
          across both.

Runs on the ~/ai-lab venv: numpy only (free), CPU, no CUDA,
no paid APIs, no downloads. Two stated seeds; every number
below is the real run.

Expected output:

Part 1 - Kelly: one bet, five stakes (p = 0.55, b = 1, f* = 0.10)
  expected log growth per bet, g(f) = p*ln(1+b*f) + (1-p)*ln(1-f):
    0.25x Kelly  f = 0.025   g = +0.00219 per bet
     0.5x Kelly  f = 0.050   g = +0.00375 per bet
     1.0x Kelly  f = 0.100   g = +0.00501 per bet
     1.5x Kelly  f = 0.150   g = +0.00374 per bet
     2.0x Kelly  f = 0.200   g = -0.00014 per bet
  g peaks exactly at f* - past it, growth FALLS while swings keep growing
  one 500-bet path (seed 90, wins realized: 275/500), every stake rides it:
    0.25x Kelly  final    2.99x   max drawdown  27.3%
     0.5x Kelly  final    6.53x   max drawdown  49.2%
     1.0x Kelly  final   12.23x   max drawdown  84.0%
     1.5x Kelly  final    6.47x   max drawdown  98.0%
     2.0x Kelly  final    0.93x   max drawdown  99.9%
  same edge, same luck - 2x Kelly ends below 1.00x. Size is a growth decision, not a courage decision.
Part 2 - volatility targeting: size = target / realized vol
    fixed 100%    daily sd: calm 0.0090  storm 0.0291  overall 0.0216
    vol-targeted  daily sd: calm 0.0106  storm 0.0120  overall 0.0113
  fixed sizing lets the storm set your risk; targeting trades stake, not nerve - calm stake ~1.17x, storm stake ~0.37x
"""
import numpy as np

# ---------- Part 1: Kelly - the growth-optimal stake ----------
P, B = 0.55, 1.0                       # win prob, payoff ratio
F_STAR = P - (1 - P) / B               # 0.55 - 0.45 = 0.10
N_BETS = 500
MULTS = (0.25, 0.5, 1.0, 1.5, 2.0)

print("Part 1 - Kelly: one bet, five stakes "
      f"(p = {P}, b = {B:.0f}, f* = {F_STAR:.2f})")
print("  expected log growth per bet, g(f) = "
      "p*ln(1+b*f) + (1-p)*ln(1-f):")


def log_growth(f):
    return P * np.log(1 + B * f) + (1 - P) * np.log(1 - f)


for m in MULTS:
    print(f"    {m:>4}x Kelly  f = {m * F_STAR:.3f}   "
          f"g = {log_growth(m * F_STAR):+.5f} per bet")
print("  g peaks exactly at f* - past it, growth FALLS "
      "while swings keep growing")

# One shared path: exactly 275 wins / 225 losses (p = 0.55
# realized exactly), shuffled into one fixed order, seed 90.
rng = np.random.default_rng(90)
outcomes = np.array([1] * 275 + [-1] * 225)
rng.shuffle(outcomes)
print(f"  one 500-bet path (seed 90, wins realized: "
      f"{int((outcomes > 0).sum())}/500), every stake rides it:")
paths = {}
for m in MULTS:
    f = m * F_STAR
    wealth = np.cumprod(1 + f * outcomes)      # start = 1.0
    paths[m] = wealth
    peak = np.maximum.accumulate(wealth)
    max_dd = (1 - wealth / peak).max()
    print(f"    {m:>4}x Kelly  final {wealth[-1]:7.2f}x   "
          f"max drawdown {max_dd:6.1%}")
print("  same edge, same luck - 2x Kelly ends below 1.00x. "
      "Size is a growth decision, not a courage decision.")

# ---------- Part 2: volatility targeting ----------
print("Part 2 - volatility targeting: size = target / "
      "realized vol")
rng2 = np.random.default_rng(7)
asset = np.concatenate([rng2.normal(0.0, 0.01, 250),   # calm
                        rng2.normal(0.0, 0.03, 250)])  # storm
TARGET, WIN = 0.01, 20
# stake for day t uses only the trailing 20 days before t
# (Day 87's rule: no peeking); days 0-19 assume the target.
vol_est = np.full(len(asset), TARGET)
for t in range(WIN, len(asset)):
    vol_est[t] = asset[t - WIN:t].std(ddof=1)
w = np.clip(TARGET / vol_est, 0.0, 2.0)    # cap leverage 2x
fixed = asset                              # 100% always
targeted = w * asset
for name, rets in (("fixed 100%", fixed),
                   ("vol-targeted", targeted)):
    print(f"    {name:<13} daily sd: calm "
          f"{rets[:250].std(ddof=1):.4f}  storm "
          f"{rets[250:].std(ddof=1):.4f}  overall "
          f"{rets.std(ddof=1):.4f}")
print("  fixed sizing lets the storm set your risk; "
      "targeting trades stake, not nerve - calm stake "
      f"~{w[100]:.2f}x, storm stake ~{w[400]:.2f}x")
