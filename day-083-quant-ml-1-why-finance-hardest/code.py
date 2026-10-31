"""Day 83 - Quant ML 1: Why Finance Is the Hardest ML Domain.

Phase G opener. Days 1-82 built the general machine:
fit f_theta, grade it on held-out data, ship it. Finance
keeps the machine and breaks its assumptions. Four
pieces of failure physics, one per paragraph of the
lesson:

  1. Signal-to-noise. A GOOD strategy's true edge is
     ~0.04% per day against ~1% daily noise - a per-day
     Sharpe of 0.04. The signal is a rounding error on
     the noise, every single day.
  2. Non-stationarity. The process generating the data
     moves: regimes flip, and yesterday's edge can
     become tomorrow's drag. Cats stay cats; markets
     rewire themselves.
  3. Tiny effective samples. One market, one history.
     Overlapping windows are not independent samples,
     so "10 years of data" is far fewer real observations
     than it looks - and they all share the same regimes.
  4. Adversarial decay. A published or crowded edge gets
     traded against and arbitraged away. The dataset
     fights back.

Tonight's code makes (1) and (2) measurable:

  Part 1  Two simulated strategies, same vol. One has
          the real edge above, one is pure noise. Watch
          the t-statistic of the mean - the standard
          "is this distinguishable from zero?" score -
          at one quarter, one year, five years, ten.
          Theory says the AVERAGE edge strategy needs
          (2*sigma/mu)^2 = 2,500 days to reach t = 2.
  Part 2  The luck factory: 100 pure-noise strategies,
          one year each. Count how many print t > 2 -
          a "proven" track record manufactured by chance
          alone, ready to be sold as skill.
  Part 3  Non-stationarity, one run: the strategy's
          edge is +0.08%/day for five years, then flips
          to -0.08%/day for five. The pooled 10-year
          backtest averages the two regimes into a lie.

Everything is simulated: no market data, no downloads,
no APIs. numpy only, one fixed seed (83) - the seed is
part of the lesson, because the exact wiggle of any one
run is luck; the SIZES are the physics. Runs in the
~/ai-lab venv on CPU in about a second.

Expected output:

  Part 1 - the edge is real, and invisible for years
    true edge: mu = +0.04%/day, sigma = 1.00%/day  (per-day Sharpe 0.04)
    theory: days for the average edge run to reach t = 2: (2*sigma/mu)^2 = 2500 days = 9.9 years
    one simulated pair, same seed, 10 years:
      after    63 days (one quarter): edge t = +0.66   noise t = +0.05
      after   252 days (one year   ): edge t = +0.57   noise t = +0.24
      after  1260 days (five years ): edge t = +2.07   noise t = +1.33
      after  2520 days (ten years  ): edge t = +2.37   noise t = +1.11
    this edge run first touches t = 2 on day 562 - and a touch is not a hold
    the edge strategy's realized Sharpe over the decade: 0.75 annualized - a good strategy
    verdict: at one year, the REAL edge is statistically indistinguishable from luck
  Part 2 - the luck factory: 100 strategies with NO edge, 1 year each
    pure-noise strategies printing t > 2 after 252 days: 5 of 100
    the best of them: t = +2.80 - at exactly one year, t IS the annualized Sharpe
    chance alone predicts ~2-3 of 100 (a one-sided t > 2 is a ~2.3% event)
    verdict: a great one-year track record is what luck looks like at scale
  Part 3 - the target moves: one strategy, regime flip at year 5
    years 1-5  (edge +0.08%/day): t = +1.93, cumulative +0.680
    years 6-10 (edge -0.08%/day): t = -3.73, cumulative -1.301
    all 10 years pooled:          t = -1.25 -> reads as 'no edge'
    verdict: pooled, the backtest learns a world that never existed
"""
import numpy as np

MU, SIGMA = 0.0004, 0.01      # true edge: +0.04%/day; daily vol: 1%
DAYS = 2520                   # 10 years x 252 trading days


def t_stat(r):
    """t-statistic of the mean: how many standard errors
    the average daily return sits from zero."""
    return r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))


def running_t(r):
    """t-statistic of every prefix of r, vectorized via
    cumulative sums (used to find the first t = 2 touch)."""
    n = np.arange(1, len(r) + 1)
    mean = np.cumsum(r) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        # n = 1 has no sample std; we only ever read n >= 30.
        var = (np.cumsum(r ** 2) / n - mean ** 2) * n / (n - 1)
        return mean / (np.sqrt(var) / np.sqrt(n))


rng = np.random.default_rng(83)
edge = rng.normal(MU, SIGMA, DAYS)     # real edge, buried in noise
noise = rng.normal(0.0, SIGMA, DAYS)   # no edge at all, same vol

# ---------- Part 1: how long until real is tellable from lucky? ----
print("Part 1 - the edge is real, and invisible for years")
print(f"  true edge: mu = +{MU * 100:.2f}%/day, sigma = {SIGMA * 100:.2f}%/day"
      f"  (per-day Sharpe {MU / SIGMA:.2f})")
need = (2.0 * SIGMA / MU) ** 2
print(f"  theory: days for the average edge run to reach t = 2:"
      f" (2*sigma/mu)^2 = {need:.0f} days = {need / 252:.1f} years")
print("  one simulated pair, same seed, 10 years:")
for n, label in ((63, "one quarter"), (252, "one year"),
                 (1260, "five years"), (2520, "ten years")):
    print(f"    after {n:>5} days ({label:<11}): "
          f"edge t = {t_stat(edge[:n]):+.2f}   noise t = {t_stat(noise[:n]):+.2f}")
cross = int(np.argmax(running_t(edge)[29:] >= 2.0)) + 30
print(f"  this edge run first touches t = 2 on day {cross}"
      " - and a touch is not a hold")
sharpe = edge.mean() / edge.std(ddof=1) * np.sqrt(252)
print(f"  the edge strategy's realized Sharpe over the decade:"
      f" {sharpe:.2f} annualized - a good strategy")
print("  verdict: at one year, the REAL edge is statistically"
      " indistinguishable from luck")

# ---------- Part 2: the luck factory ----------
print("Part 2 - the luck factory: 100 strategies with NO edge, 1 year each")
crowd = rng.normal(0.0, SIGMA, (100, 252))
cts = crowd.mean(axis=1) / (crowd.std(axis=1, ddof=1) / np.sqrt(252))
lucky = int((cts > 2.0).sum())
print(f"  pure-noise strategies printing t > 2 after 252 days: {lucky} of 100")
print(f"  the best of them: t = {cts.max():+.2f} - at exactly one year,"
      " t IS the annualized Sharpe")
print("  chance alone predicts ~2-3 of 100 (a one-sided t > 2 is a ~2.3% event)")
print("  verdict: a great one-year track record is what luck looks like at scale")

# ---------- Part 3: the target moves ----------
print("Part 3 - the target moves: one strategy, regime flip at year 5")
flip = rng.normal(0.0, SIGMA, DAYS)
flip[:1260] += 0.0008          # regime 1: the edge is real, +0.08%/day
flip[1260:] -= 0.0008          # regime 2: the same edge, inverted
print(f"  years 1-5  (edge +0.08%/day): t = {t_stat(flip[:1260]):+.2f},"
      f" cumulative {flip[:1260].sum():+.3f}")
print(f"  years 6-10 (edge -0.08%/day): t = {t_stat(flip[1260:]):+.2f},"
      f" cumulative {flip[1260:].sum():+.3f}")
print(f"  all 10 years pooled:          t = {t_stat(flip):+.2f}"
      " -> reads as 'no edge'")
print("  verdict: pooled, the backtest learns a world that never existed")
