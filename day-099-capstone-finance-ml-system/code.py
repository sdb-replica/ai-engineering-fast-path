"""Day 99 - Capstone: The Finance-ML System, Assembled End to End.

Day 92 drew the whole system on one page and shipped
a linter for it. Tonight the system RUNS - one file,
one pass, every stage the one you already built:

    lint (Day 92) -> data (Days 83/88) -> features
    (Day 85) -> labels (Day 86) -> walk-forward
    baseline (Days 87/88) -> sizing (Day 90)
    -> risk gate + paper fills + journal (Day 91)
    -> the report card

No new machinery. The capstone claim is narrower and
stronger: the pieces compose, the honesty devices
still fire when nobody is demonstrating them one at
a time, and the system grades itself in public.

  The data is Day 88's ACME again - SYNTHETIC, stated
  process, no downloads: drift mu = 0.0002, vol
  regimes 0.9%/2.0% switching at p = 0.02/day, one
  weak planted mean-reversion signal (KAPPA = 0.10).
  Seed 119 was NOT drawn first. Seed 99, the first
  tried, sent ACME up 6.5x and drowned the machinery
  in drift; seeds 99-120 were scanned and 119 was
  picked because every device in the harness fires
  on it - a clip, rejections, kill-switch trips -
  not because of its P&L. Across those 22 seeds the
  IDENTICAL system returned -3.7% to +10.3%: same
  code, different world, different report card.
  That spread is Days 83-84, measured on this file.

  Part 1  Pre-flight: Day 92's linter, rules verbatim,
          against the config this run actually uses.
          It must pass 8 of 8 before anything trades.
  Part 2  The series, regenerated; the planted signal
          measured before anything learns from it.
  Part 3  Features + labels, Day 88's code unchanged:
          the five features, point-in-time; triple-
          barrier labels that also fix each trade's
          exit (a resting bracket order, Day 91).
  Part 4  Walk-forward, Day 88's folds unchanged:
          purged, z-scored on train stats, logistic
          regression by hand, fixed lr/iters. Each
          fold also reports b, the payoff ratio of
          its own training labels - Part 5's Kelly
          input, estimated only from the past.
  Part 5  The harness (Day 91) with sizing (Day 90):
          signal = model prob >= 0.5; stake = half-
          Kelly on (p, b), vol-targeted, capped by
          the gate at 25% of equity; fills pay 5 bps
          half-spread + 2 bps slippage per side; the
          kill switch flattens any day down $400 and
          pauses entries the next day. Every decision
          is journaled with what the model saw.
  Part 6  The report card. Whatever it says, it
          prints: accuracy vs the majority baseline,
          trades, net return and Sharpe vs buy-and-
          hold over the same span, friction paid,
          kill-switch status, the linter's verdict.

Runs on the ~/ai-lab venv: numpy only (free), no
torch needed, CPU, no CUDA, no paid APIs, no
downloads. Seed stated (119); every number below
reproduces exactly.

Expected output:

  Part 1 - pre-flight: Day 92's linter vs this run's config
      PASS  Day 84  trials counted
      PASS  Day 85  no lookahead
      PASS  Day 86  first-touch labels
      PASS  Day 87  walk-forward split
      PASS  Day 88  costs charged
      PASS  Day 90  sizing capped
      PASS  Day 91  kill switch armed
      PASS  Day 91  journal on
      -> 0 of 8 checks FAIL - cleared for paper
         trading (Day 91: paper first, always). Nothing below runs
         on a config that grades itself.
  Part 2 - synthetic ACME (Day 88's process, seed 119)
    days 2016  price 98.08 -> 247.83  days in high-vol regime: 953
    planted reversion KAPPA = 0.10 -> measured lag-1 autocorrelation of returns: -0.082
    weaker than Day 88's draw (-0.127): the signal itself has weather.
  Part 3 - features (Day 85) + triple-barrier labels (Day 86)
    samples 1987  features ['ret_1d', 'mom_5d', 'mom_20d', 'vol_5d', 'up_frac_14d']
    barriers +/- 1.0 x sigma_20 x sqrt(10), vertical barrier 10 days
    label balance: up-first 0.555  down/time 0.445  mean holding 8.0 days
  Part 4 - walk-forward (Day 87) + logistic regression by hand
    fold 1: train 751 rows (purged)  test days 775-900  accuracy 0.4206  payoff b 1.19
    fold 2: train 878 rows (purged)  test days 901-1026  accuracy 0.4603  payoff b 1.19
    fold 3: train 1000 rows (purged)  test days 1027-1152  accuracy 0.5794  payoff b 1.13
    fold 4: train 1127 rows (purged)  test days 1153-1278  accuracy 0.5714  payoff b 1.14
    fold 5: train 1250 rows (purged)  test days 1279-1404  accuracy 0.5079  payoff b 1.12
    fold 6: train 1376 rows (purged)  test days 1405-1530  accuracy 0.5079  payoff b 1.13
    fold 7: train 1502 rows (purged)  test days 1531-1656  accuracy 0.6190  payoff b 1.16
    fold 8: train 1628 rows (purged)  test days 1657-1782  accuracy 0.3175  payoff b 1.12
    fold 9: train 1754 rows (purged)  test days 1783-1908  accuracy 0.6508  payoff b 1.10
    pooled test accuracy 0.5150  majority-class baseline 0.5088
    final fold's weights (z-scored features):
      ret_1d       -0.012
      mom_5d       -0.092
      mom_20d      -0.173
      vol_5d       -0.022
      up_frac_14d  +0.238
  Part 5 - the harness runs (Days 90/91): journal excerpts
    decisions journaled: 258 (approved 126, clipped 1, rejected 4, kill-switch trips 4)
    first decisions:
      day  775  BUY              APPROVED qty 74 @ 218.12 - p 0.635, stake 16.3% of equity
      day  784  SELL             bracket exit - the label's own exit, trade return +3.04%
      day  785  BUY              APPROVED qty 81 @ 225.92 - p 0.657, stake 18.4% of equity
      day  791  SELL             bracket exit - the label's own exit, trade return -4.32%
      day  792  BUY              APPROVED qty 61 @ 221.48 - p 0.629, stake 13.7% of equity
      day  802  SELL             bracket exit - the label's own exit, trade return -0.94%
    every kill-switch trip:
      day  900  KILL SWITCH SELL day P&L $-514 < -$400 - flattened, entries paused tomorrow
      day  983  KILL SWITCH SELL day P&L $-593 < -$400 - flattened, entries paused tomorrow
      day 1209  KILL SWITCH SELL day P&L $-519 < -$400 - flattened, entries paused tomorrow
      day 1552  KILL SWITCH SELL day P&L $-502 < -$400 - flattened, entries paused tomorrow
  Part 6 - the report card (whatever it says, it prints)
    accuracy vs majority    0.5150 vs 0.5088  edge +0.0062
    trades taken            127  (largest ask 26.9% of equity - the gate clipped it to 25%)
    strategy, net of costs  +2.15%  Sharpe +0.28
    buy-and-hold, same span +8.41%  Sharpe +0.20
    friction paid           $1,478.80  (7 bps/side, every fill, no exceptions)
    kill switch             tripped 4x - final state ARMED
    pre-flight linter       0 of 8 FAIL (Part 1)
    verdict: the machine ran end to end and every honesty
    device in it fired at least once - clip, rejection,
    kill switch. The edge it found is a whisper and
    buy-and-hold still won the span. Both of those are
    the system working. The deliverable was never this
    P&L - it is a pipeline whose numbers you can trust
    on data whose truth you do not control. That is the
    capstone. Tomorrow: the map of where you stand.
"""
import numpy as np

# ---------- the config this run answers to ----------
FINAL_CONFIG = {
    "trials_counted": True,      # one config, declared: this one
    "lookahead_bars": 0,
    "label_mode": "triple_barrier",
    "split": "walk_forward",
    "costs_bps": 7,              # 5 half-spread + 2 slippage, per side
    "kelly_fraction": 0.5,
    "kill_switch": True,
    "journal": True,
}

# ---------- data (Days 83/88: Day 88's generator) ----------
SEED = 119
N_DAYS = 2016                  # ~8 years of trading days
MU = 0.0002                    # daily drift
KAPPA = 0.10                   # planted mean-reversion strength
SIGMA_LO, SIGMA_HI = 0.009, 0.020
P_SWITCH = 0.02                # vol-regime switch probability / day

# ---------- labels (Day 86, as built in Day 88) ----------
HORIZON = 10                   # vertical barrier, days
BARRIER_MULT = 1.0             # barriers at +/- mult * sigma_t * sqrt(H)

# ---------- model (Days 87/88) ----------
TRAIN_DAYS, TEST_DAYS, STEP_DAYS = 756, 126, 126
LR, ITERS = 0.1, 1000
FEATURES = ("ret_1d", "mom_5d", "mom_20d", "vol_5d", "up_frac_14d")

# ---------- sizing (Day 90) + harness (Day 91) ----------
START_CASH = 100_000.0
KELLY_FRACTION = 0.5           # half-Kelly: the edge is an estimate
TARGET_VOL = 0.15              # annualized vol target for sizing
MAX_FRAC = 0.25                # gate cap: 25% of equity per position
HALF_SPREAD = 0.0005           # 5 bps, every fill (Day 91)
SLIPPAGE = 0.0002              # 2 bps adverse, every fill (Day 91)
FRICTION = HALF_SPREAD + SLIPPAGE
KILL_LOSS = 400.0              # daily-loss kill switch, dollars


# ---------- Part 1: the pre-flight linter (Day 92, verbatim) ----------
# One check per rule Phase G taught: (day, name, test, why).
RULES = [
    (84, "trials counted",
     lambda c: c["trials_counted"],
     "uncounted trials: luck never gets priced"),
    (85, "no lookahead",
     lambda c: c["lookahead_bars"] == 0,
     "features must use bars up to t, never t+1"),
    (86, "first-touch labels",
     lambda c: c["label_mode"] == "triple_barrier",
     "fixed-horizon labels grade trades you never took"),
    (87, "walk-forward split",
     lambda c: c["split"] == "walk_forward",
     "a shuffled split grades the model on the future"),
    (88, "costs charged",
     lambda c: c["costs_bps"] > 0,
     "free trades: Day 88's accuracy still lost -25.01%"),
    (90, "sizing capped",
     lambda c: 0 < c["kelly_fraction"] <= 1.0,
     "over full Kelly, one bad run ends the account"),
    (91, "kill switch armed",
     lambda c: c["kill_switch"],
     "something must be able to say no, mid-flight"),
    (91, "journal on",
     lambda c: c["journal"],
     "every decision logged with what the model saw"),
]


def lint(cfg):
    """Day 92's audit, unchanged: grade the config, count fails."""
    fails = 0
    for day, check, test, why in RULES:
        if test(cfg):
            print(f"    PASS  Day {day}  {check}")
        else:
            fails += 1
            print(f"    FAIL  Day {day}  {check} - {why}")
    return fails


print("Part 1 - pre-flight: Day 92's linter vs this run's config")
lint_fails = lint(FINAL_CONFIG)
print(f"    -> {lint_fails} of {len(RULES)} checks FAIL - "
      "cleared for paper")
print("       trading (Day 91: paper first, always). "
      "Nothing below runs")
print("       on a config that grades itself.")
assert lint_fails == 0


# ---------- Part 2: the series, process stated (Day 88) ----------
def generate_series():
    """ACME: drift + two vol regimes + one planted signal."""
    rng = np.random.default_rng(SEED)
    sigma = np.empty(N_DAYS)
    state = 0
    for t in range(N_DAYS):
        if rng.random() < P_SWITCH:
            state = 1 - state
        sigma[t] = SIGMA_HI if state else SIGMA_LO
    eps = rng.standard_normal(N_DAYS)
    r = np.empty(N_DAYS)
    prev = 0.0
    for t in range(N_DAYS):
        r[t] = MU - KAPPA * (prev - MU) + sigma[t] * eps[t]
        prev = r[t]
    return r, 100.0 * np.exp(np.cumsum(r)), sigma


r, price, sigma = generate_series()
print("Part 2 - synthetic ACME (Day 88's process, seed 119)")
print(f"  days {N_DAYS}  price {price[0]:.2f} -> {price[-1]:.2f}  "
      f"days in high-vol regime: {(sigma == SIGMA_HI).sum()}")
ac1 = np.corrcoef(r[:-1], r[1:])[0, 1]
print(f"  planted reversion KAPPA = {KAPPA:.2f} -> measured lag-1 "
      f"autocorrelation of returns: {ac1:+.3f}")
print("  weaker than Day 88's draw (-0.127): the signal itself "
      "has weather.")


# ---------- Part 3: features (Day 85) + labels (Day 86) ----------
def build_features(r):
    """One row per day t (t >= 19), data up to t only (Day 85).
    The same five features as Days 32 and 88."""
    rows = []
    for t in range(19, len(r)):
        rows.append([
            r[t],                                   # ret_1d
            r[t - 4:t + 1].mean(),                  # mom_5d
            r[t - 19:t + 1].mean(),                 # mom_20d
            r[t - 4:t + 1].std(),                   # vol_5d
            (r[t - 13:t + 1] > 0).mean(),           # up_frac_14d
        ])
    return np.array(rows), np.arange(19, len(r))


def triple_barrier(price, r, t):
    """Day 86: from close t, first barrier touched wins.
    Returns (label 1/0, exit day index, realized simple
    return). The exits double as the trade's resting
    bracket order - set at entry, never revised."""
    sig = r[t - 19:t + 1].std()
    bar = BARRIER_MULT * sig * np.sqrt(HORIZON)
    entry = price[t]
    for s in range(1, HORIZON + 1):
        cum = price[t + s] / entry - 1.0
        if cum >= bar:
            return 1, t + s, cum
        if cum <= -bar:
            return 0, t + s, cum
    cum = price[t + HORIZON] / entry - 1.0
    return (1 if cum > 0 else 0), t + HORIZON, cum


X, days = build_features(r)
keep = days + HORIZON < N_DAYS
X, days = X[keep], days[keep]
labels, exits, rets = [], [], []
for t in days:
    lab, ex, ret = triple_barrier(price, r, int(t))
    labels.append(lab)
    exits.append(ex)
    rets.append(ret)
y = np.array(labels)
exit_day = np.array(exits)
exit_ret = np.array(rets)

print("Part 3 - features (Day 85) + triple-barrier labels (Day 86)")
print(f"  samples {len(y)}  features {list(FEATURES)}")
print(f"  barriers +/- {BARRIER_MULT:.1f} x sigma_20 x sqrt({HORIZON}), "
      f"vertical barrier {HORIZON} days")
print(f"  label balance: up-first {y.mean():.3f}  down/time "
      f"{1 - y.mean():.3f}  mean holding {(exit_day - days).mean():.1f} days")


# ---------- Part 4: walk-forward + the baseline (Days 87/88) ----------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train_logreg(Xtr, ytr):
    """Full-batch GD, fixed lr/iters/zero init - no knobs turned."""
    w = np.zeros(Xtr.shape[1])
    b = 0.0
    for _ in range(ITERS):
        p = sigmoid(Xtr @ w + b)
        g = p - ytr
        w -= LR * (Xtr.T @ g) / len(ytr)
        b -= LR * g.mean()
    return w, b


def walk_forward():
    """Expanding folds, purged (Day 87); per fold, also the
    payoff ratio b of its training labels (Day 90's input)."""
    folds = []
    test_lo, fold = TRAIN_DAYS, 0
    while test_lo + TEST_DAYS <= len(y):
        test_hi = test_lo + TEST_DAYS
        tr = np.arange(0, test_lo)
        tr = tr[exit_day[tr] < days[test_lo]]   # purge label-peekers
        mu_x, sd_x = X[tr].mean(0), X[tr].std(0) + 1e-12
        w, b0 = train_logreg((X[tr] - mu_x) / sd_x, y[tr])
        p = sigmoid(((X[test_lo:test_hi] - mu_x) / sd_x) @ w + b0)
        wins = exit_ret[tr][y[tr] == 1]
        losses = exit_ret[tr][y[tr] == 0]
        folds.append({
            "fold": fold + 1, "ntr": len(tr),
            "acc": ((p >= 0.5).astype(int)
                    == y[test_lo:test_hi]).mean(),
            "lo": test_lo, "hi": test_hi, "probs": p, "w": w,
            "b": wins.mean() / abs(losses.mean()),
        })
        fold += 1
        test_lo += STEP_DAYS
    return folds


folds = walk_forward()
print("Part 4 - walk-forward (Day 87) + logistic regression by hand")
for f in folds:
    print(f"  fold {f['fold']}: train {f['ntr']} rows (purged)  "
          f"test days {days[f['lo']]}-{days[f['hi'] - 1]}  "
          f"accuracy {f['acc']:.4f}  payoff b {f['b']:.2f}")
true_all = np.concatenate([y[f["lo"]:f["hi"]] for f in folds])
pred_all = np.concatenate([(f["probs"] >= 0.5).astype(int)
                             for f in folds])
pooled = (pred_all == true_all).mean()
baseline = max(true_all.mean(), 1 - true_all.mean())
print(f"  pooled test accuracy {pooled:.4f}  "
      f"majority-class baseline {baseline:.4f}")
print("  final fold's weights (z-scored features):")
for name, wi in zip(FEATURES, folds[-1]["w"]):
    print(f"    {name:<12} {wi:+.3f}")


# ---------- Part 5: sizing (Day 90) + the harness (Day 91) ----------
def size_position(p, b, sig_t):
    """Day 90: f* = p - (1-p)/b, halved (the edge is an
    estimate), then vol-targeted: stake scales down as
    trailing vol rises above the 15% target. Returns
    (f_star, stake fraction before the gate's cap)."""
    f_star = min(max(p - (1.0 - p) / b, 0.0), 1.0)
    vol_mult = min(1.0, TARGET_VOL / (sig_t * np.sqrt(252)))
    return f_star, KELLY_FRACTION * f_star * vol_mult


class RiskGate:
    """Day 91's gate, denominated in equity fractions.
    Deterministic admission control - no model in here."""

    def review(self, frac, halted):
        if halted:
            return "REJECTED", 0.0, "kill switch: cooling down"
        if frac > MAX_FRAC:
            return "APPROVED-CLIPPED", MAX_FRAC, \
                f"requested {frac:.1%} > cap {MAX_FRAC:.0%}"
        return "APPROVED", frac, "within limits"


def run_harness():
    """Paper trading over the walk-forward test span."""
    signals = {}
    for f in folds:
        for j in range(f["lo"], f["hi"]):
            signals[int(days[j])] = (float(f["probs"][j - f["lo"]]),
                                     int(exit_day[j]), f["b"],
                                     float(exit_ret[j]))
    start = int(days[folds[0]["lo"]])
    end = int(max(exit_day[folds[0]["lo"]:folds[-1]["hi"]]))
    gate = RiskGate()
    cash, pos, entry = START_CASH, 0, None
    last_exit, cooldown = -1, -1
    friction_paid, trips, max_stake = 0.0, 0, 0.0
    journal = []
    counts = {"APPROVED": 0, "APPROVED-CLIPPED": 0,
              "REJECTED": 0, "KILL": 0}
    equity_curve, prev_eq = [], START_CASH

    def log(day, event, detail):
        journal.append((day, event, detail))

    for t in range(start, end + 1):
        mid = price[t]
        if pos > 0 and t == entry["exit_day"]:      # bracket exit
            fill = mid * (1 - FRICTION)
            cash += pos * fill
            friction_paid += pos * mid * FRICTION
            log(t, "SELL", f"bracket exit - the label's own exit, "
                           f"trade return {entry['ret']:+.2%}")
            pos, last_exit, entry = 0, t, None
        day_pnl = (cash + pos * mid) - prev_eq      # kill switch
        if pos > 0 and day_pnl < -KILL_LOSS:
            fill = mid * (1 - FRICTION)
            cash += pos * fill
            friction_paid += pos * mid * FRICTION
            log(t, "KILL SWITCH SELL",
                f"day P&L ${day_pnl:,.0f} < -${KILL_LOSS:,.0f} - "
                "flattened, entries paused tomorrow")
            counts["KILL"] += 1
            trips += 1
            pos, last_exit, entry = 0, t, None
            cooldown = t + 1
        if pos == 0 and t > last_exit and t in signals:
            p, exit_sched, b, lab_ret = signals[t]
            if p >= 0.5:
                f_star, frac = size_position(
                    p, b, r[t - 19:t + 1].std())
                if frac <= 0.0:
                    log(t, "NO BET", f"Kelly f* = {f_star:.3f} at "
                                     f"p = {p:.3f}, b = {b:.2f}")
                else:
                    verdict, ok, _ = gate.review(frac,
                                                 t == cooldown)
                    counts[verdict] += 1
                    max_stake = max(max_stake, frac)
                    if ok > 0.0:
                        qty = int(ok * cash / mid)
                        fill = mid * (1 + FRICTION)
                        cash -= qty * fill
                        friction_paid += qty * mid * FRICTION
                        pos = qty
                        entry = {"exit_day": exit_sched,
                                 "ret": lab_ret}
                        log(t, "BUY", f"{verdict} qty {qty} @ "
                                      f"{fill:.2f} - p {p:.3f}, "
                                      f"stake {ok:.1%} of equity")
                    else:
                        log(t, "BUY (no fill)", f"{verdict} - "
                                                "kill-switch cooldown")
        equity_curve.append(cash + pos * mid)
        prev_eq = equity_curve[-1]
    return {
        "start": start, "end": end, "journal": journal,
        "counts": counts, "trips": trips,
        "friction": friction_paid, "max_stake": max_stake,
        "equity": np.array(equity_curve),
        "trades": counts["APPROVED"] + counts["APPROVED-CLIPPED"],
    }


res = run_harness()
print("Part 5 - the harness runs (Days 90/91): journal excerpts")
print(f"  decisions journaled: {len(res['journal'])} "
      f"(approved {res['counts']['APPROVED']}, "
      f"clipped {res['counts']['APPROVED-CLIPPED']}, "
      f"rejected {res['counts']['REJECTED']}, "
      f"kill-switch trips {res['counts']['KILL']})")
print("  first decisions:")
for d, ev, det in res["journal"][:6]:
    print(f"    day {d:>4}  {ev:<16} {det}")
print("  every kill-switch trip:")
for d, ev, det in res["journal"]:
    if ev.startswith("KILL"):
        print(f"    day {d:>4}  {ev:<16} {det}")


# ---------- Part 6: the report card ----------
def sharpe(daily):
    return daily.mean() / (daily.std() + 1e-12) * np.sqrt(252)


eq = res["equity"]
strat_ret = eq[-1] / START_CASH - 1.0
strat_sh = sharpe(np.diff(eq) / eq[:-1])
bh_seg = price[res["start"]:res["end"] + 1]
bh_ret = bh_seg[-1] / bh_seg[0] - 1.0
bh_sh = sharpe(bh_seg[1:] / bh_seg[:-1] - 1.0)
kill_state = ("armed, never tripped" if res["trips"] == 0 else
              f"tripped {res['trips']}x - final state ARMED")

print("Part 6 - the report card (whatever it says, it prints)")
print(f"  accuracy vs majority    {pooled:.4f} vs {baseline:.4f}  "
      f"edge {pooled - baseline:+.4f}")
print(f"  trades taken            {res['trades']}  "
      f"(largest ask {res['max_stake']:.1%} of equity - "
      f"the gate clipped it to {MAX_FRAC:.0%})")
print(f"  strategy, net of costs  {strat_ret * 100:+.2f}%  "
      f"Sharpe {strat_sh:+.2f}")
print(f"  buy-and-hold, same span {bh_ret * 100:+.2f}%  "
      f"Sharpe {bh_sh:+.2f}")
print(f"  friction paid           ${res['friction']:,.2f}  "
      f"({FINAL_CONFIG['costs_bps']} bps/side, every fill, "
      "no exceptions)")
print(f"  kill switch             {kill_state}")
print(f"  pre-flight linter       {lint_fails} of {len(RULES)} "
      "FAIL (Part 1)")
print("  verdict: the machine ran end to end and every honesty")
print("  device in it fired at least once - clip, rejection,")
print("  kill switch. The edge it found is a whisper and")
print("  buy-and-hold still won the span. Both of those are")
print("  the system working. The deliverable was never this")
print("  P&L - it is a pipeline whose numbers you can trust")
print("  on data whose truth you do not control. That is the")
print("  capstone. Tomorrow: the map of where you stand.")
