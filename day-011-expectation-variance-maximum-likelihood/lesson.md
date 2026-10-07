# Day 11 — Expectation, Variance & Maximum Likelihood

**Tonight you'll be able to:** say what expectation and variance actually measure, estimate both from samples — and show that fitting a distribution to data (maximum likelihood) is exactly why minimizing MSE works.

## The one idea

Day 10 gave you the vocabulary — distributions. Tonight, the two numbers that summarize any distribution, and the one-paragraph idea that turns raw data into a fitted one. That's the engine behind every training loop you've run so far.

**Expectation, E[X]: the long-run average.** The center of mass of the distribution. For discrete outcomes it's just `Σ x·P(x)` — each outcome times its probability. Trading toy: a position paying +$3 with probability 0.4 and −$1 with probability 0.6 has E = 3·0.4 + (−1)·0.6 = **+$0.60 per trade**. Expectation is a property of the *distribution*; the sample average *estimates* it — and as you draw more samples, the estimate converges to E[X]. That convergence is the law of large numbers, and it's why backtests with 10 trades mean nothing and backtests with 10,000 mean something.

**Variance: the average squared distance from the mean.** `Var(X) = E[(X − μ)²]` — how far a typical draw sits from center, measured in squared units. Why squared? Signed distances always average to exactly zero (the mean is defined by that), so they say nothing. And absolute value has a kink at zero — *not differentiable*, and Day 5 taught you smooth beats kinked. The standard deviation σ = √Var brings it back to original units — that's your daily vol. Two distributions can share a mean and be utterly different beasts; variance tells them apart.

**Maximum likelihood, in one paragraph.** The data x₁..xₙ are fixed; the parameters θ are free. The **likelihood** L(θ) = ∏ p(xᵢ|θ) answers one question: *how probable does this model say our actual data is?* Pick the θ that maximizes it. Take logs and the product becomes a sum: log L = Σ log p(xᵢ|θ) — those summands are Day 10's `log_prob` values. Now the gaussian with σ fixed: log p(x|μ) = −(x−μ)²/2σ² − log(σ√2π). Maximizing that over μ is the same as minimizing Σ(x−μ)² — and setting the derivative to zero gives **μ̂ = mean(x)**. Fitting a gaussian to data, solved in one line.

And here's the payoff you already earned: **Days 5–9 minimized MSE — Σ(y−ŷ)².** That was never just a convenient loss. Regression with gaussian noise *is* this paragraph: minimizing squared error is exactly maximizing the likelihood of the targets. Every training loop you've run was quietly fitting a distribution to data. MSE finally has a *why*.

One bookkeeping bite: `torch.var` divides by n−1 by default (Bessel's correction — an unbiased *estimate* of the population variance). The theory and MLE use plain n. Pass `unbiased=False` when you want the textbook number.

## Analogy

**Trading:** E[X] is your backtest's average per-trade P&L — the number the equity curve converges to. Variance is the wobble around it. E = +$0.60 with tiny variance: print money. E = +$0.60 with huge variance: you need position sizing to survive the drawdowns between the wins — Day 90 will formalize exactly that with Kelly. **Systems:** you never observe the true latency distribution — you estimate E and the tail from samples. Your dashboard's mean converging as traffic grows *is* the law of large numbers; and variance is why the dashboard shows p99, not just the mean. One number never describes a distribution — that's a two-number job, every time.

## See it

![Day 11 diagram: E[X] and Var(X) on ten daily returns, plus the log-likelihood parabola whose peak is the MLE](diagram.svg)

*Top: E[X] is where the data balances (μ̂ = 0.23); Var(X) averages the squared distances from it. Bottom: the log-likelihood over candidate μ — a perfect upside-down parabola whose peak is the MLE.*

## Code it (~30 min)

Paste into `~/ai-lab` and run. Three moves: watch a sample average converge to E[X], compute variance by hand and match `torch.var`, then grid-search the μ that maximizes the log-likelihood — and watch it land exactly on the MSE minimizer.

```python
"""Day 11 - Expectation, Variance & Maximum Likelihood.

Tonight's idea in three moves:
  1. E[X] is the long-run average -- the center of mass of a distribution.
  2. Var(X) = E[(X - mu)^2] is the average squared distance from the mean.
  3. MLE: fitting a distribution = choosing the parameters that make the
     observed data most probable. For a gaussian, that choice IS least
     squares -- which is why every MSE training loop from Days 5-9 worked.

Runs on MacBook: ~/ai-lab venv, torch CPU/MPS. No downloads, no paid APIs.
"""
import torch

torch.manual_seed(11)

# ---- 1. Expectation: the long-run average ----
# A trading toy: +$3 with prob 0.4, -$1 with prob 0.6. E[X] = sum of x*P(x).
pays = torch.tensor([3.0, -1.0])
probs = torch.tensor([0.4, 0.6])
ev = (pays * probs).sum()
print("E[trade] = 3*0.4 + (-1)*0.6 =", round(ev.item(), 4), "(theory 0.6)")

draws = torch.distributions.Categorical(probs).sample((200_000,))
avg = pays[draws].mean().item()
print("avg over 200k sampled trades:", round(avg, 4), "-> converging to E[X]")

# ---- 2. Variance: average squared distance from the mean ----
# Daily returns ~ N(mu=0.0008, sigma=0.012), same toy as Day 10.
x = torch.normal(mean=0.0008, std=0.012, size=(200_000,))
mu_hat = x.mean()
print("\nsample mean (estimates E[X]):", round(mu_hat.item(), 6))
print("torch.var  (unbiased, n-1)  :", round(x.var().item(), 8))
print("torch.var  (unbiased=False) :", round(x.var(unbiased=False).item(), 8),
      "(theory 0.000144)")
manual = ((x - mu_hat) ** 2).mean()   # Var = E[(x - mu)^2], by hand
print("E[(x-mu)^2] by hand          :", round(manual.item(), 8))
print("std = sqrt(var)              :", round(x.std(unbiased=False).item(), 6),
      "(theory 0.012)")

# ---- 3. Maximum likelihood in one paragraph ----
# Ten observed daily returns (percent). Assume N(mu, sigma=1.0), sigma known.
# likelihood L(mu) = prod p(x_i | mu). Maximize the LOG instead:
#   log L(mu) = sum of log_probs  (Day 10's teaser, delivered)
# For a gaussian: log p(x|mu) = -(x-mu)^2/2 - log(sqrt(2*pi)), so
#   maximizing log-likelihood  <=>  minimizing sum (x-mu)^2  <=>  mu* = mean(x).
data = torch.tensor([1.2, -0.8, 0.5, 2.1, -1.5, 0.3, -0.2, 1.0, 0.8, -1.1])
grid = torch.linspace(data.min() - 1.0, data.max() + 1.0, 401)
step = (grid[1] - grid[0]).item()
ll = torch.distributions.Normal(grid[:, None], 1.0).log_prob(data).sum(dim=1)
mse = ((grid[:, None] - data) ** 2).mean(dim=1)
best = grid[ll.argmax()]
print("\ngrid search over mu, step", round(step, 4))
print("MLE mu*        :", round(best.item(), 4))
print("sample mean    :", round(data.mean().item(), 4))
print("argmin-MSE mu  :", round(grid[mse.argmin()].item(), 4))
print("same mu (within one grid step):",
      bool((best - data.mean()).abs() < step))
print("log-likelihood at mu* :", round(ll.max().item(), 4))
print("log-likelihood at mu=0:",
      round(torch.distributions.Normal(0.0, 1.0).log_prob(data).sum().item(), 4))
```

**Expected output:**

```
E[trade] = 3*0.4 + (-1)*0.6 = 0.6 (theory 0.6)
avg over 200k sampled trades: 0.6041 -> converging to E[X]

sample mean (estimates E[X]): 0.000812
torch.var  (unbiased, n-1)  : 0.00014422
torch.var  (unbiased=False) : 0.00014422 (theory 0.000144)
E[(x-mu)^2] by hand          : 0.00014422
std = sqrt(var)              : 0.012009 (theory 0.012)

grid search over mu, step 0.014
MLE mu*        : 0.23
sample mean    : 0.23
argmin-MSE mu  : 0.23
same mu (within one grid step): True
log-likelihood at mu* : -14.9099
log-likelihood at mu=0: -15.1744
```

*Verified by float32 numpy emulation of the exact torch ops (torch isn't on this VM). On your MacBook the sampled-trade average and last digits may wobble a touch — the convergence is the point.*

## Today's win

Run `code.py`: (1) the average of 200k sampled trades lands on E = +$0.60; (2) your hand-computed E[(x−μ)²] matches `torch.var(unbiased=False)` digit for digit; (3) a 401-point grid search finds the MLE μ̂ = 0.23 — exactly the sample mean, exactly the MSE minimizer. Two views, one number. **Done state:** you can explain, out loud, why minimizing MSE is maximum likelihood under gaussian noise.

## Short on time? 20-minute version

Read **The one idea** and study the diagram: E[X] is the long-run average, Var(X) is the average squared distance from it, and MLE says fit parameters by maximizing Σ log p(xᵢ|θ) — which for a gaussian makes μ̂ the sample mean and turns least squares into a theorem. Skip the code tonight; run it tomorrow before the next lesson.

## Tomorrow's teaser

Tomorrow: the one operation all of deep learning is built from — multiply, add, repeat. Everything since Day 1 has been pointing at it.

---
*Day 11 of 100 · Show up. Never miss twice. 1% better.*
