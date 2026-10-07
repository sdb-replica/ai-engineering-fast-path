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
