"""Day 10 - Distributions: The Vocabulary of Uncertainty.

The two distributions every AI scientist reaches for:
  - the NORMAL (gaussian) for continuous quantities  ->  N(mu, sigma)
  - the CATEGORICAL for discrete categories           ->  a prob vector summing to 1

Tonight: draw 100k samples from each in torch and watch the empirical
numbers land on the theory. That convergence IS the idea: a distribution
is a machine that answers "how likely is each outcome?"

Runs on MacBook: ~/ai-lab venv, torch CPU/MPS. No downloads, no paid APIs.
"""
import torch

torch.manual_seed(42)

# ---- 1. The normal: daily returns of a toy stock ----
# mu = +0.08% average daily drift, sigma = 1.2% daily vol (realized vol, trading-style)
mu, sigma = 0.0008, 0.012
returns = torch.normal(mean=mu, std=sigma, size=(100_000,))

print("normal N(mu=0.0008, sigma=0.012), 100k draws")
print("empirical mean:", round(returns.mean().item(), 6), "(theory 0.0008)")
print("empirical std :", round(returns.std().item(), 6), "(theory 0.012)")

# The 68-95-99.7 rule, checked empirically
for k, theory in ((1, 0.683), (2, 0.955), (3, 0.997)):
    frac = ((returns - mu).abs() <= k * sigma).float().mean().item()
    print(f"within +/-{k} sigma: {frac:.4f}  (theory {theory:.3f})")

# ---- 2. The categorical: market regimes ----
# Three possible regimes, probabilities summing to 1. This is exactly what
# softmax outputs at the head of a classifier - Day 14 will connect it.
probs = torch.tensor([0.60, 0.30, 0.10])  # calm, choppy, crisis
regime = torch.distributions.Categorical(probs)
samples = regime.sample((100_000,))
names = ["calm", "choppy", "crisis"]
print("\ncategorical over regimes, 100k draws")
for i, name in enumerate(names):
    emp = (samples == i).float().mean().item()
    print(f"{name:>6}: empirical {emp:.4f}  (theory {probs[i].item():.2f})")

# ---- 3. log_prob: the score models actually optimize ----
# Day 11 will show that fitting a distribution = maximizing the sum of these.
dist = torch.distributions.Normal(mu, sigma)
x = torch.tensor([mu])
print("\nlog_prob of the mean:", round(dist.log_prob(x).item(), 4))
