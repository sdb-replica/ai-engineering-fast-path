"""Day 55 - RLHF 2: DPO, the Simpler Modern Alternative.

Day 54's RLHF pipeline had three stages after SFT: collect
preference pairs, train a separate reward model (RM) on them,
then run an RL loop (PPO) that samples from the policy, scores
the samples with the RM, and updates - with a KL penalty
leashing the policy to the frozen reference model pi_ref.

DPO (Direct Preference Optimization) deletes two of the three.
The key identity: in that KL-leashed objective, the OPTIMAL
policy already contains the reward -

    r(x, y) = beta * log( pi(y|x) / pi_ref(y|x) )

(read: a completion's reward IS how much more likely the
policy makes it than the reference did, scaled by beta; a
per-prompt constant cancels inside every pair). So instead of
training an RM to approximate r and then RL-ing against it,
plug the implicit reward straight into the preference model
from Day 54 - P(chosen beats rejected) = sigmoid(r_w - r_l) -
and train the POLICY on the pairs with a classification loss:

    L = -log sigmoid( beta * [ (log pi(y_w) - log pi_ref(y_w))
                             - (log pi(y_l) - log pi_ref(y_l)) ] )

No reward model. No sampling. No RL loop. One loss, whose
gradient widens the chosen-vs-rejected log-prob gap, measured
relative to the frozen reference.

  Part 1  Score one pair by hand. A 4-word vocab, one prompt,
          2-token completions, policy and reference as explicit
          probability tables. The policy starts AT the reference,
          so both implicit rewards are exactly 0 and the loss is
          log 2 - a coin flip - even though the raw log-prob gap
          already favors the chosen completion.
  Part 2  Real gradient steps on the policy's 8 logits (SGD,
          lr=0.5, 10 steps, beta=1.0): the loss falls, the gap
          widens, the implicit rewards split + / -, and the
          chosen tokens' probabilities climb while the rejected
          tokens' sink. Autograd computes the gradient; note it
          never touches the frozen reference's numbers.
  Part 3  Beta is the leash. Same pair, same 10 steps, only
          beta changes: beta=1.0 nearly satisfies the loss,
          beta=0.1 barely dents it - because beta is the
          exchange rate between drift and reward. Reaching
          loss 0.10 takes a reward margin of 2.2522 at beta=1.0
          but 22.5217 at beta=0.1: small beta leaves the loss
          hungry, so training would drag the policy far from
          the reference before it is satisfied.

Honest limits: DPO still eats preference pairs (the data cost
is unchanged) and it can overfit them - the loss is happiest
when the gap explodes, not when the chosen answer is good.

Runs on the ~/ai-lab venv: torch only (free), CPU on purpose
(8 parameters), no CUDA, no paid APIs, no downloads. Every
probability is set by hand - no RNG anywhere in this lesson.

Expected output:

  Part 1 - one preference pair, scored by hand (beta = 1.0)
    prompt: "The soup was" - completions are 2 tokens from a 4-word vocab
    ref probs, position 1: warm 0.4000, fresh 0.3000, cold 0.2000, stale 0.1000
    ref probs, position 2: warm 0.2500, fresh 0.4500, cold 0.2000, stale 0.1000
    chosen   "warm fresh":  log p_ref = log(0.40) + log(0.45) = -1.7148
    rejected "cold stale":  log p_ref = log(0.20) + log(0.10) = -3.9120
    policy starts AT the reference (same table), so both implicit rewards are 0:
      r = beta * (log p_policy - log p_ref):  chosen 0.0000, rejected 0.0000
      DPO margin = r_chosen - r_rejected = 0.0000
      DPO loss = -log sigmoid(0) = 0.6931  (= log 2 - a coin flip)
  Part 2 - train the policy on the pair (SGD lr=0.5, 10 steps, beta=1.0)
    step  0: loss 0.6931  logp chosen -1.7148  logp rejected -3.9120  gap 2.1972  r_chosen 0.0000  r_rejected 0.0000
    step  1: loss 0.3133  logp chosen -1.3823  logp rejected -4.5796  gap 3.1972  r_chosen 0.3325  r_rejected -0.6675
    step  2: loss 0.1946  logp chosen -1.2272  logp rejected -4.9623  gap 3.7351  r_chosen 0.4876  r_rejected -1.0503
    step  3: loss 0.1405  logp chosen -1.1336  logp rejected -5.2224  gap 4.0888  r_chosen 0.5812  r_rejected -1.3104
    step  4: loss 0.1098  logp chosen -1.0683  logp rejected -5.4193  gap 4.3509  r_chosen 0.6465  r_rejected -1.5072
    step  5: loss 0.0901  logp chosen -1.0190  logp rejected -5.5778  gap 4.5589  r_chosen 0.6958  r_rejected -1.6658
    step  6: loss 0.0764  logp chosen -0.9796  logp rejected -5.7108  gap 4.7312  r_chosen 0.7352  r_rejected -1.7988
    step  7: loss 0.0663  logp chosen -0.9471  logp rejected -5.8253  gap 4.8782  r_chosen 0.7677  r_rejected -1.9133
    step  8: loss 0.0585  logp chosen -0.9196  logp rejected -5.9260  gap 5.0064  r_chosen 0.7952  r_rejected -2.0139
    step  9: loss 0.0524  logp chosen -0.8957  logp rejected -6.0158  gap 5.1201  r_chosen 0.8191  r_rejected -2.1038
    step 10: loss 0.0474  logp chosen -0.8748  logp rejected -6.0970  gap 5.2221  r_chosen 0.8400  r_rejected -2.1850
    token probs after training (before -> after):
      P(warm | pos 1)   0.4000 -> 0.6331   (chosen token, pushed up)
      P(fresh | pos 2)  0.4500 -> 0.6586   (chosen token, pushed up)
      P(cold | pos 1)   0.2000 -> 0.0698   (rejected token, pushed down)
      P(stale | pos 2)  0.1000 -> 0.0323   (rejected token, pushed down)
    no reward model was trained, nothing was sampled - one loss on one pair
  Part 3 - beta is the leash (same pair, same 10 steps, only beta changes)
    beta = 1.0:  final loss 0.0474, reward margin 3.0249, P(warm | pos 1) 0.6331
    beta = 0.1:  final loss 0.6455, reward margin 0.0978, P(warm | pos 1) 0.4785
    margin needed to reach loss 0.10: beta=1.0 needs 2.2522, beta=0.1 needs 22.5217
    verdict: beta is the exchange rate between drift and reward - small beta
    leaves the loss hungry, so training would drag the policy far from ref
"""
import math

import torch
import torch.nn.functional as F

VOCAB = ["warm", "fresh", "cold", "stale"]
# The reference model pi_ref (the frozen SFT model): an explicit
# probability table, one row per completion position, for the
# prompt "The soup was". Rows sum to 1.
REF_PROBS = torch.tensor([[0.40, 0.30, 0.20, 0.10],     # position 1
                          [0.25, 0.45, 0.20, 0.10]])    # position 2
CHOSEN = torch.tensor([0, 1])      # "warm fresh"  (the human pick)
REJECTED = torch.tensor([2, 3])    # "cold stale"  (the human pass)
POS = torch.arange(2)
BETA = 1.0
LR = 0.5
STEPS = 10


def f4(v):
    """Format to 4 decimals, normalizing a signed zero."""
    s = f"{v:.4f}"
    return "0.0000" if s == "-0.0000" else s


def seq_logp_from_probs(probs, tokens):
    """Sequence log-prob = sum of per-token log probs (Day 47's
    chain rule, on a table instead of a model)."""
    return torch.log(probs[POS, tokens]).sum()


def seq_logp_from_logits(logits, tokens):
    return F.log_softmax(logits, dim=-1)[POS, tokens].sum()


LP_REF_W = seq_logp_from_probs(REF_PROBS, CHOSEN)
LP_REF_L = seq_logp_from_probs(REF_PROBS, REJECTED)


def dpo_stats(logits, beta):
    """Implicit rewards, margin and the DPO loss for the pair."""
    lp_w = seq_logp_from_logits(logits, CHOSEN)
    lp_l = seq_logp_from_logits(logits, REJECTED)
    r_w = beta * (lp_w - LP_REF_W)          # the identity, applied
    r_l = beta * (lp_l - LP_REF_L)
    margin = r_w - r_l
    loss = -F.logsigmoid(margin)            # classification loss
    return lp_w, lp_l, r_w, r_l, margin, loss


def run_dpo(beta, verbose=False):
    """Train ONLY the policy logits on the pair. The reference
    table is a constant - it is read, never updated."""
    logits = torch.log(REF_PROBS).clone().requires_grad_(True)
    opt = torch.optim.SGD([logits], lr=LR)
    last = None
    for step in range(STEPS + 1):
        lp_w, lp_l, r_w, r_l, margin, loss = dpo_stats(logits, beta)
        last = (loss.item(), margin.item())
        if verbose:
            print(f"  step {step:>2}: loss {f4(loss.item())}  "
                  f"logp chosen {f4(lp_w.item())}  "
                  f"logp rejected {f4(lp_l.item())}  "
                  f"gap {f4((lp_w - lp_l).item())}  "
                  f"r_chosen {f4(r_w.item())}  "
                  f"r_rejected {f4(r_l.item())}")
        if step == STEPS:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
    return last, probs


# ---------- Part 1: score one pair by hand ----------
print("Part 1 - one preference pair, scored by hand (beta = 1.0)")
print('  prompt: "The soup was" - completions are 2 tokens from '
      "a 4-word vocab")
for t in range(2):
    table = ", ".join(f"{w} {f4(p)}"
                      for w, p in zip(VOCAB, REF_PROBS[t].tolist()))
    print(f"  ref probs, position {t + 1}: {table}")
print(f"  chosen   \"warm fresh\":  log p_ref = log(0.40) + "
      f"log(0.45) = {f4(LP_REF_W.item())}")
print(f"  rejected \"cold stale\":  log p_ref = log(0.20) + "
      f"log(0.10) = {f4(LP_REF_L.item())}")
# Policy starts AT the reference: same table, so log p_policy
# equals log p_ref for both completions and the rewards are 0.
r_w0 = BETA * (LP_REF_W - LP_REF_W)
r_l0 = BETA * (LP_REF_L - LP_REF_L)
margin0 = r_w0 - r_l0
loss0 = -F.logsigmoid(margin0)
print("  policy starts AT the reference (same table), so both "
      "implicit rewards are 0:")
print(f"    r = beta * (log p_policy - log p_ref):  "
      f"chosen {f4(r_w0.item())}, rejected {f4(r_l0.item())}")
print(f"    DPO margin = r_chosen - r_rejected = {f4(margin0.item())}")
print(f"    DPO loss = -log sigmoid(0) = {f4(loss0.item())}  "
      "(= log 2 - a coin flip)")

# ---------- Part 2: train the policy directly on the pair ----------
print("Part 2 - train the policy on the pair "
      "(SGD lr=0.5, 10 steps, beta=1.0)")
(_, _), probs = run_dpo(BETA, verbose=True)
print("  token probs after training (before -> after):")
for pos, tok, tag in ((0, 0, "chosen token, pushed up"),
                      (1, 1, "chosen token, pushed up"),
                      (0, 2, "rejected token, pushed down"),
                      (1, 3, "rejected token, pushed down")):
    print(f"    P({VOCAB[tok]} | pos {pos + 1})   "
          f"{f4(REF_PROBS[pos, tok].item())} -> "
          f"{f4(probs[pos, tok].item())}   ({tag})")
print("  no reward model was trained, nothing was sampled - "
      "one loss on one pair")

# ---------- Part 3: beta is the leash ----------
print("Part 3 - beta is the leash (same pair, same 10 steps, "
      "only beta changes)")
(loss_b1, margin_b1), probs_b1 = run_dpo(1.0)
(loss_b01, margin_b01), probs_b01 = run_dpo(0.1)
print(f"  beta = 1.0:  final loss {f4(loss_b1)}, "
      f"reward margin {f4(margin_b1)}, "
      f"P(warm | pos 1) {f4(probs_b1[0, 0].item())}")
print(f"  beta = 0.1:  final loss {f4(loss_b01)}, "
      f"reward margin {f4(margin_b01)}, "
      f"P(warm | pos 1) {f4(probs_b01[0, 0].item())}")
# Solve -log sigmoid(u) = 0.10 for u, then margin = u / beta.
u_need = math.log(math.exp(-0.1) / (1 - math.exp(-0.1)))
print(f"  margin needed to reach loss 0.10: beta=1.0 needs "
      f"{f4(u_need / 1.0)}, beta=0.1 needs {f4(u_need / 0.1)}")
print("  verdict: beta is the exchange rate between drift and "
      "reward - small beta")
print("  leaves the loss hungry, so training would drag the "
      "policy far from ref")
