"""Day 56 - RLVR: Reinforcement Learning With Verifiable Rewards.

Days 54-55 taught rewards that come from JUDGMENT: RLHF trains
a reward model on human preferences, DPO learns from preferred
vs rejected pairs. A judge can be flattered, gamed, Goodharted.
Tonight the reward comes from a CHECKER instead:

    reward = 1 if the answer verifies, 0 if it doesn't.

Math answer exact-matches, code passes its unit tests, a proof
verifies - no learned judge in the loop, nothing to sweet-talk.
Sample K attempts per prompt, score each 1/0, and push up the
probability of the ones that pass. The baseline is the group
itself: advantage = reward - group mean (GRPO's trick - the K
samples replace the critic network entirely).

  Part 1  The checker IS the reward. Six two-digit additions,
          four candidate answers each, a hand-set softmax
          policy. The checker stores no labels - it recomputes
          a + b and compares. pass@1 is 0.2081 but pass@4 is
          0.6034: the policy already passes SOMETIMES. That gap
          is the fuel RLVR burns.
  Part 2  One group, worked by hand: prompt 47 + 38, K = 4
          samples 75, 95, 85, 75 -> rewards 0, 0, 1, 0 -> group
          mean 0.25 -> advantages -0.25, -0.25, +0.75, -0.25.
          The passing sample gets pushed up 3x harder than
          each failing one gets pushed down.
  Part 3  Forty rounds of group-relative updates (SGD on the
          policy gradient -A * log p(sample)). pass@1 climbs
          0.2081 -> 0.8464, greedy goes 0/6 -> 6/6. And 40 of
          240 groups are DEAD - all-pass or all-fail, advantage
          exactly zero, no gradient: no contrast, no lesson.

Sampling uses FIXED listed draws (a hash formula, no RNG), so
every number below is exact and reproducible on any machine.

Runs on the ~/ai-lab venv: torch only (free), CPU or MPS,
no CUDA, no paid APIs, no downloads.

Expected output:

  Part 1 - the checker is the reward: no judge, just a + b
    47 + 38 = 85: choices 75, 85, 95, 84   p(correct) 0.2004
    26 + 49 = 75: choices 75, 65, 76, 85   p(correct) 0.2004
    58 + 27 = 85: choices 95, 84, 85, 75   p(correct) 0.1563
    39 + 44 = 83: choices 73, 93, 82, 83   p(correct) 0.2094
    64 + 18 = 82: choices 82, 72, 92, 81   p(correct) 0.2558
    35 + 57 = 92: choices 82, 102, 92, 91   p(correct) 0.2265
    pass@1 (expected over 1 sample):  0.2081
    pass@4 (expected over 4 samples): 0.6034
    greedy argmax correct: 0/6 - sampling finds passes the top pick misses
  Part 2 - one group, worked: prompt 47 + 38, round 0, K = 4 samples
    sampled answers: 75, 95, 85, 75
    checker rewards: 0, 0, 1, 0   group mean 0.25
    advantages (reward - mean): -0.25, -0.25, +0.75, -0.25
    the +0.75 sample gets pushed UP, the -0.25 samples DOWN
  Part 3 - group-relative training: 40 rounds, K = 4 per prompt, lr = 2.0
    round  0: pass@1 0.2081  pass@4 0.6034  greedy 0/6
    round  5: pass@1 0.2674  pass@4 0.7053  greedy 1/6
    round 10: pass@1 0.3424  pass@4 0.7997  greedy 4/6
    round 15: pass@1 0.4523  pass@4 0.8991  greedy 6/6
    round 20: pass@1 0.5656  pass@4 0.9574  greedy 6/6
    round 25: pass@1 0.6732  pass@4 0.9857  greedy 6/6
    round 30: pass@1 0.7463  pass@4 0.9952  greedy 6/6
    round 35: pass@1 0.8131  pass@4 0.9986  greedy 6/6
    round 40: pass@1 0.8464  pass@4 0.9993  greedy 6/6
    dead groups (all-pass or all-fail, zero advantage): 40 of 240
    final: pass@1 0.8464  pass@4 0.9993  greedy 6/6
    verdict: nobody judged style - a checker counted passes, and 'sometimes' became 'usually'
"""
import torch

# Six verifiable prompts: two-digit addition. The checker
# recomputes a + b - there is no label file to leak or game.
PROMPTS = [(47, 38), (26, 49), (58, 27), (39, 44), (64, 18), (35, 57)]
CHOICES = torch.tensor([[75, 85, 95, 84], [75, 65, 76, 85],
                        [95, 84, 85, 75], [73, 93, 82, 83],
                        [82, 72, 92, 81], [82, 102, 92, 91]])
# Hand-set starting policy: the correct answer is never the
# argmax, but it always has real probability mass (~0.16-0.26).
LOGITS0 = torch.tensor([[0.8, 0.2, 0.5, 0.0], [0.3, 0.9, 0.1, 0.6],
                        [0.7, 0.4, 0.1, 0.9], [0.2, 0.8, 0.5, 0.3],
                        [0.4, 0.7, 0.2, 0.1], [0.9, 0.3, 0.5, 0.6]])
CORRECT = [int((CHOICES[i] == PROMPTS[i][0] + PROMPTS[i][1])
               .nonzero()[0]) for i in range(len(PROMPTS))]
K, ROUNDS, LR = 4, 40, 2.0
P = len(PROMPTS)


def checker(p, a):
    """The whole reward model: recompute and compare. 1 or 0."""
    return int(CHOICES[p, a].item() == PROMPTS[p][0] + PROMPTS[p][1])


def draw_u(r, p, k):
    """Fixed listed draws - a hash, not an RNG. Same on every
    machine, every run, so the numbers below are exact."""
    return ((r * 2654435761 + p * 40503 + k * 65599 + 12345)
            % 100000) / 100000.0


def sample_group(probs_p, r, p):
    """Draw K answers from the policy by inverse-CDF on the
    fixed uniforms; score each with the checker."""
    cum = probs_p.cumsum(0)
    idxs = [min(int((cum <= draw_u(r, p, k)).sum().item()), 3)
            for k in range(K)]
    return idxs, [checker(p, a) for a in idxs]


def metrics(logits):
    probs = torch.softmax(logits, dim=1)
    pc = probs[torch.arange(P), torch.tensor(CORRECT)]
    pass1 = pc.mean().item()
    passk = (1 - (1 - pc) ** K).mean().item()
    greedy = int((logits.argmax(dim=1)
                  == torch.tensor(CORRECT)).sum().item())
    return probs, pass1, passk, greedy


# ---------- Part 1: the checker is the reward ----------
print("Part 1 - the checker is the reward: no judge, just a + b")
logits = LOGITS0.clone()
probs, pass1, passk, greedy = metrics(logits)
for p in range(P):
    a, b = PROMPTS[p]
    ch = ", ".join(str(int(v)) for v in CHOICES[p])
    print(f"  {a} + {b} = {a + b}: choices {ch}   "
          f"p(correct) {probs[p, CORRECT[p]].item():.4f}")
print(f"  pass@1 (expected over 1 sample):  {pass1:.4f}")
print(f"  pass@{K} (expected over {K} samples): {passk:.4f}")
print(f"  greedy argmax correct: {greedy}/6 - sampling finds "
      "passes the top pick misses")

# ---------- Part 2: one group, worked ----------
print("Part 2 - one group, worked: prompt 47 + 38, round 0, "
      "K = 4 samples")
idxs, rews = sample_group(probs[0].detach(), 0, 0)
mean = sum(rews) / K
adv = [r - mean for r in rews]
vals = ", ".join(str(int(CHOICES[0, a])) for a in idxs)
print(f"  sampled answers: {vals}")
print(f"  checker rewards: {', '.join(str(r) for r in rews)}   "
      f"group mean {mean:.2f}")
print(f"  advantages (reward - mean): "
      f"{', '.join(f'{v:+.2f}' for v in adv)}")
print("  the +0.75 sample gets pushed UP, the -0.25 samples DOWN")

# ---------- Part 3: group-relative training ----------
print(f"Part 3 - group-relative training: {ROUNDS} rounds, "
      f"K = {K} per prompt, lr = {LR}")
logits = LOGITS0.clone().requires_grad_(True)
opt = torch.optim.SGD([logits], lr=LR)
dead = 0
for r in range(ROUNDS + 1):
    with torch.no_grad():
        probs, pass1, passk, greedy = metrics(logits)
    if r % 5 == 0:
        print(f"  round {r:2d}: pass@1 {pass1:.4f}  "
              f"pass@{K} {passk:.4f}  greedy {greedy}/6")
    if r == ROUNDS:
        break
    logp = torch.log_softmax(logits, dim=1)
    terms = []
    for p in range(P):
        idxs, rews = sample_group(probs[p], r, p)
        if all(v == rews[0] for v in rews):
            dead += 1                      # no contrast, no signal
        mean = sum(rews) / K
        for k in range(K):
            adv = rews[k] - mean           # group-relative advantage
            terms.append(-adv * logp[p, idxs[k]])
    loss = torch.stack(terms).mean()       # -A * log p(sample)
    opt.zero_grad()
    loss.backward()
    opt.step()
print(f"  dead groups (all-pass or all-fail, zero advantage): "
      f"{dead} of {ROUNDS * P}")
print(f"  final: pass@1 {pass1:.4f}  pass@{K} {passk:.4f}  "
      f"greedy {greedy}/6")
print("  verdict: nobody judged style - a checker counted "
      "passes, and 'sometimes' became 'usually'")
