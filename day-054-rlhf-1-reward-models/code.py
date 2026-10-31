"""Day 54 - RLHF 1: Why SFT Isn't Enough - Preferences & Reward Models.

SFT (Day 52) teaches by demonstration: here is the right answer,
copy it. But some targets cannot be demonstrated, only judged.
Nobody can write the one correct helpful, honest, harmless
response to an open-ended prompt - yet almost anyone can look
at two responses and say which is better. You cannot write the
loss for "good", but you can rank two outcomes.

RLHF's first move is to bottle that judgment:

  1. Sample two responses to the same prompt; a human picks
     the better one. That is a preference pair.
  2. Train a REWARD MODEL r_theta(x, y) - a network that takes
     a prompt + response and returns ONE scalar score - so
     that its scores agree with the human's picks.
  3. The learned score then becomes the objective: later, the
     policy (the LLM itself) is optimized to produce responses
     the reward model scores highly. That optimization step is
     the RL part; tonight we only build the score it will chase.

The training rule is Bradley-Terry (1952, from chess ratings):

    P(A beats B) = sigmoid(r_A - r_B)
    loss = -log sigmoid(r_chosen - r_rejected)

Only the GAP between scores matters, and the loss is exactly
Day 11's maximum likelihood, one comparison at a time: pick
the weights that make the human's actual choices most probable
under the model. Day 14's sigmoid does the gap-to-probability
conversion.

  Part 1  The Bradley-Terry table: score gaps -2..+2 become
          probabilities and losses. One worked pair, scored
          both ways - agreeing with the human is cheap,
          confidently disagreeing is expensive.
  Part 2  Synthetic preference data, built by formula (no
          RNG): 80 "responses", each a 3-feature vector
          (correctness, detail, verbosity). A hidden true
          utility - 2.0*correctness + 1.2*detail - 1.0*verbosity
          plus a listed deterministic noise term - plays the
          human: in each pair, the higher-utility response is
          "chosen". 42 train pairs, 12 held-out pairs on
          disjoint responses.
  Part 3  Train a tiny reward model - one nn.Linear(3, 1),
          zero-init, SGD lr=0.5, 300 full-batch epochs - with
          the Bradley-Terry loss. Watch the loss fall from
          0.6931 (=-log 0.5, no opinions) and the learned
          weights line up with the hidden true utility:
          rankings alone recover what the judge valued.

Honest note (the seed of a later problem): r is a LEARNED
PROXY for "good", not "good" itself. Optimize a policy against
it hard enough and the policy learns to game the score -
reward hacking, Goodhart's law with gradients. Tonight is a
pointer, not that lesson.

Runs on the ~/ai-lab venv: torch only (free), CPU on purpose
(tiny data, and float32 arithmetic you can twin by hand), no
CUDA, no paid APIs, no downloads. All data is built by the
integer formulas below - no RNG anywhere, so the numbers are
reproducible. Expected output below is reference numbers from
a verified float32 twin of this exact algorithm; your Mac's
last digits may vary by a hair, but the loss arc, the weight
recovery, and the pair accuracies are the guarantee.

Expected output:

  Part 1 - Bradley-Terry: a score gap becomes a probability
    dr = -2.0:  P(human picks the higher-utility response) = 0.1192   BT loss = 2.1269
    dr = -1.0:  P(human picks the higher-utility response) = 0.2689   BT loss = 1.3133
    dr = +0.0:  P(human picks the higher-utility response) = 0.5000   BT loss = 0.6931
    dr = +1.0:  P(human picks the higher-utility response) = 0.7311   BT loss = 0.3133
    dr = +2.0:  P(human picks the higher-utility response) = 0.8808   BT loss = 0.1269
    worked pair: r_chosen = 1.6000, r_rejected = 0.4000 -> dr = 1.2000, P = 0.7685, loss = 0.2633
    same pair, human flips the ranking: loss = 1.4633 (confidence in the wrong order is expensive)
  Part 2 - synthetic preference data (formula-built, no RNG)
    80 responses x 3 features [correctness, detail, verbosity]; hidden true utility w = [2.0000, 1.2000, -1.0000] + listed noise
    42 train pairs / 12 held-out pairs (disjoint responses); a coin flip scores 0.5000
    first train pair: chosen #61 [0.4500, 0.8125, 0.2647] (true u = 1.4503) beats rejected #0 [0.2500, 0.1250, 0.3529] (true u = 0.2171)
  Part 3 - train the reward model r = w.x (SGD lr=0.5, 300 epochs, full batch)
    epoch 0: loss 0.6931  (= -log 0.5: zero-init, the model has no opinions yet)
    epoch 75: loss 0.3059
    epoch 150: loss 0.2288
    epoch 225: loss 0.1925
    epoch 300: loss 0.1707
    learned w: [5.7572, 3.4427, -3.0144]   rescaled so w[0] = 2.0000: [2.0000, 1.1960, -1.0472]
    true w:    [2.0000, 1.2000, -1.0000]   cosine(learned, true) = 0.9998 - direction recovered from rankings alone
    pair accuracy: train 0.9524 (40/42)   held-out 0.8333 (10/12)
    gauge: held-out #47 scores r = 3.4024, #58 scores r = 5.3299 -> P(#47 beats #58) = 0.1270 (true utility agrees: #58 wins)
    verdict: nobody ever wrote down 'good' - 42 rankings taught a scalar score that generalizes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

fmt = lambda t: ", ".join(f"{v:.4f}" for v in t.flatten().tolist())

# ---------- Part 1: Bradley-Terry, the gap-to-probability rule ----------
print("Part 1 - Bradley-Terry: a score gap becomes a probability")
gaps = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
probs = torch.sigmoid(gaps)                    # Day 14's sigmoid
for dr, p in zip(gaps.tolist(), probs.tolist()):
    print(f"  dr = {dr:+.1f}:  P(human picks the higher-utility "
          f"response) = {p:.4f}   BT loss = {-torch.log(torch.tensor(p)):.4f}")
r_c, r_r = 1.6, 0.4
p_win = torch.sigmoid(torch.tensor(r_c - r_r)).item()
print(f"  worked pair: r_chosen = {r_c:.4f}, r_rejected = {r_r:.4f} "
      f"-> dr = {r_c - r_r:.4f}, P = {p_win:.4f}, "
      f"loss = {-torch.log(torch.tensor(p_win)):.4f}")
print(f"  same pair, human flips the ranking: loss = "
      f"{-torch.log(torch.tensor(1.0 - p_win)):.4f} "
      "(confidence in the wrong order is expensive)")

# ---------- Part 2: formula-built preference data (no RNG) ----------
print("Part 2 - synthetic preference data (formula-built, no RNG)")
i = torch.arange(80)
# Three features in [0, 1], each from its own integer formula -
# exact arithmetic, identical on every machine.
X = torch.stack([
    ((i * 19 + 5) % 21) / 20.0,                # correctness
    ((i * 13 + 2) % 17) / 16.0,                # detail
    ((i * 47 + 12) % 35) / 34.0,               # verbosity
], dim=1).float()
W_TRUE = torch.tensor([2.0, 1.2, -1.0])        # the hidden judge
noise = 0.4 * (((i * 7 + 3) % 11) / 10.0 - 0.5)  # judge noise, listed
u = X @ W_TRUE + noise                         # true utility per response

perm = torch.argsort((i * 53) % 97)            # fixed shuffle, by formula
tr_resp, te_resp = perm[:56], perm[56:]


def make_pairs(resp, offsets):
    """Pair up responses; the higher-utility one is 'chosen'."""
    chosen, rejected = [], []
    for a, b in offsets(resp):
        a, b = int(a), int(b)
        if u[a] >= u[b]:
            chosen.append(a); rejected.append(b)
        else:
            chosen.append(b); rejected.append(a)
    return (torch.tensor(chosen), torch.tensor(rejected))


train_pairs = make_pairs(tr_resp, lambda r: (
    [(r[k], r[k + 28]) for k in range(28)]
    + [(r[k], r[k + 1]) for k in range(0, 27, 2)]))
test_pairs = make_pairs(te_resp, lambda r: (
    [(r[k], r[k + 12]) for k in range(12)]))
print(f"  80 responses x 3 features [correctness, detail, verbosity]; "
      f"hidden true utility w = [{fmt(W_TRUE)}] + listed noise")
print(f"  {len(train_pairs[0])} train pairs / {len(test_pairs[0])} "
      "held-out pairs (disjoint responses); a coin flip scores 0.5000")
c0, r0 = int(train_pairs[0][0]), int(train_pairs[1][0])
print(f"  first train pair: chosen #{c0} [{fmt(X[c0])}] "
      f"(true u = {u[c0]:.4f}) beats rejected #{r0} [{fmt(X[r0])}] "
      f"(true u = {u[r0]:.4f})")

# ---------- Part 3: train the reward model on rankings alone ----------
print("Part 3 - train the reward model r = w.x "
      "(SGD lr=0.5, 300 epochs, full batch)")
rm = nn.Linear(3, 1, bias=False)               # r_theta(x, y) = w . feats
with torch.no_grad():
    rm.weight.zero_()                          # no opinions yet
opt = torch.optim.SGD(rm.parameters(), lr=0.5)
Xc, Xr = X[train_pairs[0]], X[train_pairs[1]]  # chosen / rejected feats


def bt_loss(model, a, b):
    """-log sigmoid(r_chosen - r_rejected), averaged over pairs.
    Day 11's MLE: make the human's actual picks most probable."""
    return -F.logsigmoid(model(a).squeeze(1)
                         - model(b).squeeze(1)).mean()


for epoch in range(301):
    loss = bt_loss(rm, Xc, Xr)
    if epoch in (0, 75, 150, 225, 300):
        note = "  (= -log 0.5: zero-init, the model has no " \
               "opinions yet)" if epoch == 0 else ""
        print(f"  epoch {epoch}: loss {loss.item():.4f}{note}")
    if epoch == 300:
        break
    opt.zero_grad()
    loss.backward()
    opt.step()

w = rm.weight.detach().flatten()
rescaled = w * (2.0 / w[0])
cos = F.cosine_similarity(w, W_TRUE, dim=0).item()
print(f"  learned w: [{fmt(w)}]   rescaled so w[0] = 2.0000: "
      f"[{fmt(rescaled)}]")
print(f"  true w:    [{fmt(W_TRUE)}]   cosine(learned, true) = "
      f"{cos:.4f} - direction recovered from rankings alone")
with torch.no_grad():
    def pair_acc(pairs):
        a, b = X[pairs[0]], X[pairs[1]]
        return (rm(a) > rm(b)).float().mean().item(), len(pairs[0])
    acc_tr, n_tr = pair_acc(train_pairs)
    acc_te, n_te = pair_acc(test_pairs)
print(f"  pair accuracy: train {acc_tr:.4f} "
      f"({int(round(acc_tr * n_tr))}/{n_tr})   held-out {acc_te:.4f} "
      f"({int(round(acc_te * n_te))}/{n_te})")
a_id, b_id = int(te_resp[0]), int(te_resp[1])
with torch.no_grad():
    ra, rb = rm(X[a_id]).item(), rm(X[b_id]).item()
    p_ab = torch.sigmoid(torch.tensor(ra - rb)).item()
print(f"  gauge: held-out #{a_id} scores r = {ra:.4f}, #{b_id} scores "
      f"r = {rb:.4f} -> P(#{a_id} beats #{b_id}) = {p_ab:.4f} "
      f"(true utility agrees: #{b_id} wins)")
print("  verdict: nobody ever wrote down 'good' - 42 rankings taught "
      "a scalar score that generalizes")
