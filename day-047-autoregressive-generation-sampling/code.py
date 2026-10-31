"""Day 47 - Inference 1: Autoregressive Generation & Sampling.

A trained model does not write text. It does exactly one
thing: given the tokens so far, it returns a probability
distribution over the NEXT token (Day 14's softmax over the
vocab). Generation is the loop wrapped around that:

    forward -> distribution -> PICK one token -> append ->
    forward again, one token longer.

The model proposes; the picking rule disposes. Tonight the
pick is the whole lesson, run on ONE fixed distribution so
every strategy's mechanics show up as numbers:

  Part 1  Temperature. Softmax(logits / T) at T = 0.5, 1.0,
          2.0 on the same logits. T rescales the gaps between
          logits: T < 1 sharpens toward greedy, T > 1 flattens
          toward uniform (0.1250 each here). It never changes
          the RANKING - argmax is 'cat' at every temperature.
  Part 2  Truncation. Top-k keeps the k biggest, top-p
          (nucleus) keeps the smallest set whose cumulative
          probability reaches p; both zero the tail and
          RENORMALIZE - divide what survives by the mass it
          kept, so it sums to 1 again. Top-p adapts its count
          to the shape: 4 tokens kept here, because the mass
          crosses 0.90 at the 4th.
  Part 3  The loop itself, on a hand-built bigram table
          (next-token logits per current token - the smallest
          possible "model"). Greedy from 'the' loops forever:
          the cat sat on the cat sat on... Sampling with
          three FIXED uniform draws picks a different token
          at step 1 and ends at '.' in three picks. Same
          table, same prompt - only the pick changed.

Sampling is shown the honest way: a draw u in [0, 1) lands
in a token's slice of the cumulative distribution. Real code
rolls that u with torch.multinomial; here the three draws
are listed in the code (0.75, 0.50, 0.60), so every output
digit is exact - no RNG anywhere in this lesson.

Runs on the ~/ai-lab venv: torch only (free), MPS if your
Mac offers it else CPU, no CUDA, no paid APIs, no downloads.

Expected output:

  Part 1 - one distribution, three temperatures: softmax(logits / T)
    logits: cat 2.4000, dog 1.8000, mat 1.2000, sat 0.6000, runs 0.0000, the -0.6000, on -1.2000, . -1.8000
    T=0.5 (sharpened): cat 0.6989, dog 0.2105, mat 0.0634, sat 0.0191, runs 0.0058, the 0.0017, on 0.0005, . 0.0002
    T=1.0 (as trained): cat 0.4549, dog 0.2497, mat 0.1370, sat 0.0752, runs 0.0413, the 0.0226, on 0.0124, . 0.0068
    T=2.0 (flattened): cat 0.2850, dog 0.2112, mat 0.1564, sat 0.1159, runs 0.0859, the 0.0636, on 0.0471, . 0.0349
    uniform would be 0.1250 per token; greedy pick is 'cat' at EVERY temperature - T moves the gaps, never the ranking
  Part 2 - truncate the tail, then renormalize (from the T=1.0 distribution)
    top-k k=3: kept cat, dog, mat (mass 0.8416) -> cat 0.5405, dog 0.2967, mat 0.1628; the other 5 tokens are now exactly 0
    top-p p=0.90: kept cat, dog, mat, sat (cumulative 0.4549, 0.7046, 0.8416, 0.9168) -> cat 0.4962, dog 0.2723, mat 0.1495, sat 0.0820
    tail mass removed: top-k cuts 0.1584, top-p cuts 0.0832 - renormalizing rescales the survivors back to a sum of 1
  Part 3 - the generation loop: predict -> pick -> append -> repeat
    greedy from 'the', 8 picks: the cat sat on the cat sat on the
    greedy is deterministic - and the argmax chain walks straight into a loop
    sampled from 'the', fixed draws [0.75, 0.5, 0.6]:
      after 'the'  draw 0.7500 -> 'dog'
      after 'dog'  draw 0.5000 -> 'runs'
      after 'runs'  draw 0.6000 -> '.'
    sampled result: the dog runs .  (stops at '.')
    verdict: same table, same prompt - only the pick changed the text
"""
import torch

device = torch.device("mps") if torch.backends.mps.is_available() \
    else torch.device("cpu")

fmt = lambda pairs: ", ".join(f"{t} {v:.4f}" for t, v in pairs)

# ---------- Part 1: temperature reshapes one distribution ----------
VOCAB = ["cat", "dog", "mat", "sat", "runs", "the", "on", "."]
logits = torch.tensor([2.4, 1.8, 1.2, 0.6, 0.0, -0.6, -1.2, -1.8],
                      device=device)

print("Part 1 - one distribution, three temperatures: "
      "softmax(logits / T)")
print(f"  logits: {fmt(zip(VOCAB, logits.tolist()))}")
for temp, tag in ((0.5, "sharpened"), (1.0, "as trained"),
                  (2.0, "flattened")):
    probs = torch.softmax(logits / temp, dim=0)
    print(f"  T={temp} ({tag}): {fmt(zip(VOCAB, probs.tolist()))}")
greedy = VOCAB[int(torch.argmax(logits).item())]
print(f"  uniform would be {1 / len(VOCAB):.4f} per token; greedy pick "
      f"is '{greedy}' at EVERY temperature - T moves the gaps, "
      "never the ranking")

# ---------- Part 2: truncate the tail, then renormalize ----------
probs = torch.softmax(logits, dim=0)                   # T=1.0 again

top_vals, top_idx = torch.topk(probs, 3)
kept_mass = top_vals.sum().item()
topk = [(VOCAB[int(i)], (v / top_vals.sum()).item())
        for v, i in zip(top_vals, top_idx)]
print("Part 2 - truncate the tail, then renormalize "
      "(from the T=1.0 distribution)")
print(f"  top-k k=3: kept {', '.join(t for t, _ in topk)} "
      f"(mass {kept_mass:.4f}) -> {fmt(topk)}; the other 5 tokens "
      "are now exactly 0")

sorted_probs, sorted_idx = torch.sort(probs, descending=True)
cum = torch.cumsum(sorted_probs, dim=0)
n_keep = int((cum < 0.90).sum().item()) + 1            # include the token that crosses p
p_mass = cum[n_keep - 1].item()
topp = [(VOCAB[int(i)], (v / sorted_probs[:n_keep].sum()).item())
        for v, i in zip(sorted_probs[:n_keep], sorted_idx[:n_keep])]
cum_kept = ", ".join(f"{c:.4f}" for c in cum[:n_keep].tolist())
print(f"  top-p p=0.90: kept {', '.join(t for t, _ in topp)} "
      f"(cumulative {cum_kept}) -> {fmt(topp)}")
print(f"  tail mass removed: top-k cuts {1 - kept_mass:.4f}, "
      f"top-p cuts {1 - p_mass:.4f} - renormalizing rescales the "
      "survivors back to a sum of 1")

# ---------- Part 3: the generation loop on a bigram table ----------
TOKENS = ["the", "cat", "dog", "sat", "runs", "on", "mat", "."]
TABLE = {  # next-token logits per current token, in TOKENS order
    "the":  [-2.0, 2.0, 1.4, -1.0, -1.5, -2.0, -1.0, -3.0],
    "cat":  [-2.0, -2.0, -1.5, 2.2, 0.5, -1.0, -0.5, -2.0],
    "dog":  [-2.0, -1.0, -2.5, 0.8, 2.2, -1.5, -2.0, -0.5],
    "sat":  [-1.0, -2.0, -2.0, -2.5, -1.0, 2.0, 1.2, -1.5],
    "runs": [0.5, -1.5, -1.0, -2.0, -2.0, -1.0, -1.5, 2.4],
    "on":   [2.0, -1.0, -1.0, -2.0, -2.0, -2.0, 1.4, -1.0],
    "mat":  [-1.0, -1.0, -2.0, -2.0, -2.0, -1.0, -2.0, 2.0],
    ".":    [2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -3.0],
}


def next_probs(token):
    row = torch.tensor(TABLE[token], device=device)
    return torch.softmax(row, dim=0)


def pick_from(probs, u):
    """One categorical draw, made visible: u lands in a token's
    slice of the cumulative distribution. torch.multinomial
    does exactly this with a u it rolls itself."""
    cum = torch.cumsum(probs, dim=0)
    return int((cum <= u).sum().item())


print("Part 3 - the generation loop: predict -> pick -> append -> "
      "repeat")
seq = ["the"]
for _ in range(8):
    seq.append(TOKENS[int(torch.argmax(
        next_probs(seq[-1])).item())])
print(f"  greedy from 'the', 8 picks: {' '.join(seq)}")
print("  greedy is deterministic - and the argmax chain walks "
      "straight into a loop")

draws = [0.75, 0.50, 0.60]
seq = ["the"]
print(f"  sampled from 'the', fixed draws {draws}:")
for u in draws:
    nxt = TOKENS[pick_from(next_probs(seq[-1]), u)]
    print(f"    after '{seq[-1]}'  draw {u:.4f} -> '{nxt}'")
    seq.append(nxt)
    if nxt == ".":
        break
print(f"  sampled result: {' '.join(seq)}  (stops at '.')")
print("  verdict: same table, same prompt - only the pick changed "
      "the text")
