# Fine-Tuning 1: Supervised Fine-Tuning (SFT) — Day 52 of 100

Tonight you'll be able to: explain how supervised fine-tuning turns Day 51's text predictor into an instruction-follower — and compute its masked loss by hand, showing exactly which tokens train and which are invisible.

## The one idea

Day 51 left you with a base model: a superb next-token predictor trained on trillions of tokens of raw text. Ask it a question and it does the most document-like thing — it might answer, or it might generate three more questions, because in its training data questions are usually followed by more text, not by obedience. It has the knowledge; it does not have the *job*. **Supervised fine-tuning (SFT)** is how it gets the job: keep training the same weights, with the same loop, on examples that show the desired behavior — a prompt and the response you *wish* the model had given.

**The data shape.** An SFT example is a pair — instruction in, desired response out — wrapped in a **chat template**: special marker tokens that delimit who is speaking (`<user>` … `<asst>` … `<eos>` in tonight's toy version; production templates do the same job with their own marker strings). The markers matter: they turn "some text" into a turn structure the model can condition on, so at serving time the same template frames the user's message and the model continues in the assistant's slot. Datasets are *curated*, not scraped — humans or strong teacher models write responses worth imitating. That curation is the whole point: pretraining data taught the model what text looks like; SFT data teaches it what *its* text should look like.

**The mask — tonight's load-bearing mechanic.** The training loss is computed **only on the response tokens**. Mechanically: labels are a copy of the input ids, except every prompt position is overwritten with `-100`, which is PyTorch's `ignore_index` — cross-entropy skips those positions entirely, no loss and no gradient. (The model internally shifts labels by one, Day 42-style: the position before a token predicts it, so the first response token is trained from the `<asst>` marker exactly as it should be.) Why mask? Because an unmasked loss would spend half its gradient teaching the model to *generate the user's side* — to imitate questions, typos and all — and would dilute the signal that actually shapes behavior. The prompt is context to read; the response is the curriculum. One deliberate exception: the closing `<eos>` keeps its label, so the model learns when to *stop* — an assistant that never ends its turn was never trained to.

**Scale honesty.** Next to pretraining, SFT is tiny: typically **thousands to a few million examples** (against trillions of pretraining tokens), a few passes over the set, at a modest learning rate, finishing in hours to days instead of months. That smallness is informative: SFT mostly does not pour in new knowledge — there isn't enough data for that. It teaches format, tone, and task shape; it *surfaces* what pretraining already put in. Which is exactly why tomorrow's question even makes sense: if the change is this shallow, maybe you don't need to rewrite every weight to make it.

> **Analogy.** Onboarding, not education. Pretraining (Day 51) is the new hire who somehow read the entire internet before day one — brilliant, knowledgeable, and with no idea how your team answers a ticket. SFT is their first week with the runbook: you hand them a stack of real requests, each paired with *the reply a senior would have sent*, and you grade only their replies. You would never grade them on reproducing the customer's email — that's context, not performance. The `-100` mask is that grading policy written in code: loss lands on the response payload only; the request headers are read, never scored.

## See it

![SFT diagram: a token row split prompt/response with loss mask gates, then the base-to-chat pipeline](diagram.svg)

Top: one SFT example as a token row — the prompt side (blue) is labeled `-100`, so its loss gates are OFF: the model reads it but is never trained to produce it. The response side (green) keeps its ids as labels, gates ON — including `<eos>`, which teaches stopping. Bottom: the pipeline — same weights, same Day 7 loop; only the data (curated pairs) and the mask are new.

## Code it (~30 min)

One file, three parts, a 30-token toy vocab — small enough that every number is hand-checkable, structured exactly like the real pipeline. **Part 1** builds 8 instruction pairs, wraps them in the chat template, and prints example 1's full masked label row plus the dataset-wide fraction of tokens that carry loss. **Part 2** computes masked cross-entropy by hand on hand-set logits and matches `F.cross_entropy(ignore_index=-100)` to the digit — including how the mask changes the denominator. **Part 3** proves the mask is a wall: perturb a prompt-side logit and the loss cannot see it; perturb a response-side logit and it jumps.

```python
"""Day 52 - Fine-Tuning 1: Supervised Fine-Tuning (SFT).

Day 51's base model predicts text; it does not follow
instructions. SFT is the recipe that closes that gap: keep
training the SAME weights on (prompt, desired response)
pairs, so next-token prediction quietly becomes
instruction-following. No new architecture, no new loop -
the Day 7 training loop on new data.

The load-bearing mechanic is the MASK. Each example is one
token row: chat markers + prompt + response. The labels are
the input ids with every prompt position overwritten by
-100, PyTorch's ignore_index. Cross-entropy skips -100
positions entirely - no loss, no gradient - because you
want the model to learn to PRODUCE the assistant's side,
never to generate the user's side. The prompt is context;
only the answer is curriculum.

  Part 1  Build a tiny instruction dataset (8 prompt /
          response pairs), tokenize it with a 30-token word
          vocab plus three chat markers (<user>, <asst>,
          <eos>), and construct input_ids + labels. Example 1
          prints its full masked row: 3 of 7 tokens carry
          loss. Across all 8 examples only 20 of 58 tokens
          (0.3448) train anything at all.
  Part 2  Masked cross-entropy, by hand vs the library.
          Hand-set logits for 5 positions, labels with two
          -100s: sum the per-token CE over the 3 live
          positions and divide by 3 - F.cross_entropy with
          ignore_index=-100 matches exactly. Divide by all 5
          instead and the number lies: the mask sets the
          denominator too.
  Part 3  Prove the mask is a wall. Add +5.0 to a prompt
          position's logit: the masked loss does not move
          (diff 0.000000). Add it to a response position:
          the loss jumps. Then the scale honesty: real SFT
          is thousands-to-millions of pairs - a rounding
          error next to pretraining's trillions of tokens -
          which is why SFT takes hours-to-days, not months.

Runs on the ~/ai-lab venv: torch only (free), CPU on
purpose (tiny data, exact digits), no CUDA, no paid APIs,
no downloads. Everything is hand-set - no RNG anywhere.

Expected output:

  Part 1 - build the SFT dataset: (prompt, response) -> input_ids + masked labels
    vocab: 30 tokens (3 chat markers + 27 words)
    example 1: prompt 'say hello'  response 'hello there'
      tokens:    <user> say hello <asst> hello there <eos>
      input_ids: [0, 3, 4, 1, 4, 5, 2]
      labels:    [-100, -100, -100, -100, 4, 5, 2]
      loss tokens: 3 of 7 = 0.4286 of this example
    all 8 examples: 58 tokens total, 20 carry loss = 0.3448 (34.48%)
    the remaining 65.52% is the user's side - read as context, never trained on
  Part 2 - masked cross-entropy, by hand vs F.cross_entropy(ignore_index=-100)
    labels: [-100, -100, 2, 1, 4]  (the first two positions are 'prompt')
    per-token CE, reduction='none': [0.0000, 0.0000, 0.6196, 0.6479, 0.6196]
    by hand over the 3 live positions: (0.6196 + 0.6479 + 0.6196) / 3 = 0.6290
    F.cross_entropy (masked mean): 0.6290   max diff vs by-hand 0.000000
    the same sum over all 5 positions would read 0.3774 - the mask sets the denominator too
  Part 3 - prove the mask is a wall: perturb a logit, watch the loss
    base masked loss:         0.6290
    +5.0 on a PROMPT logit:   0.6290  (diff 0.000000 - invisible to the loss)
    +5.0 on a RESPONSE logit: 1.6052  (the loss moves - this side is what learns)
    scale check: real SFT runs on 10K-1M examples like these; Day 51's pretraining ran on trillions of tokens
    verdict: SFT = the pretraining loop + instruction pairs + a mask that trains only the answer
"""
import torch
import torch.nn.functional as F

# ---------- Part 1: the dataset, the tokens, the mask ----------
print("Part 1 - build the SFT dataset: (prompt, response) "
      "-> input_ids + masked labels")
PAIRS = [
    ("say hello", "hello there"),
    ("say bye", "bye now"),
    ("add two three", "five"),
    ("what color sky", "blue"),
    ("name first planet", "mercury"),
    ("say thank you", "you are welcome"),
    ("count one two", "three"),
    ("opposite of hot", "cold"),
]
SPECIALS = ["<user>", "<asst>", "<eos>"]       # the chat template
vocab = list(SPECIALS)
for prompt, response in PAIRS:
    for word in (prompt + " " + response).split():
        if word not in vocab:
            vocab.append(word)
stoi = {w: i for i, w in enumerate(vocab)}
USER, ASST, EOS = 0, 1, 2
print(f"  vocab: {len(vocab)} tokens "
      f"({len(SPECIALS)} chat markers + {len(vocab) - len(SPECIALS)} words)")


def build(prompt, response):
    """One SFT example: one token row, labels = ids with the
    whole prompt side (markers included) masked to -100.
    Only the response tokens and the final <eos> keep their
    ids - <eos> carries loss so the model learns to STOP."""
    p_ids = [stoi[w] for w in prompt.split()]
    r_ids = [stoi[w] for w in response.split()]
    input_ids = [USER] + p_ids + [ASST] + r_ids + [EOS]
    labels = [-100] * (2 + len(p_ids)) + r_ids + [EOS]
    return input_ids, labels


ids, labels = build(*PAIRS[0])
tokens = [vocab[i] for i in ids]
n_live = sum(1 for x in labels if x != -100)
print(f"  example 1: prompt '{PAIRS[0][0]}'  response '{PAIRS[0][1]}'")
print(f"    tokens:    {' '.join(tokens)}")
print(f"    input_ids: {ids}")
print(f"    labels:    {labels}")
print(f"    loss tokens: {n_live} of {len(ids)} = "
      f"{n_live / len(ids):.4f} of this example")
total = live = 0
for prompt, response in PAIRS:
    ex_ids, ex_labels = build(prompt, response)
    total += len(ex_ids)
    live += sum(1 for x in ex_labels if x != -100)
print(f"  all {len(PAIRS)} examples: {total} tokens total, "
      f"{live} carry loss = {live / total:.4f} "
      f"({100 * live / total:.2f}%)")
print(f"  the remaining {100 * (1 - live / total):.2f}% is the "
      "user's side - read as context, never trained on")

# ---------- Part 2: masked cross-entropy, by hand ----------
print("Part 2 - masked cross-entropy, by hand vs "
      "F.cross_entropy(ignore_index=-100)")
logits = torch.tensor([
    [2.0, 0.0, -1.0, 0.5, 0.0, 1.0],
    [0.0, 1.0, 0.5, -0.5, 2.0, 0.0],
    [0.5, -1.0, 2.0, 0.0, 1.0, -0.5],
    [1.0, 2.0, 0.0, -1.0, 0.5, 0.0],
    [-0.5, 0.0, 1.0, 0.5, 2.0, -1.0],
])
labels2 = torch.tensor([-100, -100, 2, 1, 4])
print(f"  labels: {labels2.tolist()}  "
      "(the first two positions are 'prompt')")
per_token = F.cross_entropy(logits, labels2, ignore_index=-100,
                            reduction="none")
fmt = lambda t: ", ".join(f"{v:.4f}" for v in t.tolist())
print(f"  per-token CE, reduction='none': [{fmt(per_token)}]")
logp = torch.log_softmax(logits, dim=-1)
live_idx = [i for i, x in enumerate(labels2.tolist()) if x != -100]
hand = sum(-logp[i, labels2[i]].item() for i in live_idx) / len(live_idx)
pieces = " + ".join(f"{per_token[i].item():.4f}" for i in live_idx)
masked = F.cross_entropy(logits, labels2, ignore_index=-100)
print(f"  by hand over the {len(live_idx)} live positions: "
      f"({pieces}) / {len(live_idx)} = {hand:.4f}")
print(f"  F.cross_entropy (masked mean): {masked.item():.4f}   "
      f"max diff vs by-hand {abs(hand - masked.item()):.6f}")
naive = per_token.sum().item() / len(labels2)
print(f"  the same sum over all {len(labels2)} positions would "
      f"read {naive:.4f} - the mask sets the denominator too")

# ---------- Part 3: the mask is a wall ----------
print("Part 3 - prove the mask is a wall: perturb a logit, "
      "watch the loss")
base = masked.item()
bumped_prompt = logits.clone()
bumped_prompt[0, 0] += 5.0                     # a 'prompt' position
p_loss = F.cross_entropy(bumped_prompt, labels2,
                         ignore_index=-100).item()
bumped_resp = logits.clone()
bumped_resp[2, 0] += 5.0                       # a 'response' position
r_loss = F.cross_entropy(bumped_resp, labels2,
                         ignore_index=-100).item()
print(f"  base masked loss:         {base:.4f}")
print(f"  +5.0 on a PROMPT logit:   {p_loss:.4f}  "
      f"(diff {abs(p_loss - base):.6f} - invisible to the loss)")
print(f"  +5.0 on a RESPONSE logit: {r_loss:.4f}  "
      "(the loss moves - this side is what learns)")
print("  scale check: real SFT runs on 10K-1M examples like "
      "these; Day 51's pretraining ran on trillions of tokens")
print("  verdict: SFT = the pretraining loop + instruction "
      "pairs + a mask that trains only the answer")
```

**Expected output:**

```
Part 1 - build the SFT dataset: (prompt, response) -> input_ids + masked labels
  vocab: 30 tokens (3 chat markers + 27 words)
  example 1: prompt 'say hello'  response 'hello there'
    tokens:    <user> say hello <asst> hello there <eos>
    input_ids: [0, 3, 4, 1, 4, 5, 2]
    labels:    [-100, -100, -100, -100, 4, 5, 2]
    loss tokens: 3 of 7 = 0.4286 of this example
  all 8 examples: 58 tokens total, 20 carry loss = 0.3448 (34.48%)
  the remaining 65.52% is the user's side - read as context, never trained on
Part 2 - masked cross-entropy, by hand vs F.cross_entropy(ignore_index=-100)
  labels: [-100, -100, 2, 1, 4]  (the first two positions are 'prompt')
  per-token CE, reduction='none': [0.0000, 0.0000, 0.6196, 0.6479, 0.6196]
  by hand over the 3 live positions: (0.6196 + 0.6479 + 0.6196) / 3 = 0.6290
  F.cross_entropy (masked mean): 0.6290   max diff vs by-hand 0.000000
  the same sum over all 5 positions would read 0.3774 - the mask sets the denominator too
Part 3 - prove the mask is a wall: perturb a logit, watch the loss
  base masked loss:         0.6290
  +5.0 on a PROMPT logit:   0.6290  (diff 0.000000 - invisible to the loss)
  +5.0 on a RESPONSE logit: 1.6052  (the loss moves - this side is what learns)
  scale check: real SFT runs on 10K-1M examples like these; Day 51's pretraining ran on trillions of tokens
  verdict: SFT = the pretraining loop + instruction pairs + a mask that trains only the answer
```

> **Today's win.** Done when your run prints the three proofs: example 1's labels row `[-100, -100, -100, -100, 4, 5, 2]` with **20 of 58** dataset tokens carrying loss (`0.3448`); the by-hand masked CE `0.6290` matching `F.cross_entropy` with max diff `0.000000`; and the perturbation split — prompt logit diff `0.000000`, response logit moving the loss to `1.6052`. The keeper sentence, no notes: SFT is pretraining's loop run on (prompt, response) pairs with the loss masked to the response, because you train the side you want the model to produce.

> **Short on time? 20-minute version.** Read "The one idea", study the diagram's top panel until you can say which tokens are `-100` and why, then run Parts 1 and 2 only. If you can explain why the denominator is 3 and not 5, you have the mechanic — Part 3 is the victory lap.

> **Tomorrow's teaser.** Tomorrow: SFT as you ran it tonight updates *every* weight in the model — fine at toy scale, heavy at 70B. Tomorrow we get most of the same effect by training only a small patch instead of the whole network.

*Streak: 52 day(s) · Never miss twice.*
