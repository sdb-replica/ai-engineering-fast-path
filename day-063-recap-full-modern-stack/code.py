"""Day 63 - Recap: The Full Modern Stack on One Page.

Phase D recap, same treatment as Days 20, 34 and 50: one
carry sentence, a slot map of every day in the phase, and a
small artifact that makes the map executable. The sentence:

    A frontier model is a pretrained next-token engine,
    taught manners by SFT, steered by preferences (RM/DPO)
    or checkers (RLVR), measured by evals, given fresh
    facts by RAG, and given hands by tools - and serving
    it is a separate discipline (Phase E).

Tonight the map becomes a MANIFEST: the stack written the
way you would draw it in a system-design interview. Every
stage declares its input artifact, its output artifact,
the data it eats, and its share of the training compute -
counted with Day 51's rule of thumb, C ~ 6 * N * D (six
times parameters times tokens seen).

Reference model, stated so the arithmetic is checkable:
N = 7e9 parameters, pretrained on 1.4e12 tokens. The
post-training token counts are one plausible run of each
stage, stated in the table - change them and the verdict
does not move, because pretraining is not ahead by a
little. It is ahead by four orders of magnitude.

  Part 1  The artifact chain: what each stage hands the
          next - corpus to base model to instruct model
          to aligned model to frozen artifact - plus the
          two serving-time attachments, which train
          nothing: RAG's index and the agent's tools.
  Part 2  The compute table: 6ND per training stage and
          each stage's share of the total. Evals spend
          inference, not gradient; serving stages spend
          nothing at build time at all.
  Part 3  The slot map + checks: build (51-53), steer
          (54-56), measure (57), serve-time (58-62) -
          and the printed sentence, for your notes.

Pure stdlib, deterministic arithmetic only - no RNG, no
torch needed, no APIs, no downloads. Runs as-is in the
~/ai-lab venv on the MacBook.

Expected output:

  Part 1 - the stack as an artifact chain (Days 51-62)
    Day 51  Pretraining   raw text corpus -> base model (theta)
    Day 52  SFT           base model -> instruct model
    Day 53  LoRA run      instruct model -> adapter (delta-W)
    Day 54  Reward model  instruct model -> reward model r(x, y)
    Day 55  DPO           instruct model -> aligned model
    Day 56  RLVR          instruct model -> reasoning model
    Day 57  Evals         candidate model -> scorecard: ship / no-ship
    Day 58  RAG index     his documents -> vector index
    Day 59  RAG loop      vector index + question -> grounded answer
    Day 60  Tool use      aligned model -> model that calls functions
    Day 61  Agent loop    model + tools -> agent: plan-act-observe
    Day 62  Prompts       any request -> structured request
    read it bottom-up in the diagram: every arrow is a
    handoff you could version, store, and roll back.
  Part 2 - training compute by stage (C ~ 6ND, Day 51)
    day  stage             tokens      FLOPs  share
     51  Pretraining     1.40e+12   5.88e+22  99.9893%
     52  SFT             6.00e+07   2.52e+18  0.0043%
     53  LoRA run        1.00e+07   4.20e+17  0.0007%
     54  Reward model    2.00e+07   8.40e+17  0.0014%
     55  DPO             1.00e+07   4.20e+17  0.0007%
     56  RLVR            5.00e+07   2.10e+18  0.0036%
     57  Evals                  -          -  no training
     58  RAG index              -          -  no training
     59  RAG loop               -          -  no training
     60  Tool use               -          -  no training
     61  Agent loop             -          -  no training
     62  Prompts                -          -  no training
    total training compute: 5.88e+22 FLOPs
    one pretraining run = 23,333 SFT runs
    everything after Day 51 is a rounding error in FLOPs -
    which is exactly why the stack is economical: train the
    engine once, steer it cheaply, attach facts and hands
    at serving time instead of retraining.
  Part 3 - the slot map
    BUILD (51-53)      make the engine, then teach it manners - LoRA is how you run these stages cheaply
    STEER (54-56)      preferences point it (RM, DPO); checkers grade it (RLVR) - two ways to say 'better'
    MEASURE (57)       evals are the gate between a checkpoint and a product - no scorecard, no ship
    SERVE-TIME (58-62) RAG bolts on fresh facts, tools bolt on hands, prompts steer the request - zero retraining
    the sentence:
      A frontier model is a pretrained next-token engine, taught manners
      by SFT, steered by preferences (RM/DPO) or checkers (RLVR),
      measured by evals, given fresh facts by RAG, and given hands by
      tools - and serving it is a separate discipline (Phase E).
    checks: shares sum to 100.0000% - pretraining share 99.9893% > 99.9% - serving stages train 0 parameters
  verdict: twelve days, four slots, one pipeline - if a new technique confuses you, ask which slot it changes.
"""
import textwrap

N = 7_000_000_000  # reference model: 7B parameters


def flops(tokens):
    """Day 51's rule: training compute ~ 6 * N * D FLOPs."""
    return 6 * N * tokens


# The manifest. tokens=None marks a stage that runs no
# gradient training: a gate (evals) or a serving-time
# attachment (RAG, tools, the agent loop, prompts).
STAGES = [
    # day, stage, input artifact, output artifact, data, train tokens
    (51, "Pretraining", "raw text corpus", "base model (theta)",
     "unlabeled tokens", 1_400_000_000_000),
    (52, "SFT", "base model", "instruct model",
     "(prompt, response) pairs", 60_000_000),
    (53, "LoRA run", "instruct model", "adapter (delta-W)",
     "task pairs, 0.1% of theta trained", 10_000_000),
    (54, "Reward model", "instruct model", "reward model r(x, y)",
     "chosen / rejected pairs", 20_000_000),
    (55, "DPO", "instruct model", "aligned model",
     "the same pairs, no RM trained", 10_000_000),
    (56, "RLVR", "instruct model", "reasoning model",
     "problems + a checker", 50_000_000),
    (57, "Evals", "candidate model", "scorecard: ship / no-ship",
     "benchmark sets (inference only)", None),
    (58, "RAG index", "his documents", "vector index",
     "chunks + embeddings", None),
    (59, "RAG loop", "vector index + question", "grounded answer",
     "top-k chunks in the prompt", None),
    (60, "Tool use", "aligned model", "model that calls functions",
     "tool schemas (JSON)", None),
    (61, "Agent loop", "model + tools", "agent: plan-act-observe",
     "tool results, fed back", None),
    (62, "Prompts", "any request", "structured request",
     "instructions + examples", None),
]

# ---------- Part 1: the artifact chain ----------
print("Part 1 - the stack as an artifact chain (Days 51-62)")
for day, stage, inp, out, data, _ in STAGES:
    print(f"  Day {day}  {stage:<13} {inp} -> {out}")
print("  read it bottom-up in the diagram: every arrow is a")
print("  handoff you could version, store, and roll back.")

# ---------- Part 2: who eats the compute ----------
print("Part 2 - training compute by stage (C ~ 6ND, Day 51)")
total = sum(flops(t) for *_, t in STAGES if t)
print(f"  {'day':>3}  {'stage':<13} {'tokens':>10}  "
      f"{'FLOPs':>9}  share")
for day, stage, _, _, _, t in STAGES:
    if t is None:
        print(f"  {day:>3}  {stage:<13} {'-':>10}  {'-':>9}  "
              "no training")
    else:
        share = 100.0 * flops(t) / total
        print(f"  {day:>3}  {stage:<13} {t:>10.2e}  "
              f"{flops(t):>9.2e}  {share:.4f}%")
pre_share = 100.0 * flops(STAGES[0][5]) / total
print(f"  total training compute: {total:.2e} FLOPs")
print(f"  one pretraining run = "
      f"{flops(STAGES[0][5]) // flops(STAGES[1][5]):,} SFT runs")
print("  everything after Day 51 is a rounding error in FLOPs -")
print("  which is exactly why the stack is economical: train the")
print("  engine once, steer it cheaply, attach facts and hands")
print("  at serving time instead of retraining.")

# ---------- Part 3: the slot map + the sentence ----------
print("Part 3 - the slot map")
SLOTS = [
    ("BUILD (51-53)", "make the engine, then teach it manners - "
     "LoRA is how you run these stages cheaply"),
    ("STEER (54-56)", "preferences point it (RM, DPO); checkers "
     "grade it (RLVR) - two ways to say 'better'"),
    ("MEASURE (57)", "evals are the gate between a checkpoint "
     "and a product - no scorecard, no ship"),
    ("SERVE-TIME (58-62)", "RAG bolts on fresh facts, tools bolt "
     "on hands, prompts steer the request - zero retraining"),
]
for name, desc in SLOTS:
    print(f"  {name:<18} {desc}")
SENTENCE = ("A frontier model is a pretrained next-token engine, "
            "taught manners by SFT, steered by preferences (RM/DPO) "
            "or checkers (RLVR), measured by evals, given fresh "
            "facts by RAG, and given hands by tools - and serving "
            "it is a separate discipline (Phase E).")
print("  the sentence:")
for line in textwrap.wrap(SENTENCE, 66):
    print(f"    {line}")
shares_sum = sum(100.0 * flops(t) / total
                 for *_, t in STAGES if t)
assert abs(shares_sum - 100.0) < 1e-6
assert pre_share > 99.9
print(f"  checks: shares sum to {shares_sum:.4f}% - pretraining "
      f"share {pre_share:.4f}% > 99.9% - serving stages train "
      "0 parameters")
print("verdict: twelve days, four slots, one pipeline - if a new "
      "technique confuses you, ask which slot it changes.")
