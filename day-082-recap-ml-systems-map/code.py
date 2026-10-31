"""Day 82 - Recap: The ML Systems Map - Where Your
Infra Career Plugs In.

Phase F recap, same treatment as Days 20, 34, 50 and
63: one carry sentence, a slot map of every day in
the phase, and a small artifact that makes the map
executable. Phase F (Days 73-81) was the systems
phase: the memory wall, the ways training gets split
across GPUs, mixed precision, checkpointing, the GPU
itself, inference at scale, and profiling. The
sentence:

    Every Phase F problem is a distributed-systems
    problem in new clothes - state too big for one
    node, work sharded across many, state checksummed
    and repaired, traffic shaped at the edge - and
    the scarcest seats in AI sit where that machinery
    meets the model.

Tonight the map becomes a TRANSLATION TABLE plus a
READINESS INVENTORY:

  Part 1  The map: each Phase F day, the name the
          same idea already has in your infra world,
          and a verdict - HOME TURF (you have shipped
          this), STRONG TRANSFER (the concepts are
          yours, the machinery is new), or NEW GROUND
          (honestly new: name it, never bluff it).
  Part 2  The inventory: for every row, the evidence
          already on the record, and the one honest
          gap. The record, and nothing beyond it:
          Staff SWE on Conveyor, Tesla's exabyte-scale
          Rust object store for AI training, and
          author of its Data Reliability Service;
          AWS S3 core and DynamoDB team lead (Tiered
          Storage launch); traffic shaping team since
          Aug 2026. Invented claims die in interviews;
          these nine rows don't need any.
  Part 3  The totals and the sentence: how the nine
          days sort, and the one line to carry into
          the interview loop.

Pure stdlib, deterministic - no RNG, no torch, no
APIs, no downloads. Runs as-is in the ~/ai-lab venv
on the MacBook.

Expected output:

  Part 1 - the translation table (Days 73-81)
    day  Phase F                      in your world                verdict
     73  The memory wall              capacity planning            STRONG TRANSFER
     74  Data parallelism             sharded workers + gossip     STRONG TRANSFER
     75  Tensor + pipeline parallel   striping one job             NEW GROUND
     76  FSDP / DeepSpeed             consistent sharding of state STRONG TRANSFER
     77  Mixed precision (bf16)       tiered copies of the data    NEW GROUND
     78  Checkpointing + repair       backup, restore, repair      HOME TURF
     79  GPU / CUDA mental model      a new machine architecture   NEW GROUND
     80  Inference at scale           split paths, shaped traffic  STRONG TRANSFER
     81  Profile a training step      observability, as always     HOME TURF
    every row is the same idea wearing two vocabularies:
    the left column is what the ML team calls it, the
    middle column is what you have been calling it.
  Part 2 - readiness inventory: evidence + the honest gap
    Day 73  The memory wall
      evidence: capacity math at exabyte scale (Conveyor storage)
      gap:      training-job budgets (activations, optimizer states) are new
      arithmetic
    Day 74  Data parallelism
      evidence: sharded data systems at S3 / DynamoDB scale
      gap:      no multi-GPU training job run yet - all-reduce met in simulation
      (Day 74)
    Day 75  Tensor + pipeline parallel
      evidence: striping / pipelining intuition from storage data paths
      gap:      no hands-on model splitting (Megatron-style) yet
    Day 76  FSDP / DeepSpeed
      evidence: placement + sharding design (DynamoDB partitions, Conveyor)
      gap:      FSDP / DeepSpeed not yet operated on a real cluster
    Day 77  Mixed precision (bf16)
      evidence: tiering instinct: led the DynamoDB Tiered Storage launch
      gap:      numerics judgement (bf16 range, loss scaling) is new
    Day 78  Checkpointing + repair
      evidence: authored the Data Reliability Service on Conveyor
      gap:      training-checkpoint tooling (async, distributed) not yet operated
    Day 79  GPU / CUDA mental model
      evidence: deep systems fundamentals - no GPU claims made
      gap:      CUDA depth: mental model only, no kernels written - the big honest
      gap
    Day 80  Inference at scale
      evidence: traffic shaping team since Aug 2026: routing, load, edges
      gap:      no production inference fleet operated yet
    Day 81  Profile a training step
      evidence: reliability + observability work on Conveyor / DRS
      gap:      GPU profilers (Nsight) and training-loop ownership still ahead
    the gaps are all of one kind - machinery not yet
    operated - and none of them is 'systems thinking'.
    That is the strongest sentence in the inventory.
  Part 3 - the totals
    HOME TURF:       2 of 9 - checkpointing/repair, profiling
    STRONG TRANSFER: 4 of 9 - memory wall, data parallelism, FSDP, inference at scale
    NEW GROUND:      3 of 9 - model parallelism, mixed precision, the GPU itself
    the sentence:
      Every Phase F problem is a distributed-systems problem in new
      clothes - state too big for one node, work sharded across many,
      state checksummed and repaired, traffic shaped at the edge - and
      the scarcest seats in AI sit where that machinery meets the model.
    checks: 9 rows, Days 73-81 each exactly once - verdicts sum to 9 - home-turf rows are the reliability rows (78, 81)
  verdict: you are not entering AI from outside - you are translating a decade of systems work into its vocabulary, gaps named out loud.
"""
import textwrap

# The map: day, Phase F concept, the infra-world name
# for the same idea, and the honest verdict.
MAP = [
    (73, "The memory wall", "capacity planning",
     "STRONG TRANSFER"),
    (74, "Data parallelism", "sharded workers + gossip",
     "STRONG TRANSFER"),
    (75, "Tensor + pipeline parallel", "striping one job",
     "NEW GROUND"),
    (76, "FSDP / DeepSpeed", "consistent sharding of state",
     "STRONG TRANSFER"),
    (77, "Mixed precision (bf16)", "tiered copies of the data",
     "NEW GROUND"),
    (78, "Checkpointing + repair", "backup, restore, repair",
     "HOME TURF"),
    (79, "GPU / CUDA mental model", "a new machine architecture",
     "NEW GROUND"),
    (80, "Inference at scale", "split paths, shaped traffic",
     "STRONG TRANSFER"),
    (81, "Profile a training step", "observability, as always",
     "HOME TURF"),
]

# The inventory: evidence strictly from the record,
# plus the one honest gap per row.
INVENTORY = [
    (73, "capacity math at exabyte scale (Conveyor storage)",
     "training-job budgets (activations, optimizer states) "
     "are new arithmetic"),
    (74, "sharded data systems at S3 / DynamoDB scale",
     "no multi-GPU training job run yet - all-reduce met "
     "in simulation (Day 74)"),
    (75, "striping / pipelining intuition from storage "
     "data paths",
     "no hands-on model splitting (Megatron-style) yet"),
    (76, "placement + sharding design (DynamoDB "
     "partitions, Conveyor)",
     "FSDP / DeepSpeed not yet operated on a real cluster"),
    (77, "tiering instinct: led the DynamoDB Tiered "
     "Storage launch",
     "numerics judgement (bf16 range, loss scaling) is new"),
    (78, "authored the Data Reliability Service on Conveyor",
     "training-checkpoint tooling (async, distributed) "
     "not yet operated"),
    (79, "deep systems fundamentals - no GPU claims made",
     "CUDA depth: mental model only, no kernels written - "
     "the big honest gap"),
    (80, "traffic shaping team since Aug 2026: routing, "
     "load, edges",
     "no production inference fleet operated yet"),
    (81, "reliability + observability work on Conveyor / DRS",
     "GPU profilers (Nsight) and training-loop ownership "
     "still ahead"),
]

NAMES = {day: concept for day, concept, _, _ in MAP}

# ---------- Part 1: the translation table ----------
print("Part 1 - the translation table (Days 73-81)")
print(f"  {'day':>3}  {'Phase F':<28} {'in your world':<28} verdict")
for day, concept, infra, verdict in MAP:
    print(f"  {day:>3}  {concept:<28} {infra:<28} {verdict}")
print("  every row is the same idea wearing two vocabularies:")
print("  the left column is what the ML team calls it, the")
print("  middle column is what you have been calling it.")

# ---------- Part 2: readiness inventory ----------
print("Part 2 - readiness inventory: evidence + the honest gap")
for day, evidence, gap in INVENTORY:
    print(f"  Day {day}  {NAMES[day]}")
    for line in textwrap.wrap("evidence: " + evidence, 76):
        print(f"    {line}")
    for line in textwrap.wrap("gap:      " + gap, 76):
        print(f"    {line}")
print("  the gaps are all of one kind - machinery not yet")
print("  operated - and none of them is 'systems thinking'.")
print("  That is the strongest sentence in the inventory.")

# ---------- Part 3: totals + the sentence ----------
print("Part 3 - the totals")
counts = {}
for *_, verdict in MAP:
    counts[verdict] = counts.get(verdict, 0) + 1
home = [d for d, _, _, v in MAP if v == "HOME TURF"]
print(f"  HOME TURF:       {counts['HOME TURF']} of 9 - "
      "checkpointing/repair, profiling")
print(f"  STRONG TRANSFER: {counts['STRONG TRANSFER']} of 9 - "
      "memory wall, data parallelism, FSDP, inference at scale")
print(f"  NEW GROUND:      {counts['NEW GROUND']} of 9 - "
      "model parallelism, mixed precision, the GPU itself")
SENTENCE = ("Every Phase F problem is a distributed-systems "
            "problem in new clothes - state too big for one "
            "node, work sharded across many, state checksummed "
            "and repaired, traffic shaped at the edge - and "
            "the scarcest seats in AI sit where that machinery "
            "meets the model.")
print("  the sentence:")
for line in textwrap.wrap(SENTENCE, 66):
    print(f"    {line}")
days = [d for d, *_ in MAP]
assert days == list(range(73, 82))
assert sum(counts.values()) == 9
assert home == [78, 81]
print(f"  checks: 9 rows, Days 73-81 each exactly once - "
      f"verdicts sum to 9 - home-turf rows are the "
      f"reliability rows ({', '.join(map(str, home))})")
print("verdict: you are not entering AI from outside - you are "
      "translating a decade of systems work into its vocabulary, "
      "gaps named out loud.")
