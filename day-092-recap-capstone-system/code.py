"""Day 92 - Recap: The Capstone System on One Page.

Phase G recap, same treatment as Days 20, 34, 50, 63
and 82: one carry sentence, a slot map of every day
in the phase, and a small artifact that makes the map
executable. Phase G (Days 83-91) built one system,
joint by joint: the terrain (83) and the trap (84),
features (85), labels (86), validation (87), two
baseline builds (88, 89), sizing (90) and the harness
that runs it (91). The sentence:

    A trading system is a pipeline with an honesty
    device at every joint - point-in-time features,
    first-touch labels, walk-forward grades, sized
    bets, a risk gate that can say no, and a journal
    that remembers what the model saw - and the
    model's accuracy is the least important number
    in it.

Days 88 and 89 are the proof of that last clause.
Day 88's baseline beat its majority-class baseline
(accuracy 0.5467 vs 0.5247) and still lost money:
-25.01% net of costs, Sharpe -0.16. Day 89's crypto
build scored 0.581 against a 0.547 baseline - and
buy-and-hold returned +95.4% over the same windows.
Accuracy went up in both builds. The account did not
follow. The numbers that decided the outcome lived
at the joints: costs, barriers, sizing, regime.

Tonight the recap becomes a MANIFEST plus a LINTER:

  Part 1  The manifest: every stage of the capstone
          pipeline - its days, what flows in, what
          flows out, the honesty device bolted onto
          the joint, and the failure that device
          exists to prevent.
  Part 2  The pre-flight audit: eight checks, one
          per rule Phase G taught, run against two
          configs. Config A is the tempting first
          draft - shuffled split, peeking features,
          free trades, 2x Kelly, no kill switch.
          Config B is the capstone config. The
          linter grades both before a single paper
          trade is allowed.
  Part 3  The totals and the sentence: the manifest
          covers Days 83-91 exactly once, the audit
          separates the two configs completely, and
          the one line to carry out of Phase G.

Pure stdlib, deterministic - no RNG, no torch, no
APIs, no downloads. Runs as-is in the ~/ai-lab venv
on the MacBook.

Expected output:

  Part 1 - the capstone manifest (Days 83-91)
    Ground rules (Day 83-84)
      in:  any research result  ->  out: a deflated prior
      honesty device: count every trial you ran
      failure it prevents: luck mistaken for edge
    Features (Day 85)
      in:  raw bars  ->  out: feature rows, as of t
      honesty device: point-in-time only
      failure it prevents: lookahead leakage
    Labels (Day 86)
      in:  bars and barriers  ->  out: first-touch labels
      honesty device: barriers fixed before the outcome
      failure it prevents: labels no real trade could earn
    Validation (Day 87)
      in:  features and labels  ->  out: walk-forward grades
      honesty device: train strictly before test, purged
      failure it prevents: grading your model on the future
    Baseline model (Day 88-89)
      in:  validated rows  ->  out: predictions
      honesty device: baseline first, net of costs
      failure it prevents: accuracy worship
    Sizing (Day 90)
      in:  predictions, edge estimate  ->  out: position sizes
      honesty device: fractional Kelly, vol-targeted
      failure it prevents: ruin by an overconfident edge
    Harness (Day 91)
      in:  sized orders  ->  out: paper fills and a journal
      honesty device: a risk gate that can say no
      failure it prevents: a bot nobody can audit or stop
    read it top to bottom and it is the whole of
    Phase G: every stage is ordinary engineering -
    the devices at the joints are the discipline.
  Part 2 - the pre-flight audit (the recap as a linter)
    config A: the tempting first draft
      FAIL  Day 84  trials counted - uncounted trials: luck never gets priced
      FAIL  Day 85  no lookahead - features must use bars up to t, never t+1
      FAIL  Day 86  first-touch labels - fixed-horizon labels grade trades you never took
      FAIL  Day 87  walk-forward split - a shuffled split grades the model on the future
      FAIL  Day 88  costs charged - free trades: Day 88's accuracy still lost -25.01%
      FAIL  Day 90  sizing capped - over full Kelly, one bad run ends the account
      FAIL  Day 91  kill switch armed - something must be able to say no, mid-flight
      FAIL  Day 91  journal on - every decision logged with what the model saw
      -> 8 of 8 checks FAIL - this config grades
         itself. Do not let it trade, even on paper.
    config B: the capstone config
      PASS  Day 84  trials counted
      PASS  Day 85  no lookahead
      PASS  Day 86  first-touch labels
      PASS  Day 87  walk-forward split
      PASS  Day 88  costs charged
      PASS  Day 90  sizing capped
      PASS  Day 91  kill switch armed
      PASS  Day 91  journal on
      -> 0 of 8 checks FAIL - cleared for paper
         trading (Day 91: paper first, always).
  Part 3 - the totals
    manifest: 7 stages, Days 83-91 each exactly once
    audit: 8 checks - config A fails 8, config B fails 0
    the sentence:
      A trading system is a pipeline with an honesty device at every
      joint - point-in-time features, first-touch labels, walk-forward
      grades, sized bets, a risk gate that can say no, and a journal
      that remembers what the model saw - and the model's accuracy is
      the least important number in it.
    checks: manifest covers Days 83-91 exactly once - the linter
            fails the tempting config on all 8 rules and passes
            the capstone config on all 8
  verdict: the model is one box in the pipeline - the system is
           the honesty devices around it.
"""
import textwrap

# The manifest: stage, days, input, output, the
# honesty device at the joint, the failure it prevents.
MANIFEST = [
    ("Ground rules", [83, 84], "any research result",
     "a deflated prior", "count every trial you ran",
     "luck mistaken for edge"),
    ("Features", [85], "raw bars",
     "feature rows, as of t", "point-in-time only",
     "lookahead leakage"),
    ("Labels", [86], "bars and barriers",
     "first-touch labels",
     "barriers fixed before the outcome",
     "labels no real trade could earn"),
    ("Validation", [87], "features and labels",
     "walk-forward grades",
     "train strictly before test, purged",
     "grading your model on the future"),
    ("Baseline model", [88, 89], "validated rows",
     "predictions", "baseline first, net of costs",
     "accuracy worship"),
    ("Sizing", [90], "predictions, edge estimate",
     "position sizes", "fractional Kelly, vol-targeted",
     "ruin by an overconfident edge"),
    ("Harness", [91], "sized orders",
     "paper fills and a journal",
     "a risk gate that can say no",
     "a bot nobody can audit or stop"),
]

# ---------- Part 1: the manifest ----------
print("Part 1 - the capstone manifest (Days 83-91)")
for stage, days, inn, out, device, failure in MANIFEST:
    tag = "-".join(map(str, days)) if len(days) > 1 else str(days[0])
    print(f"  {stage} (Day {tag})")
    print(f"    in:  {inn}  ->  out: {out}")
    print(f"    honesty device: {device}")
    print(f"    failure it prevents: {failure}")
print("  read it top to bottom and it is the whole of")
print("  Phase G: every stage is ordinary engineering -")
print("  the devices at the joints are the discipline.")

# The audit: one check per rule Phase G taught.
# Each rule: (day, check name, test, why it exists).
RULES = [
    (84, "trials counted",
     lambda c: c["trials_counted"],
     "uncounted trials: luck never gets priced"),
    (85, "no lookahead",
     lambda c: c["lookahead_bars"] == 0,
     "features must use bars up to t, never t+1"),
    (86, "first-touch labels",
     lambda c: c["label_mode"] == "triple_barrier",
     "fixed-horizon labels grade trades you never took"),
    (87, "walk-forward split",
     lambda c: c["split"] == "walk_forward",
     "a shuffled split grades the model on the future"),
    (88, "costs charged",
     lambda c: c["costs_bps"] > 0,
     "free trades: Day 88's accuracy still lost -25.01%"),
    (90, "sizing capped",
     lambda c: 0 < c["kelly_fraction"] <= 1.0,
     "over full Kelly, one bad run ends the account"),
    (91, "kill switch armed",
     lambda c: c["kill_switch"],
     "something must be able to say no, mid-flight"),
    (91, "journal on",
     lambda c: c["journal"],
     "every decision logged with what the model saw"),
]

CONFIG_A = {  # the tempting first draft
    "trials_counted": False, "lookahead_bars": 1,
    "label_mode": "fixed_horizon", "split": "shuffle",
    "costs_bps": 0, "kelly_fraction": 2.0,
    "kill_switch": False, "journal": False,
}
CONFIG_B = {  # the capstone config
    "trials_counted": True, "lookahead_bars": 0,
    "label_mode": "triple_barrier", "split": "walk_forward",
    "costs_bps": 5, "kelly_fraction": 0.5,
    "kill_switch": True, "journal": True,
}


def audit(name, cfg):
    """Run every rule against one config; return fails."""
    print(f"  config {name}")
    fails = 0
    for day, check, test, why in RULES:
        if test(cfg):
            print(f"    PASS  Day {day}  {check}")
        else:
            fails += 1
            print(f"    FAIL  Day {day}  {check} - {why}")
    return fails


# ---------- Part 2: the pre-flight audit ----------
print("Part 2 - the pre-flight audit (the recap as a linter)")
fails_a = audit("A: the tempting first draft", CONFIG_A)
print(f"    -> {fails_a} of {len(RULES)} checks FAIL - this "
      "config grades")
print("       itself. Do not let it trade, even on paper.")
fails_b = audit("B: the capstone config", CONFIG_B)
print(f"    -> {fails_b} of {len(RULES)} checks FAIL - "
      "cleared for paper")
print("       trading (Day 91: paper first, always).")

# ---------- Part 3: totals + the sentence ----------
print("Part 3 - the totals")
covered = sorted(d for _, days, *_ in MANIFEST for d in days)
print(f"  manifest: {len(MANIFEST)} stages, Days 83-91 "
      "each exactly once")
print(f"  audit: {len(RULES)} checks - config A fails "
      f"{fails_a}, config B fails {fails_b}")
SENTENCE = ("A trading system is a pipeline with an honesty "
            "device at every joint - point-in-time features, "
            "first-touch labels, walk-forward grades, sized "
            "bets, a risk gate that can say no, and a journal "
            "that remembers what the model saw - and the "
            "model's accuracy is the least important number "
            "in it.")
print("  the sentence:")
for line in textwrap.wrap(SENTENCE, 64):
    print(f"    {line}")
assert covered == list(range(83, 92))
assert fails_a == len(RULES) and fails_b == 0
print("  checks: manifest covers Days 83-91 exactly once - "
      "the linter")
print("          fails the tempting config on all 8 rules "
      "and passes")
print("          the capstone config on all 8")
print("verdict: the model is one box in the pipeline - the "
      "system is")
print("         the honesty devices around it.")
