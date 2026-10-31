"""Day 61 - Agents 2: The Agentic Loop (Plan -> Act -> Observe)

Day 60 gave a model hands: tool schemas + one call/observation round-trip.
Today we put that round-trip in a LOOP and it becomes an agent: the planner
picks a tool, sees the result, picks the next one, until the task is done.

The model is NOT the lesson - the control system around it is:
  * state        = the message history, which grows by 2 entries per step
  * stop rules   = final answer | max_steps budget | repeated-call detector

The planner here is a small rule-based stand-in for an LLM (named honestly:
a real model returns the same {"tool": ..., "args": ...} decisions, just
less predictably). Stdlib only, embedded data, fully deterministic.
Run in the ~/ai-lab venv:  python3 code.py
"""

import json

# ---------------------------------------------------------------- the world
CLOSES = {  # this week's daily closes, Mon..Fri (embedded, no API)
    "NVDA": [200.00, 204.00, 202.00, 208.00, 212.40],
    "AMD": [160.00, 158.40, 161.00, 159.50, 156.80],
}


class World:
    """The tools from Day 60, same schemas, now called repeatedly."""

    def __init__(self):
        self.watchlist = ["META"]

    def get_closes(self, ticker):
        return {"ticker": ticker, "closes": CLOSES[ticker]}

    def pct_change(self, ticker, first, last):
        return {"ticker": ticker, "pct": round((last - first) / first * 100, 2)}

    def add_to_watchlist(self, ticker):
        if ticker not in self.watchlist:
            self.watchlist.append(ticker)
        return {"added": ticker, "watchlist": list(self.watchlist)}


# ------------------------------------------------------------- the planners
def results_by(history, tool):
    """All observations so far for one tool, keyed by ticker."""
    out = {}
    for msg in history:
        if msg["role"] == "tool" and msg["name"] == tool:
            out[msg["result"]["ticker"]] = msg["result"]
    return out


def planner(history):
    """Rule-based stand-in for the LLM. Reads ONLY the history - exactly
    like a real model, which sees nothing but the conversation."""
    closes = results_by(history, "get_closes")
    for ticker in ("NVDA", "AMD"):
        if ticker not in closes:
            return {"thought": f"Need {ticker}'s closes before I can compare.",
                    "tool": "get_closes", "args": {"ticker": ticker}}
    changes = results_by(history, "pct_change")
    for ticker in ("NVDA", "AMD"):
        if ticker not in changes:
            c = closes[ticker]["closes"]
            return {"thought": f"Have {ticker} closes; compute the week's move.",
                    "tool": "pct_change",
                    "args": {"ticker": ticker, "first": c[0], "last": c[-1]}}
    added = [m for m in history
             if m["role"] == "tool" and m["name"] == "add_to_watchlist"]
    if not added:
        stronger = max(changes, key=lambda t: changes[t]["pct"])
        return {"thought": f"{stronger} is stronger "
                           f"({changes[stronger]['pct']:+.2f}%). Add it.",
                "tool": "add_to_watchlist", "args": {"ticker": stronger}}
    nvda, amd = changes["NVDA"]["pct"], changes["AMD"]["pct"]
    return {"thought": "Task complete - both compared, stronger one added.",
            "final": f"NVDA {nvda:+.2f}% beat AMD {amd:+.2f}% this week; "
                     f"NVDA added to the watchlist."}


def stubborn_planner(history):
    """A broken planner: whatever it observes, it asks for NVDA closes."""
    return {"thought": "Get NVDA's closes.",
            "tool": "get_closes", "args": {"ticker": "NVDA"}}


# ------------------------------------------------------------ the loop itself
def run_agent(task, plan_fn, world, max_steps=8):
    """THE AGENTIC LOOP. ~25 lines - this is the whole runtime."""
    tools = {"get_closes": world.get_closes,
             "pct_change": world.pct_change,
             "add_to_watchlist": world.add_to_watchlist}
    history = [{"role": "user", "text": task}]   # the state. It only grows.
    seen_calls = set()
    print(f"TASK: {task}")
    for step in range(1, max_steps + 1):
        decision = plan_fn(history)                       # PLAN
        print(f" step {step}: thought : {decision['thought']}")
        if "final" in decision:                           # stop rule 1
            history.append({"role": "assistant", "text": decision["final"]})
            print(f"         FINAL   : {decision['final']}")
            print(f"         stopped : final answer at step {step} "
                  f"| history = {len(history)} messages")
            return history
        call = (decision["tool"],
                json.dumps(decision["args"], sort_keys=True))
        if call in seen_calls:                            # stop rule 2
            print(f"         call    : {call[0]}({call[1]})")
            print(f"         STOPPED : repeated call detected - "
                  f"loop killed at step {step}, task NOT completed")
            return history
        seen_calls.add(call)
        result = tools[decision["tool"]](**decision["args"])   # ACT
        history.append({"role": "assistant", "call": call})    # record both
        history.append({"role": "tool", "name": decision["tool"],
                        "result": result})                     # OBSERVE
        print(f"         call    : {call[0]}({call[1]})")
        print(f"         observe : {json.dumps(result, sort_keys=True)}")
        print(f"         history : {len(history)} messages")
    print(f" STOPPED : max_steps={max_steps} spent, no final answer")  # rule 3
    return history


# -------------------------------------------------------------------- runs
TASK = "Compare NVDA and AMD this week; add the stronger one to my watchlist."

print("=== RUN 1: a working agent (planner + loop + tools) ===")
w1 = World()
run_agent(TASK, planner, w1)
print(f" watchlist after run 1: {w1.watchlist}")

print()
print("=== RUN 2: same loop, broken planner (repeats itself) ===")
w2 = World()
run_agent(TASK, stubborn_planner, w2)
print(f" watchlist after run 2: {w2.watchlist}  (unchanged - killed in time)")
