"""Day 95 - Deep Dive: RAG Over Your Trading Journal.

Day 59 pointed the tiny RAG at lesson notes. Today the same
engine is pointed at the corpus where a wrong answer costs
the most: your trading journal - the entries your tradebot
collects (date, ticker, action, and the reason you typed at
the time). The journal below is FICTIONAL: 24 invented
entries in your journal's format, standing in for the real
thing. None of these are real trades.

Doc-RAG ranked the whole corpus by word overlap and gated
on the best score. A journal breaks that design in three
places, and each break gets a fix today:

  1. Some questions are EXACT, not fuzzy. "My META entries"
     is a WHERE clause, not a vibe. So retrieval is now
     two-stage: a metadata filter (ticker, date range)
     slices the journal deterministically, and only then
     does Day 59's TF-IDF rank inside the slice. Hybrid
     retrieval: SQL brain first, fuzzy brain second.

  2. Some questions are about EVENTS in time. "Why did I
     exit in October?" is answered by the entries AROUND
     the exit - the trim before it, the note where the
     exit triggers were written down, the post-exit review
     after it. So the system can expand an anchor entry
     into a window on that ticker's timeline.

  3. The refusal gate matters MORE here. A hallucinated
     cake temperature is funny; a hallucinated memory of
     why you sold is a false fact about your own money.
     The gate now also fires BEFORE ranking: if the filter
     matches nothing, there is nothing to rank, and the
     system refuses without ever consulting the fuzzy brain.

No chunking this time: a journal entry is already atomic -
one date, one action, one reason. The citation changes
shape to match: [date TICKER action] names the exact
journal row a sentence came from.

  Part 1  The journal as an index: 24 fictional entries,
          Day 59's TF-IDF engine, one row per entry.
  Part 2  Stage 1 vs no stage 1: the same question ranked
          over the whole journal, then inside the META
          slice - watch what the filter removes for free.
  Part 3  Three questions end to end: filter + rank with
          citations; an event window around the October
          exit; and a ticker never traded, refused at the
          filter, ranking never consulted.

Runs anywhere: Python 3.11+ standard library ONLY (math,
re). No torch, no APIs, no downloads, no RNG - every score
below is exact and reproducible digit for digit.

Expected output:
  Part 1 - THE JOURNAL AS AN INDEX: one row per entry, no chunking
    24 fictional entries, 2026-06-09 .. 2026-10-27
    per ticker: AMD 5  META 8  NVDA 6  PLTR 5
    no chunking: an entry is already atomic - one date, one
    action, one reason. The row IS the chunk (Day 59 contrast).
    vocabulary = 258 terms; index shape: 24 entry vectors x 258 dims
  Part 2 - STAGE 1 FIRST: same question, whole journal vs the slice
    Q: "was the trim about the thesis or about position size" (no ticker named in the words)
    ranked over ALL 24 entries (Day 59's way):
      #1  [2026-06-23 PLTR buy   ] cos 0.2403
      #2  [2026-09-15 META trim  ] cos 0.2228
      #3  [2026-10-13 NVDA trim  ] cos 0.1550
    stage 1 filter ticker=META: 8 of 24 entries survive
    ranked inside the META slice only:
      #1  [2026-09-15 META trim  ] cos 0.2228
      #2  [2026-10-26 META note  ] cos 0.1276
      #3  [2026-07-28 META add   ] cos 0.1195
    the filter does not make the right entry score higher - it
    makes the wrong-ticker entries IMPOSSIBLE. Exact beats fuzzy
    wherever the question is exact.
  Part 3 - END TO END: filter -> rank -> gate, or the window, or refuse
    Q1: "why did I trim META in September - thesis or position size"
      stage 1: ticker=META -> 8 entries; stage 2 top score 0.3384
      gate: 0.3384 >= 0.25 -> ANSWER
        "Reason: position size, not thesis - it had grown past my single-name limit." [2026-09-15 META trim]
    Q2: "why did I exit META in October"
      stage 1: ticker=META, month=10 -> 3 entries; anchor = the sell
      event window on META's timeline (reaches past the month filter):
         [2026-09-15 META trim  ] Trimmed a third into the rally.
         [2026-10-07 META note  ] Pre-earnings jitters in the group.
      >> [2026-10-21 META sell  ] Exited the rest after the print.
         [2026-10-26 META note  ] Post-exit review: the June thesis died on guidance, not price.
      ANSWER (the anchor entry states its own reason):
        "Exited the rest after the print." [2026-10-21 META sell]
        "Guidance cut the ad growth outlook and capex jumped again; the thesis I wrote in June no longer matches the numbers, so I closed it rather than argue with the tape." [2026-10-21 META sell]
    Q3: "why did I buy the COIN dip in August"
      stage 1: ticker=COIN -> 0 of 24 entries
      stage 2: SKIPPED - no candidates, ranking never runs
      REFUSE: this journal has no COIN entries, so there is
              nothing to remember. A bare LLM, asked why you
              bought a dip you never traded, would invent a
              reason. The filter makes the invention impossible.
    verdict: exact questions get exact filters, fuzzy questions
             get ranked inside the slice, events get a window -
             and every path still ends at the gate.
"""
import math
import re
from collections import Counter

# The journal: 24 FICTIONAL entries in tradebot format -
# date, ticker, action, and the reason typed at the time.
# Invented for this lesson; NOT real trades.
JOURNAL = [
    {"date": "2026-06-09", "ticker": "META", "action": "buy",
     "note": "Opened a starter position after the post-earnings dip. "
             "Thesis: ad revenue still compounding and the valuation "
             "finally looks reasonable."},
    {"date": "2026-06-16", "ticker": "NVDA", "action": "buy",
     "note": "Started a position ahead of the data center print. "
             "Hyperscaler capex commentary keeps pointing at more "
             "accelerators."},
    {"date": "2026-06-23", "ticker": "PLTR", "action": "buy",
     "note": "Small starter. Commercial revenue growth is the number "
             "I care about; government work is the base load."},
    {"date": "2026-06-30", "ticker": "META", "action": "note",
     "note": "Reality Labs losses widened again. Watching whether "
             "management guides capex up at the next print."},
    {"date": "2026-07-07", "ticker": "NVDA", "action": "note",
     "note": "Supply chatter: lead times normalizing. Not a thesis "
             "break, but the scarcity premium may fade."},
    {"date": "2026-07-14", "ticker": "AMD", "action": "buy",
     "note": "Starter position as the second-source play on AI "
             "accelerators. Smaller size than NVDA because the "
             "software moat is thinner."},
    {"date": "2026-07-28", "ticker": "META", "action": "add",
     "note": "Added on strength after the earnings beat. Ad "
             "impressions up, pricing held, and the AI capex guide "
             "was raised without the stock selling off."},
    {"date": "2026-07-30", "ticker": "PLTR", "action": "note",
     "note": "Bootcamp conversions look real but lumpy. I want two "
             "more quarters of evidence before sizing up."},
    {"date": "2026-08-11", "ticker": "AMD", "action": "note",
     "note": "Accelerator traction is the whole thesis. I need "
             "design-win evidence, not roadmap slides."},
    {"date": "2026-08-19", "ticker": "META", "action": "note",
     "note": "Position is now my largest. Reminder to self: "
             "concentration is a risk even when the thesis is intact."},
    {"date": "2026-08-26", "ticker": "NVDA", "action": "add",
     "note": "Added after earnings. Data center revenue beat and the "
             "guide implied another step up in accelerator shipments."},
    {"date": "2026-09-02", "ticker": "PLTR", "action": "add",
     "note": "Added after the commercial segment accelerated again. "
             "Growth at this quality is hard to ignore."},
    {"date": "2026-09-08", "ticker": "AMD", "action": "sell",
     "note": "Sold the starter. Two quarters without the design wins "
             "I asked for in July; capital works harder elsewhere. "
             "Small loss, thesis exit, not a price exit."},
    {"date": "2026-09-15", "ticker": "META", "action": "trim",
     "note": "Trimmed a third into the rally. Reason: position size, "
             "not thesis - it had grown past my single-name limit."},
    {"date": "2026-09-22", "ticker": "NVDA", "action": "note",
     "note": "Customer concentration is the bear case I take "
             "seriously: two buyers are a huge share of revenue."},
    {"date": "2026-09-29", "ticker": "PLTR", "action": "note",
     "note": "Valuation is the whole argument against. Multiple "
             "expansion did the heavy lifting this year, not estimates."},
    {"date": "2026-09-30", "ticker": "AMD", "action": "note",
     "note": "Watchlist only now. Would revisit on a named "
             "hyperscaler deployment, in size."},
    {"date": "2026-10-07", "ticker": "META", "action": "note",
     "note": "Pre-earnings jitters in the group. Decided to hold "
             "through the print but wrote down my exit triggers "
             "in advance."},
    {"date": "2026-10-13", "ticker": "NVDA", "action": "trim",
     "note": "Trimmed into all-time highs to fund other ideas. Still "
             "bullish; this was sizing discipline, not a verdict "
             "on the quarter."},
    {"date": "2026-10-15", "ticker": "AMD", "action": "note",
     "note": "Still no deployment news. Staying patient; no "
             "fear-of-missing-out re-entry."},
    {"date": "2026-10-19", "ticker": "PLTR", "action": "note",
     "note": "No action. Watching whether US commercial growth "
             "holds up at the next print."},
    {"date": "2026-10-21", "ticker": "META", "action": "sell",
     "note": "Exited the rest after the print. Guidance cut the ad "
             "growth outlook and capex jumped again; the thesis I "
             "wrote in June no longer matches the numbers, so I "
             "closed it rather than argue with the tape."},
    {"date": "2026-10-26", "ticker": "META", "action": "note",
     "note": "Post-exit review: the June thesis died on guidance, "
             "not price. Rule reinforced - when the stated reason "
             "breaks, the position goes."},
    {"date": "2026-10-27", "ticker": "NVDA", "action": "note",
     "note": "Holding the remainder. Next checkpoint is the January "
             "print and any export-control headlines."},
]

K = 3                # top-k retrieval
THRESHOLD = 0.25     # Day 59's gate, unchanged: best cosine must clear this


def tokenize(text):
    """Lowercase word tokens; punctuation is dropped. (Day 59, verbatim.)"""
    return re.findall(r"[a-z0-9]+", text.lower())


MONTHS = ["january", "february", "march", "april", "may", "june",
          "july", "august", "september", "october", "november",
          "december"]


def entry_text(e):
    """What the index sees: the note, plus the metadata as words -
    ticker, action, and the month name - so 'September' is
    searchable text too. The stage-1 filter stays the exact path;
    this only helps the fuzzy ranker."""
    month = MONTHS[int(e["date"][5:7]) - 1]
    return f"{e['ticker']} {e['action']} {month} {e['note']}"


def entry_id(e):
    """The journal citation: one exact row. [2026-09-15 META trim]"""
    return f"{e['date']} {e['ticker']} {e['action']}"


# ---------- Part 1: THE JOURNAL AS AN INDEX (Day 59's engine) --------------
print("Part 1 - THE JOURNAL AS AN INDEX: one row per entry, no chunking")
counts = Counter(e["ticker"] for e in JOURNAL)
print(f"  {len(JOURNAL)} fictional entries, "
      f"{JOURNAL[0]['date']} .. {JOURNAL[-1]['date']}")
print("  per ticker: " + "  ".join(f"{t} {counts[t]}"
                                   for t in sorted(counts)))
print("  no chunking: an entry is already atomic - one date, one")
print("  action, one reason. The row IS the chunk (Day 59 contrast).")

entry_tokens = [tokenize(entry_text(e)) for e in JOURNAL]
vocab = sorted({t for toks in entry_tokens for t in toks})
V = {w: i for i, w in enumerate(vocab)}
N = len(JOURNAL)
df = Counter()
for toks in entry_tokens:
    df.update(set(toks))
idf = {w: math.log((1 + N) / (1 + df[w])) + 1.0 for w in vocab}


def embed(tokens):
    """TF-IDF vector: count of each word, weighted by its idf.
    (Day 59, verbatim.)"""
    counts_ = Counter(tokens)
    vec = [0.0] * len(vocab)
    for w, c in counts_.items():
        if w in V:                       # unknown words are ignored
            vec[V[w]] = c * idf[w]
    return vec


index = [embed(toks) for toks in entry_tokens]
print(f"  vocabulary = {len(vocab)} terms; index shape: "
      f"{len(index)} entry vectors x {len(vocab)} dims")


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ---------- The two stages --------------------------------------------------
def stage1_filter(ticker=None, month=None):
    """STAGE 1 - exact metadata filter. Returns journal indices.
    Deterministic: a row matches or it does not. No scores."""
    return [i for i, e in enumerate(JOURNAL)
            if (ticker is None or e["ticker"] == ticker)
            and (month is None or e["date"][5:7] == month)]


def stage2_rank(query, subset, k=K):
    """STAGE 2 - Day 59's TF-IDF cosine rank, INSIDE the slice only."""
    qv = embed(tokenize(query))
    scored = [(cosine(qv, index[i]), i) for i in subset]
    scored.sort(key=lambda p: (-p[0], p[1]))
    return scored[:k]


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


def extract_answer(query, hits, n=2):
    """Extractive assembly, Day 59's rules with one journal
    twist: lift the best COMPLETE sentences from the TOP
    entry's note only, verbatim, each printed with its
    [date TICKER action] citation. Day 59 read the top 2
    chunks because a chunk is a fragment of a document;
    a journal entry is a complete record of one event,
    so the answer quotes the event itself."""
    qtoks = set(tokenize(query))
    cands = []
    for _, i in hits[:1]:
        for si, sent in enumerate(sentences(JOURNAL[i]["note"])):
            if not (sent[0].isupper() and sent[-1] in ".!?"):
                continue                 # a fragment, not a sentence
            shared = qtoks & set(tokenize(sent))
            s = sum(idf[w] for w in shared if w in idf)
            if s > 0:
                cands.append((s, i, si, sent))
    if not cands:
        return []
    floor = 0.4 * max(s for s, _, _, _ in cands)
    strong = [c for c in cands if c[0] >= floor]
    strong.sort(key=lambda t: (-t[0], t[1], t[2]))
    picked = sorted(strong[:n], key=lambda t: (t[1], t[2]))
    return [(sent, entry_id(JOURNAL[i])) for _, i, _, sent in picked]


# ---------- Part 2: WHY THE FILTER GOES FIRST -------------------------------
print("Part 2 - STAGE 1 FIRST: same question, whole journal vs the slice")
Q = "was the trim about the thesis or about position size"
print(f"  Q: \"{Q}\" (no ticker named in the words)")
all_hits = stage2_rank(Q, range(len(JOURNAL)))
print("  ranked over ALL 24 entries (Day 59's way):")
for rank, (score, i) in enumerate(all_hits, start=1):
    print(f"    #{rank}  [{entry_id(JOURNAL[i]):<22}] cos {score:.4f}")
meta_slice = stage1_filter(ticker="META")
print(f"  stage 1 filter ticker=META: {len(meta_slice)} of 24 entries survive")
meta_hits = stage2_rank(Q, meta_slice)
print("  ranked inside the META slice only:")
for rank, (score, i) in enumerate(meta_hits, start=1):
    print(f"    #{rank}  [{entry_id(JOURNAL[i]):<22}] cos {score:.4f}")
print("  the filter does not make the right entry score higher - it")
print("  makes the wrong-ticker entries IMPOSSIBLE. Exact beats fuzzy")
print("  wherever the question is exact.")

# ---------- Part 3: THREE QUESTIONS, END TO END -----------------------------
print("Part 3 - END TO END: filter -> rank -> gate, or the window, or refuse")

# Q1: filter + rank, answered with citations.
Q1 = "why did I trim META in September - thesis or position size"
print(f"  Q1: \"{Q1}\"")
q1_hits = stage2_rank(Q1, meta_slice)
best = q1_hits[0][0]
print(f"    stage 1: ticker=META -> {len(meta_slice)} entries; "
      f"stage 2 top score {best:.4f}")
if best < THRESHOLD:
    print(f"    gate: {best:.4f} < {THRESHOLD:.2f} -> REFUSE")
else:
    print(f"    gate: {best:.4f} >= {THRESHOLD:.2f} -> ANSWER")
    for sent, cid in extract_answer(Q1, q1_hits):
        print(f"      \"{sent}\" [{cid}]")

# Q2: the event window around the October META exit.
print("  Q2: \"why did I exit META in October\"")
oct_meta = stage1_filter(ticker="META", month="10")
anchor = next(i for i in oct_meta if JOURNAL[i]["action"] == "sell")
timeline = stage1_filter(ticker="META")          # META's own timeline
pos = timeline.index(anchor)
window = timeline[max(0, pos - 2):pos + 2]        # 2 before, anchor, 1 after
print(f"    stage 1: ticker=META, month=10 -> {len(oct_meta)} entries; "
      f"anchor = the sell")
print("    event window on META's timeline (reaches past the month filter):")
for i in window:
    mark = ">>" if i == anchor else "  "
    print(f"    {mark} [{entry_id(JOURNAL[i]):<22}] "
          f"{sentences(JOURNAL[i]['note'])[0]}")
print("    ANSWER (the anchor entry states its own reason):")
for sent in sentences(JOURNAL[anchor]["note"]):
    print(f"      \"{sent}\" [{entry_id(JOURNAL[anchor])}]")

# Q3: a ticker never traded - refused at the filter, before ranking.
print("  Q3: \"why did I buy the COIN dip in August\"")
coin_slice = stage1_filter(ticker="COIN")
print(f"    stage 1: ticker=COIN -> {len(coin_slice)} of 24 entries")
if not coin_slice:
    print("    stage 2: SKIPPED - no candidates, ranking never runs")
    print("    REFUSE: this journal has no COIN entries, so there is")
    print("            nothing to remember. A bare LLM, asked why you")
    print("            bought a dip you never traded, would invent a")
    print("            reason. The filter makes the invention impossible.")
print("  verdict: exact questions get exact filters, fuzzy questions")
print("           get ranked inside the slice, events get a window -")
print("           and every path still ends at the gate.")
