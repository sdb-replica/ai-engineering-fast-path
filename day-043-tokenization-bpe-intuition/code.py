"""Day 43 - Tokenization: BPE Intuition.

Day 42's tiny GPT read text one CHARACTER at a time.
Real models don't. Before the embedding lookup (Day 15)
ever runs, a tokenizer chops text into TOKENS - chunks
that are sometimes a whole word, sometimes a syllable,
sometimes a single letter - and swaps each chunk for an
id. The model only ever sees the ids.

Why tokens? One dial with two bad ends:
  characters - vocab is tiny, but sequences balloon:
               every chunk of text costs hundreds of
               steps, and attention pays per step.
  words      - sequences shrink, but the vocab explodes
               (every word, every form), and a word the
               vocab never saw is UNREPRESENTABLE - the
               dreaded UNK, a dead end.
Tokens are the learned middle: BYTE-PAIR ENCODING (BPE)
learns the chunks from data. Start from characters, then
repeat one move - count every adjacent pair, merge the
most frequent one into a new symbol - until the vocab
reaches its budget. Frequent strings collapse into one
token; rare strings stay spelled out in pieces. So a
token is NOT a word: a common word costs 1 token, a rare
word costs 4, and a word training never saw still encodes
as known pieces - no UNK, ever.

Tonight's corpus is four trading-desk words with made-up
frequencies, small enough to run the whole loop by hand:

  Part 1  The starting point: characters + an end-of-word
          marker </w>. The round-1 pair table goes up so
          you can call the first merge before the loop
          does (three pairs tie at 9 - the tie-break rule
          is part of the lesson).
  Part 2  The merge loop, 8 rounds. Each round prints the
          winning pair, the new symbol, the vocab size and
          the corpus token total - watch the trade happen:
          vocab 11 -> 19 buys corpus 95 -> 36 tokens.
  Part 3  Encode with the learned merges: newest = 1
          token, low = 2, widest = 4, and "lowest" - a
          word training NEVER saw - still encodes as 2
          known pieces. Plus the ids: tokens are just row
          numbers in Day 15's embedding table.

Runs anywhere Python 3 runs: standard library only - a
tokenizer is string surgery, no torch, no GPU, no paid
APIs, no downloads. In the ~/ai-lab venv it works exactly
as-is. No RNG: every printed number is exact.

Expected output:

  Part 1 - the starting point: characters, no chunks yet
    low     x5  as symbols: [l, o, w, </w>]
    lower   x2  as symbols: [l, o, w, e, r, </w>]
    newest  x6  as symbols: [n, e, w, e, s, t, </w>]
    widest  x3  as symbols: [w, i, d, e, s, t, </w>]
    vocab: 11 symbols - d, e, i, l, n, o, r, s, t, w, </w>
    the whole corpus at character level: 95 tokens
    round-1 pair counts, top of the table:
      e + s     9
      s + t     9
      t + </w>  9
      w + e     8
      l + o     7
      o + w     7
    three pairs tie at 9 - first met in scan order wins: e + s (9)
  Part 2 - the merge loop: 8 rounds of one move
    round 1: e + s -> es  (9 occurrences)  vocab 12  corpus 95 -> 86 tokens
    round 2: es + t -> est  (9 occurrences)  vocab 13  corpus 86 -> 77 tokens
    round 3: est + </w> -> est</w>  (9 occurrences)  vocab 14  corpus 77 -> 68 tokens
    round 4: l + o -> lo  (7 occurrences)  vocab 15  corpus 68 -> 61 tokens
    round 5: lo + w -> low  (7 occurrences)  vocab 16  corpus 61 -> 54 tokens
    round 6: n + e -> ne  (6 occurrences)  vocab 17  corpus 54 -> 48 tokens
    round 7: ne + w -> new  (6 occurrences)  vocab 18  corpus 48 -> 42 tokens
    round 8: new + est</w> -> newest</w>  (6 occurrences)  vocab 19  corpus 42 -> 36 tokens
    final vocab: 19 symbols (ids 0-18, one embedding row each)
    the trade: +8 vocab entries bought a corpus of 95 -> 36 tokens (2.6x shorter)
  Part 3 - encode with the learned merges: tokens != words
    newest  -> [newest</w>]  ids [18]  (1 token)  the corpus favourite
    low     -> [low, </w>]  ids [15, 10]  (2 tokens)  common, but never merged with </w>
    widest  -> [w, i, d, est</w>]  ids [9, 2, 0, 13]  (4 tokens)  rarer - stays in pieces
    lowest  -> [low, est</w>]  ids [15, 13]  (2 tokens)  NEVER in training - still no UNK
    decode check: join the pieces, </w> becomes a space -
      [low, est</w>] decodes back to 'lowest'
    verdict: the vocab is learned, not designed - frequency
      decides what earns a single token, and every string
      stays representable. Tokens are the unit everything
      downstream counts: context windows and API bills
      are both denominated in them.
"""

CORPUS = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
END = "</w>"          # end-of-word marker: one extra symbol
NUM_MERGES = 8


def pair_counts(words):
    """Count adjacent pairs across the corpus, weighted by
    word frequency. The dict keeps first-encounter order
    (corpus order, left to right) - that order is the
    tie-break rule when two pairs share the top count."""
    counts = {}
    for symbols, freq in words.items():
        for a, b in zip(symbols, symbols[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + freq
    return counts


def merge_pair(symbols, pair, new):
    """One left-to-right pass replacing every non-overlapping
    occurrence of pair with the new merged symbol."""
    merged = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            merged.append(new)
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return merged


def show(symbols):
    return "[" + ", ".join(symbols) + "]"


# ---------- Part 1: characters only, and the first call ----------
print("Part 1 - the starting point: characters, no chunks yet")
words = {}
for word, freq in CORPUS.items():
    words[tuple(list(word) + [END])] = freq
    print(f"  {word:<7} x{freq}  as symbols: {show(list(word) + [END])}")
vocab = sorted({s for symbols in words for s in symbols if s != END})
vocab.append(END)
total = sum(len(symbols) * freq for symbols, freq in words.items())
print(f"  vocab: {len(vocab)} symbols - {', '.join(vocab)}")
print(f"  the whole corpus at character level: {total} tokens")
counts = pair_counts(words)
top = sorted(counts.items(), key=lambda kv: -kv[1])[:6]
print("  round-1 pair counts, top of the table:")
for (a, b), n in top:
    print(f"    {a} + {b:<5} {n}")
best, best_n = max(counts.items(), key=lambda kv: kv[1])
print(f"  three pairs tie at 9 - first met in scan order wins: "
      f"{best[0]} + {best[1]} ({best_n})")

# ---------- Part 2: the merge loop ----------
print(f"Part 2 - the merge loop: {NUM_MERGES} rounds of one move")
merges = []
for rnd in range(1, NUM_MERGES + 1):
    counts = pair_counts(words)
    (a, b), n = max(counts.items(), key=lambda kv: kv[1])
    new = a + b
    merges.append(((a, b), new))
    words = {tuple(merge_pair(list(symbols), (a, b), new)): freq
             for symbols, freq in words.items()}
    vocab.append(new)
    new_total = sum(len(symbols) * freq
                    for symbols, freq in words.items())
    print(f"  round {rnd}: {a} + {b} -> {new}  ({n} occurrences)"
          f"  vocab {len(vocab)}  corpus {total} -> {new_total} tokens")
    total = new_total
print(f"  final vocab: {len(vocab)} symbols "
      f"(ids 0-{len(vocab) - 1}, one embedding row each)")
print(f"  the trade: +{len(vocab) - 11} vocab entries bought a corpus "
      f"of 95 -> {total} tokens (2.6x shorter)")

# ---------- Part 3: encode - tokens are not words ----------
print("Part 3 - encode with the learned merges: tokens != words")
ids = {symbol: i for i, symbol in enumerate(vocab)}


def encode(word):
    symbols = list(word) + [END]
    for pair, new in merges:
        symbols = merge_pair(symbols, pair, new)
    return symbols


for word, note in (("newest", "the corpus favourite"),
                   ("low", "common, but never merged with </w>"),
                   ("widest", "rarer - stays in pieces"),
                   ("lowest", "NEVER in training - still no UNK")):
    tokens = encode(word)
    token_ids = [ids[t] for t in tokens]
    print(f"  {word:<7} -> {show(tokens)}  ids {token_ids}"
          f"  ({len(tokens)} token{'s' if len(tokens) > 1 else ''})"
          f"  {note}")
print("  decode check: join the pieces, </w> becomes a space -")
joined = "".join(encode("lowest")).replace(END, " ").strip()
print(f"    [low, est</w>] decodes back to '{joined}'")
print("  verdict: the vocab is learned, not designed - frequency")
print("    decides what earns a single token, and every string")
print("    stays representable. Tokens are the unit everything")
print("    downstream counts: context windows and API bills")
print("    are both denominated in them.")
