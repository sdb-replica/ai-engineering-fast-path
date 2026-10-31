# RAG 2: Build a Tiny RAG Over Your Own Docs

**Day 59 of 100** · Phase D — The modern LLM stack

**Tonight you'll be able to:** build the complete system from yesterday's diagram — overlapped chunking, the Day 58 index, answers assembled from your own notes with [doc#chunk] citations — and a refusal gate that says “not in my docs” instead of hallucinating.

## The one idea

Yesterday ended on a stub: retrieval worked, generation was a placeholder, and the corpus arrived pre-cut. Today you close the loop over a corpus that stands in for *your* docs folder — ten short notes: summaries of your own Days 5, 15, 27, 31, 32, 48, 53 and 58, a Conveyor DRS note, and a DynamoDB tiered-storage note. Everything from Day 58 survives. Exactly two boxes are new, plus one repair to an old one.

**The repair: chunking with overlap.** Real documents arrive whole, so a chunker cuts them — and cutting creates Day 58's first failure mode: an idea straddling a boundary arrives as two half-answers. The standard mitigation is a sliding window with a shared seam: 40-word windows advancing 30 words at a time, so neighbours overlap by 10 words and a sentence cut at one window's edge gets a second chance to be whole in the next. Part 1 prints the seam itself as proof. Honest limits: overlap is a mitigation, not a guarantee — a long sentence can still straddle — so the assembler in Part 3 only ever quotes *complete* sentences and treats fragments as non-evidence. The price is a bigger index: windows advance 30 words, not 40.

**New box 1: the gate.** Between retrieval and generation sit three lines of code: if the best cosine score is below `0.25`, refuse. Today's numbers show why it is load-bearing. The two answerable questions score `0.3329` and `0.4573`. The chocolate-cake question — nothing to do with any note — still manages `0.1978`, because TF-IDF credits it for shared filler words like *the*, *a*, *what*. That score looks real and is pure noise. Without a gate, the question proceeds to generation with RAG-architecture chunks as its "context", and something fluent and invented comes back. The gate converts a hallucination into an honest miss. It is also honestly fragile: a paraphrased question that shares no words with its chunk scores low and gets refused too — the known limit of word-matching retrieval, and exactly why production systems swap TF-IDF for Day 15's learned embeddings. But it errs toward silence, which is the right direction when the promise is "answers from *my* docs".

**New box 2: generation, with the label left on.** There is no LLM in this build, so the generation box is *extractive assembly*: lift the best complete sentences out of the top-2 chunks — scored by the rare words they share with the question — and print each one verbatim with its `[doc#chunk]` address attached. No paraphrase, no connective tissue, nothing that is not in your notes. Watch Q1's answer: exactly one sentence clears the quality bar, so the answer is one sentence long. A real model dropped into this box would write smoother prose from the *same evidence*; it would not change what gets retrieved, what gets refused, or what can be cited. That is the swap Day 58 promised — one box, everything else bolted down.

> **Analogy.** You would fire a storage system that answered a read for a missing key with a plausible invented value and a 200 OK. That is precisely what an ungated LLM does with an out-of-corpus question — it cannot return NOT FOUND, so it fabricates a row. The gate is the read path regaining the right to say *miss*. And the citations are provenance: `[day-048-kv-cache#0]` is the request ID of an answer — it names the exact stored object a sentence came from, so a human can verify the claim with one lookup instead of trusting the prose. You already run your systems this way: no provenance, no trust.

## See it

![RAG build diagram: Day 58's write path with an overlapping chunker, the 20-chunk index, and a read path where retrieval feeds a score gate — pass and the extractive GENERATE box answers with [doc#chunk] citations, fail and the system refuses](diagram.svg)

*Day 58's two-lane diagram, completed. Top lane unchanged except the chunker, which now overlaps its windows; the index holds 20 chunks, each with a citable address. Bottom lane: retrieval feeds a gate on the best score — pass, and the extractive assembler (today's honest stand-in for the LLM, drawn with the heavier border) answers with citations; fail, and the system refuses. The GENERATE box is the only box a real model would change.*

## Code it (~30 min)

Pure standard library again — no torch, no downloads, no APIs; run it in `~/ai-lab` or anywhere with `python3 code.py`. **Part 1** chunks the ten notes with overlap, proves the seam, and builds Day 58's TF-IDF index over the 20 chunks. **Part 2** replays the read path for three questions. **Part 3** is the payoff: the gate fires on all three — two answers assembled from verbatim sentences with citations, one refusal. Note that Q2's two answer sentences both come from chunk `#0` of the KV-cache note: retrieval found the right *chunk*, and assembly found the right *sentences* inside it.

```python
"""Day 59 - RAG 2: Build a Tiny RAG Over Your Own Docs.

Yesterday (Day 58) drew the architecture and ran retrieval
over 8 pre-cut chunks - with a stub where the answer goes.
Today the stub is filled in and the loop is closed: a
complete, honest little RAG system over a corpus that
stands in for YOUR docs folder.

  WRITE PATH (offline)
    10 short notes (summaries of your own earlier lessons
    plus two infra notes from your old teams) -> chunked
    with a sliding word window WITH OVERLAP -> TF-IDF
    embedded with Day 58's exact code -> indexed, every
    chunk carrying an address: [doc#chunk].

  READ PATH (online)
    question -> same embedder -> cosine top-k -> a GATE on
    the best score -> either an answer assembled from the
    retrieved chunks, or a refusal.

Two honest labels, up front:

  * The "generation" here is EXTRACTIVE: the answer is the
    best-matching sentences lifted verbatim out of the
    retrieved chunks, each printed with its [doc#chunk]
    citation. No LLM, no API. Swapping in a real model
    changes ONLY that box - it would paraphrase and connect
    the same sentences; retrieval, the gate, and the
    citation contract do not move.

  * The refusal is the point. A bare LLM answers every
    question, fluently, including the ones your docs say
    nothing about. This system checks retrieval's best
    score first: below the threshold it says "not in my
    docs" and generates NOTHING. Silence over fabrication.

  Part 1  Write path: chunk 10 docs with overlap, print
          per-doc chunk counts and prove the overlap seam.
  Part 2  Read path: 3 questions, cosine top-3 each.
  Part 3  The gate + the answer: two questions answered
          from the docs with citations; one question the
          docs cannot answer - watch the refusal fire.

Runs anywhere: Python 3.11+ standard library ONLY (math,
re). No torch, no APIs, no downloads, no RNG - every score
below is exact and reproducible digit for digit.

Expected output:

  Part 1 - WRITE PATH: chunk with overlap -> embed -> index
    day-005-loss-functions        68 words -> 2 chunks
    day-015-embeddings            66 words -> 2 chunks
    day-048-kv-cache              69 words -> 2 chunks
    conveyor-drs-note             60 words -> 2 chunks
    day-027-cnn-classifier        62 words -> 2 chunks
    dynamodb-tiered-storage-note  62 words -> 2 chunks
    day-058-rag-architecture      69 words -> 2 chunks
    day-032-tabular-mlp           65 words -> 2 chunks
    day-053-lora-note             61 words -> 2 chunks
    day-031-debugging-training    62 words -> 2 chunks
    total: 10 docs -> 20 chunks (window 40 words, overlap 10)
    overlap check: doc 'day-005-loss-functions' chunks 0 and 1 share their 10-word seam: True
    the seam itself: "absolute error grows linearly instead, so it resists outliers but"
    vocabulary = 387 terms; index shape: 20 chunk vectors x 387 dims
  Part 2 - READ PATH: embed query -> cosine-rank -> top-k
    Q1: "why does one outlier dominate the gradient in squared error"
      #1  day-005-loss-functions#0     cos 0.3329  The loss function tells a model how wrong it is. Mean squared error
      #2  day-031-debugging-training#1 cos 0.1736  a dead activation or a broken gradient path. A spike that never recovers
      #3  day-048-kv-cache#1           cos 0.1175  time they are computed. Later steps then only compute the new token and
    Q2: "why is the first token slow and what does the kv cache store"
      #1  day-048-kv-cache#0           cos 0.4573  Autoregressive generation recomputes attention over every previous token for each new token, which
      #2  day-048-kv-cache#1           cos 0.2614  time they are computed. Later steps then only compute the new token and
      #3  day-058-rag-architecture#1   cos 0.1685  the question is embedded with the same embedder, ranked by cosine similarity, and
    Q3: "what temperature do i bake a chocolate cake at"
      #1  day-058-rag-architecture#1   cos 0.1978  the question is embedded with the same embedder, ranked by cosine similarity, and
      #2  day-031-debugging-training#0 cos 0.1767  Loss curves are observability for training. A healthy curve falls steadily and flattens
      #3  day-031-debugging-training#1 cos 0.1407  a dead activation or a broken gradient path. A spike that never recovers
  Part 3 - THE GATE: answer with citations, or refuse
    Q1: "why does one outlier dominate the gradient in squared error"
      gate: best score 0.3329 >= 0.25 -> ANSWER
      ANSWER (extractive - verbatim sentences, citations attached):
        "Mean squared error squares each residual, so one big outlier dominates the gradient and drags the fit toward it." [day-005-loss-functions#0]
    Q2: "why is the first token slow and what does the kv cache store"
      gate: best score 0.4573 >= 0.25 -> ANSWER
      ANSWER (extractive - verbatim sentences, citations attached):
        "Autoregressive generation recomputes attention over every previous token for each new token, which makes the first token slow." [day-048-kv-cache#0]
        "The KV cache stores each token's key and value vectors the first time they are computed." [day-048-kv-cache#0]
    Q3: "what temperature do i bake a chocolate cake at"
      gate: best score 0.1978 < 0.25 -> REFUSE
      ANSWER: not in my docs - no chunk clears the gate, so nothing is
              generated. A bare LLM would have invented a temperature here.
    verdict: same pipeline as Day 58 - the only new boxes are the
             gate and the assembler. Swap the assembler for a real LLM
             and every other line of this system stays exactly as is.
"""
import math
import re
from collections import Counter

# The corpus: 10 short notes standing in for a docs folder.
# Each is one small markdown-style note, a few sentences.
DOCS = [
    ("day-005-loss-functions",
     "The loss function tells a model how wrong it is. Mean squared error "
     "squares each residual, so one big outlier dominates the gradient and "
     "drags the fit toward it. Mean absolute error grows linearly instead, "
     "so it resists outliers but is less smooth to optimize. A loss must be "
     "differentiable, because the training loop needs a gradient at every "
     "step to know which way to nudge the weights."),
    ("day-015-embeddings",
     "An embedding turns a category into a vector of learned numbers. "
     "One-hot vectors say every pair of categories is equally different, "
     "which is useless for meaning. A learned embedding table places "
     "similar meanings near each other, so cosine similarity reads bull "
     "and rally as close and bear as far. Only the rows looked up in a "
     "batch receive gradients; the rest of the table stays untouched."),
    ("day-048-kv-cache",
     "Autoregressive generation recomputes attention over every previous "
     "token for each new token, which makes the first token slow. The KV "
     "cache stores each token's key and value vectors the first time they "
     "are computed. Later steps then only compute the new token and read "
     "the cache. Memory grows linearly with context length, trading RAM "
     "for speed. This is why long conversations get cheaper per token but "
     "heavier in memory."),
    ("conveyor-drs-note",
     "The Data Reliability Service scans stored objects for corruption "
     "and schedules repairs before redundancy runs out. It compares "
     "checksums against replicas, flags silent bit rot, and prioritizes "
     "objects whose remaining copies are fewest. Repair traffic is rate "
     "limited so background healing never starves foreground reads. A "
     "repair reads a healthy replica, rewrites the bad shard, and "
     "verifies the new checksum."),
    ("day-027-cnn-classifier",
     "The small CNN stacks two convolution blocks before a linear head. "
     "Each block convolves, applies ReLU, then max-pools, halving the "
     "feature map while channels grow. Pooling keeps the strongest "
     "activation in each window, a rollup that throws away position but "
     "keeps presence. With under two thousand parameters it beat the "
     "plain MLP on handwritten digits, reaching about ninety eight "
     "percent validation accuracy."),
    ("dynamodb-tiered-storage-note",
     "Tiered storage moves cold partitions to cheaper media automatically "
     "when access patterns cool off. The hot tier keeps recent, frequently "
     "read data on fast SSDs close to compute. A background process tracks "
     "access counts per partition and demotes the coldest ones first. "
     "Reads from the cold tier pay extra latency, so the system warms a "
     "partition back to hot when traffic returns."),
    ("day-058-rag-architecture",
     "Retrieval augmented generation keeps knowledge in an external index "
     "instead of the model weights. Offline, documents are chunked, "
     "embedded, and indexed with each vector stored beside its chunk "
     "text. Online, the question is embedded with the same embedder, "
     "ranked by cosine similarity, and the top chunks are stuffed into "
     "the prompt as context. The model answers from what it reads, so a "
     "missed retrieval becomes a confident wrong answer."),
    ("day-032-tabular-mlp",
     "The tabular MLP predicts next day direction from five features "
     "built out of raw price history. Feature engineering was the real "
     "work: returns, momentum windows, realized volatility, and the "
     "fraction of up days. A chronological split kept the future out of "
     "training, because shuffling time series leaks tomorrow into "
     "yesterday. The small network generalized; the wide one memorized "
     "the training rows and failed on validation."),
    ("day-053-lora-note",
     "LoRA fine tunes a model by training only two small low rank "
     "matrices injected beside each frozen weight matrix. The base model "
     "never changes, so one copy serves many adapters. Merging the "
     "adapter back into the weights costs nothing at inference time. "
     "Because so few parameters train, LoRA runs on a single modest GPU "
     "where full fine tuning would not fit."),
    ("day-031-debugging-training",
     "Loss curves are observability for training. A healthy curve falls "
     "steadily and flattens at a floor set by label noise. A flatline "
     "means almost no signal reaches the weights, often a dead activation "
     "or a broken gradient path. A spike that never recovers points at "
     "one corrupted batch. The train and validation curves together "
     "separate underfitting from overfitting before any metric does."),
]

CHUNK_WORDS = 40     # window size, in words
OVERLAP = 10         # words shared between neighbouring chunks
K = 3                # top-k retrieval
THRESHOLD = 0.25     # the refusal gate: best cosine must clear this


def tokenize(text):
    """Lowercase word tokens; punctuation is dropped. (Day 58, verbatim.)"""
    return re.findall(r"[a-z0-9]+", text.lower())


def chunk_doc(text, size=CHUNK_WORDS, overlap=OVERLAP):
    """Sliding word window. Neighbouring chunks share `overlap`
    words, so an idea sitting on a boundary survives whole in at
    least one chunk - Day 58's 'bad chunks' failure mode, fixed."""
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if start > 0 and len(piece) <= overlap:
            break                    # tail adds nothing new - skip it
        chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


# ---------- Part 1: WRITE PATH - chunk (with overlap) -> embed -> index ----
print("Part 1 - WRITE PATH: chunk with overlap -> embed -> index")
entries = []               # (doc name, chunk no, chunk text) - the store rows
for name, text in DOCS:
    pieces = chunk_doc(text)
    for ci, piece in enumerate(pieces):
        entries.append((name, ci, piece))
    print(f"  {name:<28} {len(text.split()):>3} words -> {len(pieces)} chunks")
print(f"  total: {len(DOCS)} docs -> {len(entries)} chunks "
      f"(window {CHUNK_WORDS} words, overlap {OVERLAP})")

# Prove the seam: chunk 1 of doc 0 must OPEN with the 10 words
# chunk 0 CLOSED with - that shared strip is the overlap.
c0 = entries[0][2].split()
c1 = entries[1][2].split()
print(f"  overlap check: doc '{entries[0][0]}' chunks 0 and 1 share "
      f"their {OVERLAP}-word seam: {c0[-OVERLAP:] == c1[:OVERLAP]}")
print(f"  the seam itself: \"{' '.join(c0[-OVERLAP:])}\"")

# Day 58's embedder, unchanged - now over chunks instead of documents.
chunk_tokens = [tokenize(text) for _, _, text in entries]
vocab = sorted({t for toks in chunk_tokens for t in toks})
V = {w: i for i, w in enumerate(vocab)}
N = len(entries)
df = Counter()
for toks in chunk_tokens:
    df.update(set(toks))
idf = {w: math.log((1 + N) / (1 + df[w])) + 1.0 for w in vocab}


def embed(tokens):
    """TF-IDF vector: count of each word, weighted by its idf."""
    counts = Counter(tokens)
    vec = [0.0] * len(vocab)
    for w, c in counts.items():
        if w in V:                       # unknown words are ignored
            vec[V[w]] = c * idf[w]
    return vec


index = [embed(toks) for toks in chunk_tokens]   # the "vector store"
print(f"  vocabulary = {len(vocab)} terms; index shape: "
      f"{len(index)} chunk vectors x {len(vocab)} dims")

# ---------- Part 2: READ PATH - embed query -> cosine-rank -> top-k --------
print("Part 2 - READ PATH: embed query -> cosine-rank -> top-k")
QUERIES = [
    "why does one outlier dominate the gradient in squared error",
    "why is the first token slow and what does the kv cache store",
    "what temperature do i bake a chocolate cake at",
]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def retrieve(query, k=K):
    qv = embed(tokenize(query))         # SAME embedder as the write path
    scored = [(cosine(qv, row), i) for i, row in enumerate(index)]
    scored.sort(key=lambda p: (-p[0], p[1]))
    return scored[:k]


def chunk_id(i):
    name, ci, _ = entries[i]
    return f"{name}#{ci}"


results = {}
for qi, q in enumerate(QUERIES, start=1):
    hits = retrieve(q)
    results[q] = hits
    print(f"  Q{qi}: \"{q}\"")
    for rank, (score, i) in enumerate(hits, start=1):
        preview = " ".join(entries[i][2].split()[:13])
        print(f"    #{rank}  {chunk_id(i):<28} cos {score:.4f}  {preview}")

# ---------- Part 3: THE GATE + EXTRACTIVE ANSWER (or refusal) --------------
print("Part 3 - THE GATE: answer with citations, or refuse")


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


def extract_answer(query, hits, n=2):
    """The generation box, honestly stubbed: lift the sentences
    from the top-2 chunks that best match the query (scored by
    summed idf of shared words), verbatim, citations attached.
    Two quality bars: a sentence must be COMPLETE (chunk edges
    leave fragments - not quotable evidence) and must score at
    least 40% of the best sentence, so a weak second chunk
    cannot pad the answer with off-topic lines."""
    qtoks = set(tokenize(query))
    cands = []
    for _, i in hits[:2]:
        for si, sent in enumerate(sentences(entries[i][2])):
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
    return [(sent, chunk_id(i)) for _, i, _, sent in picked]


for qi, q in enumerate(QUERIES, start=1):
    hits = results[q]
    best = hits[0][0]
    print(f"  Q{qi}: \"{q}\"")
    if best < THRESHOLD:
        print(f"    gate: best score {best:.4f} < {THRESHOLD:.2f} -> REFUSE")
        print("    ANSWER: not in my docs - no chunk clears the gate, "
              "so nothing is")
        print("            generated. A bare LLM would have invented a "
              "temperature here.")
    else:
        print(f"    gate: best score {best:.4f} >= {THRESHOLD:.2f} -> ANSWER")
        print("    ANSWER (extractive - verbatim sentences, citations "
              "attached):")
        for sent, cid in extract_answer(q, hits):
            print(f"      \"{sent}\" [{cid}]")
print("  verdict: same pipeline as Day 58 - the only new boxes are the")
print("           gate and the assembler. Swap the assembler for a real LLM")
print("           and every other line of this system stays exactly as is.")
```

**Expected output:**

```
Part 1 - WRITE PATH: chunk with overlap -> embed -> index
  day-005-loss-functions        68 words -> 2 chunks
  day-015-embeddings            66 words -> 2 chunks
  day-048-kv-cache              69 words -> 2 chunks
  conveyor-drs-note             60 words -> 2 chunks
  day-027-cnn-classifier        62 words -> 2 chunks
  dynamodb-tiered-storage-note  62 words -> 2 chunks
  day-058-rag-architecture      69 words -> 2 chunks
  day-032-tabular-mlp           65 words -> 2 chunks
  day-053-lora-note             61 words -> 2 chunks
  day-031-debugging-training    62 words -> 2 chunks
  total: 10 docs -> 20 chunks (window 40 words, overlap 10)
  overlap check: doc 'day-005-loss-functions' chunks 0 and 1 share their 10-word seam: True
  the seam itself: "absolute error grows linearly instead, so it resists outliers but"
  vocabulary = 387 terms; index shape: 20 chunk vectors x 387 dims
Part 2 - READ PATH: embed query -> cosine-rank -> top-k
  Q1: "why does one outlier dominate the gradient in squared error"
    #1  day-005-loss-functions#0     cos 0.3329  The loss function tells a model how wrong it is. Mean squared error
    #2  day-031-debugging-training#1 cos 0.1736  a dead activation or a broken gradient path. A spike that never recovers
    #3  day-048-kv-cache#1           cos 0.1175  time they are computed. Later steps then only compute the new token and
  Q2: "why is the first token slow and what does the kv cache store"
    #1  day-048-kv-cache#0           cos 0.4573  Autoregressive generation recomputes attention over every previous token for each new token, which
    #2  day-048-kv-cache#1           cos 0.2614  time they are computed. Later steps then only compute the new token and
    #3  day-058-rag-architecture#1   cos 0.1685  the question is embedded with the same embedder, ranked by cosine similarity, and
  Q3: "what temperature do i bake a chocolate cake at"
    #1  day-058-rag-architecture#1   cos 0.1978  the question is embedded with the same embedder, ranked by cosine similarity, and
    #2  day-031-debugging-training#0 cos 0.1767  Loss curves are observability for training. A healthy curve falls steadily and flattens
    #3  day-031-debugging-training#1 cos 0.1407  a dead activation or a broken gradient path. A spike that never recovers
Part 3 - THE GATE: answer with citations, or refuse
  Q1: "why does one outlier dominate the gradient in squared error"
    gate: best score 0.3329 >= 0.25 -> ANSWER
    ANSWER (extractive - verbatim sentences, citations attached):
      "Mean squared error squares each residual, so one big outlier dominates the gradient and drags the fit toward it." [day-005-loss-functions#0]
  Q2: "why is the first token slow and what does the kv cache store"
    gate: best score 0.4573 >= 0.25 -> ANSWER
    ANSWER (extractive - verbatim sentences, citations attached):
      "Autoregressive generation recomputes attention over every previous token for each new token, which makes the first token slow." [day-048-kv-cache#0]
      "The KV cache stores each token's key and value vectors the first time they are computed." [day-048-kv-cache#0]
  Q3: "what temperature do i bake a chocolate cake at"
    gate: best score 0.1978 < 0.25 -> REFUSE
    ANSWER: not in my docs - no chunk clears the gate, so nothing is
            generated. A bare LLM would have invented a temperature here.
  verdict: same pipeline as Day 58 - the only new boxes are the
           gate and the assembler. Swap the assembler for a real LLM
           and every other line of this system stays exactly as is.
```

## Today's win

Done when your run prints `10 docs -> 20 chunks` with the overlap seam `True`, Q1 answered by the single outlier sentence cited `[day-005-loss-functions#0]`, Q2 answered by two sentences both cited `[day-048-kv-cache#0]`, and Q3 refused at `0.1978 < 0.25` with nothing generated. Keeper test, no notes: say why overlap exists, why the gate sits *before* generation rather than after, and which single box a real LLM would replace.

## Short on time? 20-minute version

Read "The one idea", study the diagram until you can point at the two new boxes, then run the code and read only Part 3. Now mentally lower `THRESHOLD` to `0.15`: Q3 suddenly "answers" a baking question out of RAG-architecture chunks. That shiver is the entire lesson — you have felt exactly what the gate is for.

## Tomorrow's teaser

Tomorrow the pipeline grows hands: the system stops only answering questions and starts calling code you wrote — one tool at a time, with the same discipline about what happens when the answer is "no".

---

*Streak: 59 day(s) · Never miss twice.*
