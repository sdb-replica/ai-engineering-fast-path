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
