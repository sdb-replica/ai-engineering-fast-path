"""Day 58 - RAG 1: Retrieval-Augmented Generation, the Architecture.

Yesterday's problem (Day 49): context is expensive, so a model
can only read a little at a time - and its weights froze at
training time anyway. RAG's answer is an architecture, not a
bigger model:

    Don't memorize the library in the weights. Keep knowledge
    in an external index and FETCH the relevant chunks at
    query time.

Two paths, exactly like a storage system:

  WRITE PATH (offline, once per document change)
    documents -> chunk -> embed each chunk -> index the
    vectors next to the chunk text. This is a build job:
    no user is waiting on it.

  READ PATH (online, once per question)
    question -> embed it the SAME way -> cosine-rank the
    index -> take the top-k chunks -> stuff them into the
    prompt as context -> the model generates an answer that
    can only be as good as what retrieval handed it.

The model is a very smart but amnesiac compute engine. The
index is its external memory - a read-through cache / a
secondary index in front of it. Nothing here retrains
anything: new knowledge = re-index, not re-train.

Today is retrieval ONLY - no generation (that assembly is
tomorrow's build). The "embeddings" are transparent TF-IDF
vectors built with the stdlib, standing in for Day 15's
learned embeddings: rare words weigh more (idf), and a chunk
is a bag of weighted word-counts. Real systems swap this
vectorizer for a neural embedder and an approximate index;
the shape of the pipeline does not change.

  Part 1  Write path: 8 hand-written chunks about object
          storage, chunked, tokenized, TF-IDF embedded, and
          indexed - print the index shape and one vector.
  Part 2  Read path: two questions, embedded the same way,
          cosine-ranked against the index - print the top 3
          chunks with scores. Watch WHICH chunks win and
          which relevant-looking one misses.
  Part 3  Stuff the context: assemble the exact prompt block
          that would be handed to a model for question 1 -
          context chunks + question - and a stub where the
          generated answer would go.

Runs anywhere: Python 3.11+ standard library ONLY (math, re).
No torch, no APIs, no downloads, no RNG - every score below
is exact and reproducible digit for digit.

Expected output:

  Part 1 - WRITE PATH (offline): chunk -> embed -> index
    8 chunks indexed; vocabulary = 108 terms
    index shape: 8 chunk vectors x 108 dims (one row per chunk)
    chunk 0 vector (first 6 of 108 dims): [0.0000, 0.0000, 2.5041, 1.5878, 2.5041, 0.0000]
    (dims are vocabulary words in sorted order; most entries are 0 - sparse)
  Part 2 - READ PATH (online): embed query -> cosine-rank -> top-k
    Q1: "how does erasure coding protect data when disks fail"
      #1  chunk 0  cos 0.4295  Erasure coding splits an object into data shards and parity shards, so the
      #2  chunk 6  cos 0.0982  Compaction merges small objects into large segments and rewrites live data, reclaiming space
      #3  chunk 5  cos 0.0856  A cold object lives on dense hard disks; the first read pays a
    Q2: "why is the first read of a cold object slow"
      #1  chunk 5  cos 0.4574  A cold object lives on dense hard disks; the first read pays a
      #2  chunk 2  cos 0.3167  A write is acknowledged only once a quorum of nodes has durably stored
      #3  chunk 7  cos 0.2570  The metadata index maps each object name to its shard locations, so a
  Part 3 - STUFF THE CONTEXT: the prompt the model would receive (Q1, k=3)
    ----- PROMPT -----
    Answer the question using only the context below.
    Context:
    [chunk 0] Erasure coding splits an object into data shards and parity shards, so the object survives the loss of any few disks or nodes.
    [chunk 6] Compaction merges small objects into large segments and rewrites live data, reclaiming space from deleted objects.
    [chunk 5] A cold object lives on dense hard disks; the first read pays a disk seek, so cold reads are far slower than cached reads.
    Question: how does erasure coding protect data when disks fail
    ----- END PROMPT -----
    [stub] generation happens here tomorrow - today the win is that the
           model would be reading chunk 0, the one chunk that answers Q1.
    verdict: the model never memorized any of this - it was fetched, at query time
"""
import math
import re
from collections import Counter

# The "corpus": 8 short chunks, as if split from one design doc
# about an object storage system. In production a chunker cuts
# real documents; here each chunk is already one idea.
CHUNKS = [
    "Erasure coding splits an object into data shards and parity shards, "
    "so the object survives the loss of any few disks or nodes.",
    "Replication keeps three full copies of every object on different "
    "nodes; reads are fast, but storage costs three times the raw size.",
    "A write is acknowledged only once a quorum of nodes has durably "
    "stored the shards; that quorum wait is what makes writes slower than reads.",
    "Checksums are computed on every read and write; a mismatch marks "
    "the shard corrupt and triggers repair from the surviving shards.",
    "Hot objects are cached in memory on the read path, so a cache hit "
    "skips disk and keeps tail latency low.",
    "A cold object lives on dense hard disks; the first read pays a disk "
    "seek, so cold reads are far slower than cached reads.",
    "Compaction merges small objects into large segments and rewrites "
    "live data, reclaiming space from deleted objects.",
    "The metadata index maps each object name to its shard locations, "
    "so a read first looks up where the pieces live.",
]

QUERIES = [
    "how does erasure coding protect data when disks fail",
    "why is the first read of a cold object slow",
]
K = 3


def tokenize(text):
    """Lowercase word tokens; punctuation is dropped."""
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------- Part 1: WRITE PATH - chunk -> embed -> index ----------
print("Part 1 - WRITE PATH (offline): chunk -> embed -> index")
chunk_tokens = [tokenize(c) for c in CHUNKS]
vocab = sorted({t for toks in chunk_tokens for t in toks})
V = {w: i for i, w in enumerate(vocab)}

# idf: a word in few chunks is informative; a word everywhere is not.
N = len(CHUNKS)
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
print(f"  {N} chunks indexed; vocabulary = {len(vocab)} terms")
print(f"  index shape: {len(index)} chunk vectors x {len(vocab)} dims "
      "(one row per chunk)")
first6 = ", ".join(f"{v:.4f}" for v in index[0][:6])
print(f"  chunk 0 vector (first 6 of {len(vocab)} dims): [{first6}]")
print("  (dims are vocabulary words in sorted order; most entries "
      "are 0 - sparse)")

# ---------- Part 2: READ PATH - embed query -> rank -> top-k ----------
print("Part 2 - READ PATH (online): embed query -> cosine-rank -> top-k")


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def retrieve(query, k=K):
    qv = embed(tokenize(query))         # SAME embedder as the write path
    scored = [(cosine(qv, row), i) for i, row in enumerate(index)]
    scored.sort(key=lambda p: (-p[0], p[1]))
    return scored[:k]


results = {}
for qi, q in enumerate(QUERIES, start=1):
    hits = retrieve(q)
    results[q] = hits
    print(f"  Q{qi}: \"{q}\"")
    for rank, (score, ci) in enumerate(hits, start=1):
        preview = " ".join(CHUNKS[ci].split()[:13])
        print(f"    #{rank}  chunk {ci}  cos {score:.4f}  {preview}")

# ---------- Part 3: STUFF THE CONTEXT - build the prompt ----------
print("Part 3 - STUFF THE CONTEXT: the prompt the model would "
      "receive (Q1, k=3)")
q1 = QUERIES[0]
lines = ["----- PROMPT -----",
         "Answer the question using only the context below.",
         "Context:"]
for _, ci in results[q1]:
    lines.append(f"[chunk {ci}] {CHUNKS[ci]}")
lines.append(f"Question: {q1}")
lines.append("----- END PROMPT -----")
for line in lines:
    print(f"  {line}")
print("  [stub] generation happens here tomorrow - today the win is that the")
print("         model would be reading chunk 0, the one chunk that answers Q1.")
print("  verdict: the model never memorized any of this - it was "
      "fetched, at query time")
