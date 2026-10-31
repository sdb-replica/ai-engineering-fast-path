"""Day 45 - Encoder vs Decoder vs Encoder-Decoder: Same Block, Three Wirings.

Days 37-42 you built one transformer block (Day 40) and ran
it decoder-style, behind a causal mask. Tonight that same
block gets sorted into the only three wiring patterns the
field ships. The difference is never the block - it is who
is allowed to READ whom inside attention:

  Encoder           self-attention, NO mask. Every token
                    reads every token, past and future.
                    Output: one informed vector per input
                    token. Built to UNDERSTAND a finished
                    input - BERT, embeddings, search.
  Decoder           self-attention WITH the causal mask
                    (Days 38/42). Each token reads only
                    itself and the past, so the stack can
                    GENERATE left to right - GPT, chat,
                    code completion.
  Encoder-decoder   two stacks. The encoder reads the
                    source fully (no mask - it all exists);
                    the decoder generates causally AND gets
                    one extra sublayer per block:
                    cross-attention, where the queries come
                    from decoder states and the keys/values
                    come from the encoder's output. Built to
                    turn one finished sequence into a NEW
                    sequence - translation, Whisper, T5.

  Part 1  One tiny attention() function, identity
          projections, four hand-set tokens. No mask: the
          bidirectional weight matrix. Token 1 already
          leans hardest on token 4 - exactly right for
          understanding, cheating for generation.
  Part 2  The SAME function, one argument different: the
          causal mask. Every future weight lands on
          exactly 0.0000 and every row still sums to 1.
          That mask is the entire difference between an
          encoder layer and a decoder layer.
  Part 3  Cross-attention at shape level. Queries from
          two decoder states, keys/values from three
          encoder states: the weight matrix is RECTANGULAR
          (2 x 3), its rows still softmax to 1, and the
          output carries the DECODER's length - queries
          ask, the memory answers.

Runs on the ~/ai-lab venv: torch only (free), CPU on
purpose (tiny data), no CUDA, no paid APIs, no downloads.
Every tensor is set by hand, so every number is exact -
no RNG anywhere in this lesson.

Expected output:

  Part 1 - ENCODER wiring: no mask, every token reads every token
    token 1 reads: 0.2212, 0.1091, 0.2212, 0.4486   row sum 1.0000
    token 2 reads: 0.1651, 0.3349, 0.3349, 0.1651   row sum 1.0000
    token 3 reads: 0.1651, 0.1651, 0.3349, 0.3349   row sum 1.0000
    token 4 reads: 0.1573, 0.0382, 0.1573, 0.6471   row sum 1.0000
    token 1 puts 0.4486 on token 4 - it reads the FUTURE, on purpose
    verdict: built to UNDERSTAND a finished input (BERT-style)
  Part 2 - DECODER wiring: same call + the causal mask (Days 38/42)
    token 1 reads: 1.0000, 0.0000, 0.0000, 0.0000   row sum 1.0000
    token 2 reads: 0.3302, 0.6698, 0.0000, 0.0000   row sum 1.0000
    token 3 reads: 0.2483, 0.2483, 0.5035, 0.0000   row sum 1.0000
    token 4 reads: 0.1573, 0.0382, 0.1573, 0.6471   row sum 1.0000
    biggest future weight anywhere: 0.0000 - the future is sealed
    token 4's row is unchanged: it has no future to hide
    verdict: same block + one mask, now it can GENERATE (GPT-style)
  Part 3 - CROSS-attention: Q from the decoder, K/V from the encoder
    shapes: Q (2, 2) from decoder, K (3, 2) and V (3, 2) from encoder
    weight matrix shape (2, 3) - RECTANGULAR: 2 asking, 3 answering
    decoder step 1 reads source: 0.4011, 0.1978, 0.4011   row sum 1.0000
    decoder step 2 reads source: 0.2483, 0.2483, 0.5035   row sum 1.0000
    output row 1: 0.8022, 0.5989   output row 2: 0.7517, 0.7517
    output shape (2, 2): length follows the QUERIES (decoder),
      content flows from the MEMORY (encoder output)
    verdict: TRANSFORM one finished sequence into a new one (T5/Whisper)
"""
import math

import torch


def attention(Q, K, V, mask=None):
    """Day 38's scaled dot-product attention, unchanged.

    Q (Tq, d), K (Tk, d), V (Tk, dv) -> (out (Tq, dv), weights (Tq, Tk)).
    mask: 1 = may read, 0 = blocked (score set to -inf pre-softmax).
    Self-attention is Q, K, V from the SAME sequence; cross-attention
    is Q from one sequence, K/V from another. One function, both.
    """
    d = Q.shape[-1]
    scores = Q @ K.T / math.sqrt(d)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ V, weights


row = lambda t: ", ".join(f"{v:.4f}" for v in t.tolist())

# Four hand-set tokens, d = 2. Identity projections (Wq = Wk = Wv = I)
# keep Day 38's arithmetic on the table - the mask is the only
# moving part today.
X = torch.tensor([[1.0, 0.0],     # the
                  [0.0, 1.0],     # cat
                  [1.0, 1.0],     # sat
                  [2.0, 0.0]])    # down

# ---------- Part 1: encoder wiring - no mask ----------
print("Part 1 - ENCODER wiring: no mask, every token reads every token")
_, w_enc = attention(X, X, X)
for i, w in enumerate(w_enc):
    print(f"  token {i + 1} reads: {row(w)}   row sum {w.sum().item():.4f}")
print(f"  token 1 puts {w_enc[0, 3].item():.4f} on token 4"
      " - it reads the FUTURE, on purpose")
print("  verdict: built to UNDERSTAND a finished input (BERT-style)")

# ---------- Part 2: decoder wiring - same call + causal mask ----------
print("Part 2 - DECODER wiring: same call + the causal mask (Days 38/42)")
causal = torch.tril(torch.ones(4, 4))          # 1 on/below diagonal
_, w_dec = attention(X, X, X, mask=causal)
for i, w in enumerate(w_dec):
    print(f"  token {i + 1} reads: {row(w)}   row sum {w.sum().item():.4f}")
future = w_dec.triu(diagonal=1).abs().max().item()
print(f"  biggest future weight anywhere: {future:.4f}"
      " - the future is sealed")
print("  token 4's row is unchanged: it has no future to hide")
print("  verdict: same block + one mask, now it can GENERATE (GPT-style)")

# ---------- Part 3: cross-attention - Q and K/V from different stacks ----
print("Part 3 - CROSS-attention: Q from the decoder, K/V from the encoder")
E = torch.tensor([[1.0, 0.0],     # encoder output, source token 1
                  [0.0, 1.0],     # encoder output, source token 2
                  [1.0, 1.0]])    # encoder output, source token 3
D = torch.tensor([[1.0, 0.0],     # decoder state, step 1
                  [1.0, 1.0]])    # decoder state, step 2
out, w_cross = attention(D, E, E)
print(f"  shapes: Q {tuple(D.shape)} from decoder,"
      f" K {tuple(E.shape)} and V {tuple(E.shape)} from encoder")
print(f"  weight matrix shape {tuple(w_cross.shape)}"
      " - RECTANGULAR: 2 asking, 3 answering")
for i, w in enumerate(w_cross):
    print(f"  decoder step {i + 1} reads source: {row(w)}"
          f"   row sum {w.sum().item():.4f}")
print(f"  output row 1: {row(out[0])}   output row 2: {row(out[1])}")
print(f"  output shape {tuple(out.shape)}: length follows the QUERIES"
      " (decoder),")
print("    content flows from the MEMORY (encoder output)")
print("  verdict: TRANSFORM one finished sequence into a new one"
      " (T5/Whisper)")
