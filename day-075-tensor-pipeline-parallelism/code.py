"""Day 75 - Tensor & Pipeline Parallelism: Splitting the Model Itself.

Day 74 split the data: every GPU held the whole model and
trained on a shard of the rows. But Day 73's wall said the
model itself - weights + grads + Adam state - can exceed one
GPU. When the model doesn't fit, you split the MODEL. Two
cuts, at two different granularities:

  Tensor parallelism (TP) cuts INSIDE one layer. A layer is
  mostly one matmul, Y = X @ W, and a matmul can be cut two
  ways: by columns of W (each GPU owns some output features
  outright) or by rows of W (each GPU holds some input
  features and produces a PARTIAL sum - the partials must be
  added together, an all-reduce, before the layer's output
  exists). Megatron-style TP pairs the two cuts across a
  transformer's two matmuls, so the all-reduce lands once per
  layer - in the forward pass and again in the backward. That
  seam is on the critical path of every layer, so TP lives
  inside one node, on fast interconnect (NVLink), at small
  degree (2/4/8).

  Pipeline parallelism (PP) cuts ACROSS layers. Stage 1 holds
  layers 1-8, stage 2 layers 9-16, and activations flow
  stage to stage like cars on an assembly line. The messages
  are tiny (one activation tensor per boundary), so PP can
  span nodes - but the line takes time to fill: with P stages
  and M micro-batches, a step takes M + P - 1 slots and the
  idle "bubble" fraction is (P-1)/(M+P-1). Micro-batches are
  how you buy the bubble down.

  Part 1  One matmul, both TP cuts. X (4x8) @ W (8x6) with
          integer values built by formula, so float sums are
          exact in any order. Column cut: concat the shards.
          Row cut: all-reduce the partials. Both match the
          one-GPU answer with max diff 0.0 - TP changes where
          the numbers live, never what they are. Each GPU
          holds 24 of W's 48 numbers.
  Part 2  The pipeline bubble, priced. For P = 4 stages,
          print the schedule as a Gantt (M = 4) and the bubble
          table for M = 1, 4, 8, 32: 75.0% idle at M = 1,
          8.6% at M = 32. GPipe's trade in one line: more
          micro-batches shrink the bubble but keep more
          activations in flight.

Runs on the ~/ai-lab venv: numpy only (free), CPU on purpose,
no CUDA, no paid APIs, no downloads. No RNG - every matrix
entry comes from a formula, so every digit is exact.

Expected output:

  Part 1 - tensor parallelism: one matmul, two GPUs, two cuts
    X (4, 8) @ W (8, 6) -> Y (4, 6), W holds 48 numbers
    column cut: GPU0 W[:, :3] + GPU1 W[:, 3:], concat -> max diff vs one GPU: 0.0
    row cut:    GPU0 rows 0-3, GPU1 rows 4-7, partial sums all-reduced -> max diff: 0.0
    Y row 0, one GPU:  [3, -11, 15, -4, -3, 3]
    Y row 0, partials: [8, -11, 5, -4, 2, 8] + [-5, 0, 10, 0, -5, -5]  (neither is the answer alone)
    memory: each GPU stores 24 of W's 48 numbers - TP halves the layer per GPU
  Part 2 - pipeline parallelism: P stages, M micro-batches
    schedule for M = 4 (each token = one slot, . = bubble):
      stage 1: m1 m2 m3 m4  .  .  .
      stage 2:  . m1 m2 m3 m4  .  .
      stage 3:  .  . m1 m2 m3 m4  .
      stage 4:  .  .  . m1 m2 m3 m4
    M =  1: step =  4 slots, bubble = 3 slots (75.0%) per stage
    M =  4: step =  7 slots, bubble = 3 slots (42.9%) per stage
    M =  8: step = 11 slots, bubble = 3 slots (27.3%) per stage
    M = 32: step = 35 slots, bubble = 3 slots (8.6%) per stage
    verdict: micro-batches buy the bubble down - the cost is activations kept in flight
    TP shards weights inside a layer, PP shards layers, DP (Day 74) shards data - clusters compose all three
"""
import numpy as np

# ---------- Part 1: tensor parallelism - cut the matmul ----------
print("Part 1 - tensor parallelism: one matmul, two GPUs, two cuts")
# X: 4 rows (tokens) x 8 features.  W: 8 inputs -> 6 outputs.
# Integer values from formulas: products and sums are exact in
# float64 no matter what order they are added in.
X = np.array([[((i * 3 + j * 5) % 7) - 3 for j in range(8)]
              for i in range(4)], dtype=np.float64)
W = np.array([[((i * 2 + j * 3) % 5) - 2 for j in range(6)]
              for i in range(8)], dtype=np.float64)
Y = X @ W                                    # the one-GPU answer
print(f"  X {X.shape} @ W {W.shape} -> Y {Y.shape}, "
      f"W holds {W.size} numbers")

# Cut 1 - by columns: GPU 0 owns output features 0-2, GPU 1 owns 3-5.
Y_left = X @ W[:, :3]
Y_right = X @ W[:, 3:]
Y_cols = np.concatenate([Y_left, Y_right], axis=1)
d_cols = np.abs(Y_cols - Y).max()
print(f"  column cut: GPU0 W[:, :3] + GPU1 W[:, 3:], concat -> "
      f"max diff vs one GPU: {d_cols:.1f}")

# Cut 2 - by rows: GPU 0 owns input features 0-3, GPU 1 owns 4-7.
# Each computes a PARTIAL sum; neither holds the answer alone.
P0 = X[:, :4] @ W[:4, :]
P1 = X[:, 4:] @ W[4:, :]
Y_rows = P0 + P1                             # <-- the all-reduce
d_rows = np.abs(Y_rows - Y).max()
print(f"  row cut:    GPU0 rows 0-3, GPU1 rows 4-7, partial sums "
      f"all-reduced -> max diff: {d_rows:.1f}")
print(f"  Y row 0, one GPU:  {Y[0].astype(int).tolist()}")
print(f"  Y row 0, partials: {P0[0].astype(int).tolist()} + "
      f"{P1[0].astype(int).tolist()}  (neither is the answer alone)")
print(f"  memory: each GPU stores {W.size // 2} of W's {W.size} "
      f"numbers - TP halves the layer per GPU")

# ---------- Part 2: pipeline parallelism - price the bubble ----------
print("Part 2 - pipeline parallelism: P stages, M micro-batches")
P = 4

def bubble(P, M):
    """Step slots, idle slots per stage, bubble fraction."""
    slots = M + P - 1                        # fill + steady + drain
    return slots, P - 1, (P - 1) / slots

print("  schedule for M = 4 (each token = one slot, . = bubble):")
for s in range(P):
    cells = []
    for t in range(bubble(P, 4)[0]):
        m = t - s                            # stage s sees mb m at t
        cells.append(f"m{m + 1}" if 0 <= m < 4 else " .")
    print(f"    stage {s + 1}: " + " ".join(cells))
for M in (1, 4, 8, 32):
    slots, idle, frac = bubble(P, M)
    print(f"  M = {M:>2}: step = {slots:>2} slots, bubble = {idle} "
          f"slots ({frac:.1%}) per stage")
print("  verdict: micro-batches buy the bubble down - the cost is "
      "activations kept in flight")
print("  TP shards weights inside a layer, PP shards layers, "
      "DP (Day 74) shards data - clusters compose all three")
