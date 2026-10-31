"""Day 74 - Data Parallelism: The Straightforward Scale-Out.

Day 73's problem was one GPU that cannot hold training. Tonight's
first scale-out move: do NOT split the model. Copy the whole
model onto N workers and split the DATA instead.

  Part 1  The reference: Day 7's linear-regression gradient on
          the full dataset, one machine, one step.
  Part 2  Four simulated workers in one process. Each gets a
          quarter of the data (equal shards) and computes its
          own local gradient from the SAME starting theta.
          No worker steps yet - stepping alone would give
          four different models.
  Part 3  The ring all-reduce, watched step by step: the
          gradient is a 4-vector, so each worker owns one
          chunk. Scatter-reduce rotates chunks and adds
          partial sums; all-gather rotates the finished
          chunks until every worker holds the total. Average
          it, step with it, and prove the result is EXACTLY
          the single-machine step from Part 1.
  Part 4  The comm math for a 7B-parameter model in fp32:
          ring traffic per worker, and why it barely grows
          from N=4 to N=1024 - while the hub design blows up.

Pure Python stdlib, deterministic data built by a formula, no
RNG, no downloads, no paid APIs. Runs anywhere, including
the ~/ai-lab venv on a MacBook. One process plays all four
workers so you can watch the algorithm instead of a library
call hiding it.

Expected output:

  Part 1 - one machine, full batch (Day 7's gradient)
    theta0 = 0.500000, 0.250000, 0.000000, 0.000000
    full-batch grad = -15.875000, -85.828125, -65.891602, -54.102722
    after one step (lr 0.05) = 1.293750, 4.541406, 3.294580, 2.705136
  Part 2 - four workers, one data shard each, same theta0
    worker 0: shard x = 0.5, 2.5, 4.5, 6.5  local grad = -13.000000, -63.000000, -43.343750, -31.617188
    worker 1: shard x = 1.0, 3.0, 5.0, 7.0  local grad = -15.125000, -77.375000, -56.890625, -44.615234
    worker 2: shard x = 1.5, 3.5, 5.5, 7.5  local grad = -16.625000, -91.687500, -71.144531, -58.881592
    worker 3: shard x = 2.0, 4.0, 6.0, 8.0  local grad = -18.750000, -111.250000, -92.187500, -81.296875
    average of local grads = -15.875000, -85.828125, -65.891602, -54.102722
    diff vs full-batch grad = 0.000000
  Part 3 - ring all-reduce on the local gradients
    scatter-reduce 1/3: sends W0:c0  W1:c1  W2:c2  W3:c3 (receiver adds)
    scatter-reduce 2/3: sends W0:c3  W1:c0  W2:c1  W3:c2 (receiver adds)
    scatter-reduce 3/3: sends W0:c2  W1:c3  W2:c0  W3:c1 (receiver adds)
    after scatter-reduce, finished chunks live at: chunk 0 on W3, chunk 1 on W0, chunk 2 on W1, chunk 3 on W2
    all-gather 1/3: sends W0:c1  W1:c2  W2:c3  W3:c0 (receiver overwrites)
    all-gather 2/3: sends W0:c0  W1:c1  W2:c2  W3:c3 (receiver overwrites)
    all-gather 3/3: sends W0:c3  W1:c0  W2:c1  W3:c2 (receiver overwrites)
    every worker now holds the total sum:
      W0 = -63.500000, -343.312500, -263.566406, -216.410889
    all four workers identical: True
    averaged grad (sum / 4) = -15.875000, -85.828125, -65.891602, -54.102722
    diff vs full-batch grad = 0.000000
    after one step = 1.293750, 4.541406, 3.294580, 2.705136
    diff vs single-machine step = 0.000000
    verdict: same theta0 + same averaged grad = same step, on every replica - the replicas never diverge
  Part 4 - what the sync costs at 7B parameters (fp32)
    one gradient = 7,000,000,000 params x 4 bytes = 28.0 GB
    ring traffic per worker = 2(N-1)/N x grad size, 2(N-1) steps:
      N=4     ring: 42.0 GB/worker/step, 6 steps
      N=64    ring: 55.1 GB/worker/step, 126 steps
      N=1024  ring: 55.9 GB/worker/step, 2046 steps
    hub (parameter server) at N=64: one node must receive and resend 3,584 GB/step through a single link
    at 50 GB/s effective, the N=64 ring sync takes 1.10 s per step - and note
    the shard size never appears: sync cost depends on the
    MODEL size only, so bigger shards (more compute per
    step) make the sync relatively cheaper, not dearer
    one-line failure note: the step is a barrier - the
    slowest worker sets the step time, and one dead
    worker stalls the whole run (Day 78: checkpointing)
"""


def features(x):
    """A 4-wide feature vector so the gradient is a 4-vector.

    phi = [1, x, x^2/8, x^3/64]. Day 7's gradient formula is
    unchanged: dL/dtheta_j = mean(2 * err * phi_j). One feature
    would give a 1-vector and the ring would have nothing to
    chunk; four features give one chunk per worker.
    """
    return [1.0, x, x * x / 8.0, x * x * x / 64.0]


def predict(theta, phi):
    return sum(t * p for t, p in zip(theta, phi))


def gradient(theta, points):
    """Full-batch MSE gradient over a list of (x, y) points."""
    n = len(points)
    sums = [0.0, 0.0, 0.0, 0.0]
    for x, y in points:
        err = predict(theta, features(x)) - y
        for j, p in enumerate(features(x)):
            sums[j] += 2.0 * err * p
    return [s / n for s in sums]


def fmt(vec):
    return ", ".join(f"{v:.6f}" for v in vec)


# Deterministic data, built by a formula. x runs 0.5..8.0, the
# true line is y = 1 + 2x, and the noise is a small sawtooth.
# Every number is a dyadic fraction, so the arithmetic below
# is exact - the equality proofs in Part 3 print 0.0, not ~0.
POINTS = []
for i in range(1, 17):
    x = 0.5 * i
    noise = ((i * 7) % 5 - 2) * 0.25
    POINTS.append((x, 1.0 + 2.0 * x + noise))

THETA0 = [0.5, 0.25, 0.0, 0.0]
LR = 0.05
N_WORKERS = 4

# ---------- Part 1: the single-machine reference ----------
print("Part 1 - one machine, full batch (Day 7's gradient)")
full_grad = gradient(THETA0, POINTS)
theta_ref = [t - LR * g for t, g in zip(THETA0, full_grad)]
print(f"  theta0 = {fmt(THETA0)}")
print(f"  full-batch grad = {fmt(full_grad)}")
print(f"  after one step (lr 0.05) = {fmt(theta_ref)}")

# ---------- Part 2: four workers, four equal data shards ----------
print("Part 2 - four workers, one data shard each, same theta0")
shards = [POINTS[i::N_WORKERS] for i in range(N_WORKERS)]
# NB: POINTS[i::4] gives 4-point shards; equal sizes are what
# make "average of shard gradients" equal "gradient of all data".
local_grads = []
for r, shard in enumerate(shards):
    g = gradient(THETA0, shard)
    local_grads.append(g)
    print(f"  worker {r}: shard x = "
          f"{', '.join(f'{x:.1f}' for x, _ in shard)}  "
          f"local grad = {fmt(g)}")
naive_avg = [sum(g[j] for g in local_grads) / N_WORKERS
             for j in range(4)]
print(f"  average of local grads = {fmt(naive_avg)}")
print(f"  diff vs full-batch grad = "
      f"{max(abs(a - b) for a, b in zip(naive_avg, full_grad)):.6f}")

# ---------- Part 3: the ring all-reduce, step by step ----------
print("Part 3 - ring all-reduce on the local gradients")


def ring_allreduce(vectors):
    """Sum N vectors with the ring algorithm. Each vector has
    N elements, so worker r owns chunk r (one element here).
    Scatter-reduce: chunks rotate one hop per step and the
    receiver adds the incoming chunk into its copy. After
    N-1 steps each worker holds exactly one finished chunk.
    All-gather: finished chunks rotate until every worker
    holds all of them. Returns the summed vectors plus a
    human-readable trace of every step."""
    n = len(vectors)
    work = [list(v) for v in vectors]
    trace = []
    # Scatter-reduce.
    for step in range(n - 1):
        sends = []
        for r in range(n):
            send_chunk = (r - step) % n
            recv_chunk = (r - step - 1) % n
            sends.append((r, send_chunk, recv_chunk))
        # Apply the step: worker r receives chunk recv_chunk
        # from worker (r-1) and adds it into its own copy.
        new = [list(v) for v in work]
        for r, send_chunk, recv_chunk in sends:
            src = (r - 1) % n
            new[r][recv_chunk] += work[src][recv_chunk]
        work = new
        desc = "  ".join(f"W{r}:c{c}" for r, c, _ in sends)
        trace.append(f"  scatter-reduce {step + 1}/{n - 1}: "
                     f"sends {desc} (receiver adds)")
    # Who owns which finished chunk now? Check against the
    # naive total instead of trusting the textbook diagram.
    total = [sum(v[j] for v in vectors) for j in range(n)]
    owners = []
    for j in range(n):
        owner = next(r for r in range(n) if work[r][j] == total[j]
                     and all(work[r][k] != total[k] or k == j
                             for k in range(n)))
        owners.append(owner)
    trace.append("  after scatter-reduce, finished chunks live at: "
                 + ", ".join(f"chunk {j} on W{owners[j]}"
                             for j in range(n)))
    # All-gather.
    for step in range(n - 1):
        sends = []
        for r in range(n):
            send_chunk = (r - step + 1) % n
            recv_chunk = (r - step) % n
            sends.append((r, send_chunk, recv_chunk))
        new = [list(v) for v in work]
        for r, send_chunk, recv_chunk in sends:
            src = (r - 1) % n
            new[r][recv_chunk] = work[src][recv_chunk]
        work = new
        desc = "  ".join(f"W{r}:c{c}" for r, c, _ in sends)
        trace.append(f"  all-gather {step + 1}/{n - 1}: "
                     f"sends {desc} (receiver overwrites)")
    return work, trace, total


summed, trace, naive_total = ring_allreduce(local_grads)
for line in trace:
    print(line)
print(f"  every worker now holds the total sum:")
print(f"    W0 = {fmt(summed[0])}")
identical = all(v == summed[0] for v in summed)
print(f"  all four workers identical: {identical}")
ring_avg = [v / N_WORKERS for v in summed[0]]
theta_ring = [t - LR * g for t, g in zip(THETA0, ring_avg)]
print(f"  averaged grad (sum / 4) = {fmt(ring_avg)}")
print(f"  diff vs full-batch grad = "
      f"{max(abs(a - b) for a, b in zip(ring_avg, full_grad)):.6f}")
print(f"  after one step = {fmt(theta_ring)}")
print(f"  diff vs single-machine step = "
      f"{max(abs(a - b) for a, b in zip(theta_ring, theta_ref)):.6f}")
print("  verdict: same theta0 + same averaged grad = same step, "
      "on every replica - the replicas never diverge")

# ---------- Part 4: comm math for a 7B model, fp32 ----------
print("Part 4 - what the sync costs at 7B parameters (fp32)")
params = 7_000_000_000
grad_gb = params * 4 / 1e9
print(f"  one gradient = {params:,} params x 4 bytes = {grad_gb:.1f} GB")
print("  ring traffic per worker = 2(N-1)/N x grad size, "
      "2(N-1) steps:")
for n in (4, 64, 1024):
    traffic = 2 * (n - 1) / n * grad_gb
    print(f"    N={n:<5} ring: {traffic:.1f} GB/worker/step, "
          f"{2 * (n - 1)} steps")
hub_gb = 64 * grad_gb * 2
print(f"  hub (parameter server) at N=64: one node must receive "
      f"and resend {hub_gb:,.0f} GB/step through a single link")
print(f"  at 50 GB/s effective, the N=64 ring sync takes "
      f"{2 * 63 / 64 * grad_gb / 50:.2f} s per step - and note")
print("  the shard size never appears: sync cost depends on the")
print("  MODEL size only, so bigger shards (more compute per")
print("  step) make the sync relatively cheaper, not dearer")
print("  one-line failure note: the step is a barrier - the")
print("  slowest worker sets the step time, and one dead")
print("  worker stalls the whole run (Day 78: checkpointing)")
