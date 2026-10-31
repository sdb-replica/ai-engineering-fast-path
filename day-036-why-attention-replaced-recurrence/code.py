"""Day 36 - Sequence Modeling Recap: Why Attention Replaced Recurrence.

Phase B ended with Day 28's recurrent cell as our best sequence
model:

  h_t = tanh(W_xh @ x_t + W_hh @ h_prev + b),   h_0 = 0

One cell, reused at every step, carrying a fixed-size hidden
state h -- a lossy running summary of everything seen so far.
Today we stop adding machinery and instead audit that design,
the way you would audit a system you inherited. It has three
structural problems, and none of them is a tuning problem:

  1. SEQUENTIAL. Step t cannot start until step t-1 has
     finished: the computation is a chain of T dependent steps.
     Day 13's GPUs win by doing many independent things at
     once; a chain keeps one lane busy and T-1 lanes idle.
  2. FADING MEMORY. Day 28 measured the gradient arriving at
     step 1 about 100x weaker than at step 10. Extend the
     sequence and the fade compounds -- and it is not only the
     gradient. The information itself is rewritten away.
  3. THE SQUEEZE. Everything the model will ever know about
     the past must fit inside h, whose size is fixed while the
     sequence is not. A 10,000-step past, compressed into the
     same fixed number of floats as a 10-step past.

  Part 1  Day 28's gradient probe, stretched from T=10 to
          T=20: the first step's gradient is now tens of
          thousands of times weaker than the last step's.
  Part 2  The distance test, on the forward pass this time.
          Two sequences that differ ONLY in a marker at step 1
          (+1.0 vs -1.0), then identical filler. Read the final
          state: if the marker was 1 hop back, the states are
          far apart; 19 hops back, they are indistinguishable.
          Memory does not fade in storage -- it is overwritten.
  Part 3  The three problems, counted. For T = 4, 20, 100:
          chained steps, hops from step 1 to the output, and
          how much the past gets to occupy -- against a design
          that keeps every step and lets the reader look at
          all of them at once: 1 parallel step, 1 hop, no
          squeeze. That design has a name -- attention -- and
          tomorrow we open the box.

Runs on the ~/ai-lab venv: torch only (free), CPU or MPS, no
CUDA, no paid APIs, no downloads. No randomness anywhere --
the same hand-set scalar cell as Day 28 (w_hh = 0.8,
w_xh = 1.0), so your output should match the expected output
below digit for digit.

Expected output:

  device: mps
  Part 1 - the gradient fade, stretched to T=20
    final h after 20 steps: 0.5822
    grad wrt x_1: 1.1426e-05
    grad wrt x_5: 5.4759e-05
    grad wrt x_10: 1.1396e-03
    grad wrt x_15: 2.7360e-02
    grad wrt x_20: 6.6109e-01
    last step 0.6611 vs first step 1.1426e-05 -- ratio 57,860x
  Part 2 - the distance test: can the final state still see step 1?
    marker 1 hop back:   h = +0.6692 (marker +1) vs -0.3879 (marker -1), gap 1.0570
    marker 9 hops back:  h = +0.5827 (marker +1) vs +0.5654 (marker -1), gap 0.0172
    marker 19 hops back: h = +0.5822 (marker +1) vs +0.5821 (marker -1), gap 0.000030
    the marker's fingerprint is 35.3 thousand x weaker at 19 hops than at 1 hop
  Part 3 - the three problems, counted (hidden size fixed at 4)
    T=4    RNN:   4 chained steps,  3 hops, state = 4 numbers   all-at-once: 1 step, 1 hop, keeps 16 numbers
    T=20   RNN:  20 chained steps, 19 hops, state = 4 numbers   all-at-once: 1 step, 1 hop, keeps 80 numbers
    T=100  RNN: 100 chained steps, 99 hops, state = 4 numbers   all-at-once: 1 step, 1 hop, keeps 400 numbers
    the chain and the hops grow with T; the state never does
"""
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)

# The Day 28 scalar cell, reused all day: h = tanh(0.8 * h + 1.0 * x)

# ---------- Part 1: the gradient fade, stretched to T=20 ----------
print("Part 1 - the gradient fade, stretched to T=20")
x = torch.full((20,), 0.2, requires_grad=True, device=device)
h = torch.zeros((), device=device)
for t in range(20):
    h = torch.tanh(0.8 * h + 1.0 * x[t])   # one chained step; t+1 waits for t
print(f"  final h after 20 steps: {h.item():.4f}")
h.backward()                               # blame flows back through all 20 hops
g = x.grad.tolist()
for t in (0, 4, 9, 14, 19):
    print(f"  grad wrt x_{t + 1}: {g[t]:.4e}")
print(f"  last step {g[-1]:.4f} vs first step {g[0]:.4e}"
      f" -- ratio {g[-1] / g[0]:,.0f}x")

# ---------- Part 2: the distance test ----------
print("Part 2 - the distance test: can the final state still see step 1?")

def final_state(marker, fillers):
    """Run the cell on [marker] + fillers of 0.2; return the final h."""
    h = torch.zeros((), device=device)
    with torch.no_grad():
        for xv in [marker] + [0.2] * fillers:
            h = torch.tanh(0.8 * h + xv)
    return h.item()

gaps = {}
for hops in (1, 9, 19):
    hp = final_state(1.0, hops)    # step 1 was +1.0, then `hops` filler steps
    hm = final_state(-1.0, hops)   # step 1 was -1.0, same filler after it
    gap = abs(hp - hm)
    gaps[hops] = gap
    gap_s = f"{gap:.6f}" if hops == 19 else f"{gap:.4f}"
    label = f"marker {hops} hop{'s' if hops != 1 else ''} back:"
    print(f"  {label.ljust(21)}h = {hp:+.4f} (marker +1)"
          f" vs {hm:+.4f} (marker -1), gap {gap_s}")
print(f"  the marker's fingerprint is {gaps[1] / gaps[19] / 1000:.1f} thousand x"
      " weaker at 19 hops than at 1 hop")

# ---------- Part 3: the three problems, counted ----------
print("Part 3 - the three problems, counted (hidden size fixed at 4)")
H = 4  # the hidden state is 4 numbers, whatever T does
for T in (4, 20, 100):
    print("  T=%-4d RNN: %3d chained steps, %2d hops, state = %d numbers"
          "   all-at-once: 1 step, 1 hop, keeps %d numbers"
          % (T, T, T - 1, H, T * H))
print("  the chain and the hops grow with T; the state never does")
