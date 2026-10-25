"""Day 28 - RNNs & Sequences: Recurrence and the Fading Memory.

Every model so far took a fixed-size input and had no memory:
shuffle the rows (Day 19) and an MLP cannot even tell. Sequences
-- words in a sentence, candles in a session, ticks in a stream --
break both assumptions. The length varies, and the ORDER is the
information: step t only makes sense given what came before it.

A recurrent neural network (RNN) answers with one cell, reused:

  h_t = tanh(W_xh @ x_t + W_hh @ h_prev + b),   h_0 = 0

h is the hidden state -- a fixed-size running summary, rewritten
at every step from the new input and its own previous value. The
SAME W_xh and W_hh are used at every step: weight sharing over
time, exactly as Day 26's kernel shared weights over space. So
the parameter count ignores the sequence length, and any length
works. Unroll the loop in time and the RNN is a very deep
network -- one layer per step, all sharing weights -- whose
final hidden state (or the state at every step) feeds an
ordinary head, e.g. Day 24's Linear layer.

  Part 1  The cell by hand: fixed weights, a 4-step sequence of
          toy candle moves (+1.0, +0.5, -1.0, +1.0), print h
          after every step and watch step 3's -1.0 rewrite the
          summary the first two steps built.
  Part 2  nn.RNN is the same cell, packaged: copy the Part 1
          weights in, check the outputs match to 0.000000, and
          show the parameter count (10) does not change when the
          sequence grows from T=4 to T=12.
  Part 3  Why they fade: a scalar gradient probe (w_hh = 0.8,
          ten identical inputs of 0.2). The loss reads only the
          FINAL hidden state; autograd reports how much each
          step's input could have changed it. The first step's
          gradient arrives about 100x smaller than the last
          step's -- backprop through time multiplies by a factor
          below 1 at every hop (about 0.53 here), and products
          of numbers below 1 vanish exponentially. That is why
          plain RNNs learn short-range patterns and go deaf to
          distant ones.

Runs on the ~/ai-lab venv: torch only (free), CPU or MPS, no
CUDA, no paid APIs, no downloads. No randomness anywhere in
this file -- every weight is hand-set, so your output should
match the expected output below digit for digit.

Expected output:

  device: mps
  Part 1 - one cell, unrolled by hand
    h0 = [+0.0000, +0.0000]
    t=1  x=+1.0  h=[+0.6640, -0.3799]
    t=2  x=+0.5  h=[+0.6006, -0.1676]
    t=3  x=-1.0  h=[-0.4750, +0.5737]
    t=4  x=+1.0  h=[+0.5510, -0.2594]
  Part 2 - nn.RNN: the same cell, packaged
    input shape (1, 4, 1)  (batch, time, features)
    output shape (1, 4, 2)  (hidden state at every step)
    h_n shape (1, 1, 2)  (the final hidden state)
    hand loop vs nn.RNN max diff: 0.000000
    params: 10  (T=4)
    T=12 output shape (1, 12, 2)  params still 10
  Part 3 - why they fade: the gradient through time
    final h after 10 steps: 0.5803
    grad wrt x_t, t=1..10: 0.0066, 0.0086, 0.0123, 0.0190, 0.0319, 0.0561, 0.1020, 0.1888, 0.3529, 0.6633
    last step grad 0.6633 vs first step grad 0.0066 -- ratio 99.7x
"""
import torch
import torch.nn as nn

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)

# ---------- Part 1: one cell, unrolled by hand ----------
print("Part 1 - one cell, unrolled by hand")
W_xh = torch.tensor([[0.8], [-0.5]], device=device)          # (2, 1)
W_hh = torch.tensor([[0.5, 0.1], [0.2, 0.4]], device=device)  # (2, 2)
b_h = torch.tensor([0.0, 0.1], device=device)
xs = [1.0, 0.5, -1.0, 1.0]  # toy candle moves
h = torch.zeros(2, device=device)                             # h_0 = 0
print(f"  h0 = [{h[0]:+.4f}, {h[1]:+.4f}]")
hand_hs = []
for t, xv in enumerate(xs, 1):
    x_t = torch.tensor([xv], device=device)
    h = torch.tanh(W_xh @ x_t + W_hh @ h + b_h)               # the cell
    hand_hs.append(h.clone())
    print(f"  t={t}  x={xv:+.1f}  h=[{h[0]:+.4f}, {h[1]:+.4f}]")

# ---------- Part 2: nn.RNN -- the same cell, packaged ----------
print("Part 2 - nn.RNN: the same cell, packaged")
rnn = nn.RNN(input_size=1, hidden_size=2, batch_first=True).to(device)
with torch.no_grad():
    rnn.weight_ih_l0.copy_(W_xh)   # PyTorch's name for W_xh
    rnn.weight_hh_l0.copy_(W_hh)   # PyTorch's name for W_hh
    rnn.bias_ih_l0.copy_(b_h)      # the cell's bias lives here...
    rnn.bias_hh_l0.zero_()         # ...so the second bias stays 0
x_seq = torch.tensor(xs, device=device).reshape(1, 4, 1)  # (B, T, F)
with torch.no_grad():
    out, h_n = rnn(x_seq)
print(f"  input shape {tuple(x_seq.shape)}  (batch, time, features)")
print(f"  output shape {tuple(out.shape)}  (hidden state at every step)")
print(f"  h_n shape {tuple(h_n.shape)}  (the final hidden state)")
hand = torch.stack(hand_hs)                                   # (4, 2)
diff = (hand - out[0]).abs().max().item()
print(f"  hand loop vs nn.RNN max diff: {diff:.6f}")
n_params = sum(p.numel() for p in rnn.parameters())
print(f"  params: {n_params}  (T=4)")
x_long = torch.zeros(1, 12, 1, device=device)
with torch.no_grad():
    out12, _ = rnn(x_long)
print(f"  T=12 output shape {tuple(out12.shape)}  params still {n_params}")

# ---------- Part 3: why they fade -- the gradient through time ----------
print("Part 3 - why they fade: the gradient through time")
x = torch.full((10,), 0.2, requires_grad=True, device=device)
h = torch.zeros((), device=device)
for t in range(10):
    h = torch.tanh(0.8 * h + 1.0 * x[t])   # scalar cell: w_hh=0.8, w_xh=1.0
print(f"  final h after 10 steps: {h.item():.4f}")
h.backward()                               # blame flows back through time
g = x.grad
print("  grad wrt x_t, t=1..10: " + ", ".join(f"{v:.4f}" for v in g.tolist()))
print(f"  last step grad {g[-1]:.4f} vs first step grad {g[0]:.4f}"
      f" -- ratio {(g[-1] / g[0]).item():.1f}x")
