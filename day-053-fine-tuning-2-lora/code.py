"""Day 53 - Fine-Tuning 2: LoRA / QLoRA.

Yesterday (Day 52) SFT taught the basic recipe - and its price:
full fine-tuning rewrites EVERY weight, so for every task you
pay for a full copy of the model plus the training state to
move it (grads + Adam's two states per weight). Tonight: don't
rebuild the binary. Ship a patch.

    LoRA: freeze W. Learn the update as a product of two thin
    matrices, dW = B x A, where the middle dimension r is tiny
    next to d and k. Trainable params per layer fall from
    d x k to r(d + k). A starts random-ish, B starts at ZERO -
    so at init dW = 0 and the adapted model is the base model,
    exactly. When training is done, merge: W' = W + B x A is
    one ordinary matrix again, and serving pays nothing extra.

  Part 1  Build LoRALinear by hand around a frozen Linear.
          Hand-set W and A, B zero-init: the adapted layer's
          outputs match the plain Linear with max diff 0.
  Part 2  Train ONLY A and B on a toy target (SGD, 30 steps).
          The loss falls, dW = B x A fills in - and W itself
          is bit-for-bit untouched, because the optimizer was
          never handed it.
  Part 3  Merge: W' = W + B x A in a fresh Linear reproduces
          the adapted layer exactly. One matrix, zero extra
          latency, and the adapter file you kept is 5 numbers.
  Part 4  The bill at real scale: one 4096 x 4096 layer.
          Full FT trains 16,777,216 params; LoRA r=16 trains
          131,072 - 0.78% - and the adapter is a 0.50 MiB file.

QLoRA (theory section) is this same patch on a base frozen
harder: W stored in 4 bits instead of 16, dequantized on the
fly, adapters still trained in 16-bit. Quantization gets its
own day (Day 65); tonight the patch is the concept.

Runs on the ~/ai-lab venv: torch only (free), CPU on purpose
(tiny hand-set matrices, so every digit is exact - no RNG
anywhere), no CUDA, no paid APIs, no downloads. Every claimed
number below was verified with a float32/float64 numpy twin
of the exact training math.

Expected output:

  Part 1 - at init, the adapted layer IS the base layer
    base Linear(3, 2): 8 params - and LoRA freezes every one
    LoRA r=1 patch: A (1, 3) + B (2, 1) = 5 trainable params
    plain Linear preds:   3.5000, 2.5000, -3.0000, 3.5000
    adapted preds, init:  3.5000, 2.5000, -3.0000, 3.5000
    max diff at init: 0.000000  (B = 0 -> dW = B x A = 0 -> base model, exactly)
  Part 2 - train ONLY the patch (SGD lr=0.05, 30 steps, MSE to y)
    step  0: loss 4.4375
    step 10: loss 0.7976
    step 20: loss 0.5837
    step 30: loss 0.5399
    W after training == W before: True  (frozen means frozen)
    dW = B x A after training:
      row 0: 0.5477, -0.2824, 0.6127
      row 1: 0.5445, -0.2807, 0.6092
    the full 2x3 update, written by 5 numbers - rank 1 by construction
  Part 3 - merge: fold the patch into W and serve one matrix
    merged weight W' = W + dW:
      row 0: 1.5477, 0.2176, -0.3873
      row 1: 0.5445, 1.7193, 1.6092
    merged layer vs adapted layer: max diff 0.000000
    one matrix again - inference pays zero extra latency
  Part 4 - the bill at real scale: ONE 4096 x 4096 layer
    full fine-tune: 16,777,216 trainable params
      fp32 weights 64 MiB - weights + grads + Adam m/v = 256 MiB per layer
    LoRA r= 8: 65,536 trainable = 0.39% of full - adapter file 0.25 MiB
    LoRA r=16: 131,072 trainable = 0.78% of full - adapter file 0.50 MiB
    LoRA r=64: 524,288 trainable = 3.12% of full - adapter file 2.00 MiB
    verdict: freeze the binary, train the patch, merge before shipping
"""
import torch
import torch.nn as nn

fmt = lambda t: ", ".join(f"{v:.4f}" for v in t.flatten().tolist())


class LoRALinear(nn.Module):
    """A frozen nn.Linear plus a trainable low-rank patch.

    forward(x) = base(x) + scaling * (x @ A.T @ B.T)
               = (W + scaling * B x A) x + b

    W never trains: every base parameter is frozen, and the
    optimizer is only ever handed A and B. A is (r, d_in) -
    it squeezes the input down to r numbers. B is (d_out, r) -
    it lifts them back. B starts at zero, so the patch
    contributes exactly nothing until training moves it.
    scaling = alpha / r keeps the patch's size steady when
    you change r (alpha=1 here, so scaling is 1 / r).
    """

    def __init__(self, base, r, alpha=1.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)            # freeze W and b
        d_out, d_in = base.weight.shape
        self.r = r
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.zeros(r, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, r))   # zero-init

    def forward(self, x):
        patch = (x @ self.A.T) @ self.B.T      # squeeze, then lift
        return self.base(x) + self.scaling * patch

    def merged_weight(self):
        """W' = W + scaling * B x A - the patch, folded in."""
        with torch.no_grad():
            return self.base.weight + self.scaling * (self.B @ self.A)


# ---------- Part 1: at init, adapted == base, exactly ----------
print("Part 1 - at init, the adapted layer IS the base layer")
W0 = torch.tensor([[1.0, 0.5, -1.0], [0.0, 2.0, 1.0]])
b0 = torch.tensor([0.5, -0.5])
base = nn.Linear(3, 2)
with torch.no_grad():
    base.weight.copy_(W0)
    base.bias.copy_(b0)
lora = LoRALinear(base, r=1, alpha=1.0)
with torch.no_grad():
    # In real LoRA, A is drawn random; here it is hand-set so
    # every digit is checkable. B stays zero - that is the init.
    lora.A.copy_(torch.tensor([[0.5, -1.0, 0.25]]))
n_frozen = sum(p.numel() for p in lora.parameters()
               if not p.requires_grad)
n_train = sum(p.numel() for p in lora.parameters() if p.requires_grad)
print(f"  base Linear(3, 2): {n_frozen} params - and LoRA freezes "
      "every one")
print(f"  LoRA r=1 patch: A (1, 3) + B (2, 1) = {n_train} trainable "
      "params")
X = torch.tensor([[1.0, 2.0, -1.0], [0.5, 0.0, 4.0]])
y = torch.tensor([[4.0, 1.0], [0.0, 6.0]])
with torch.no_grad():
    plain = nn.functional.linear(X, W0, b0)
    adapted = lora(X)
    diff0 = (plain - adapted).abs().max().item()
print(f"  plain Linear preds:   {fmt(plain)}")
print(f"  adapted preds, init:  {fmt(adapted)}")
print(f"  max diff at init: {diff0:.6f}  (B = 0 -> dW = B x A = 0 "
      "-> base model, exactly)")

# ---------- Part 2: train ONLY the patch ----------
print("Part 2 - train ONLY the patch (SGD lr=0.05, 30 steps, "
      "MSE to y)")
opt = torch.optim.SGD([lora.A, lora.B], lr=0.05)   # W not handed over
for step in range(31):
    loss = ((lora(X) - y) ** 2).mean()
    if step % 10 == 0:
        print(f"  step {step:>2}: loss {loss.item():.4f}")
    if step < 30:
        opt.zero_grad()
        loss.backward()
        opt.step()
with torch.no_grad():
    dW = lora.B @ lora.A                       # scaling is 1.0 (r=1)
    w_same = bool(torch.equal(lora.base.weight, W0))
print(f"  W after training == W before: {w_same}  (frozen means "
      "frozen)")
print("  dW = B x A after training:")
for i in range(dW.shape[0]):
    print(f"    row {i}: {fmt(dW[i])}")
print("  the full 2x3 update, written by 5 numbers - rank 1 by "
      "construction")

# ---------- Part 3: merge for serving ----------
print("Part 3 - merge: fold the patch into W and serve one matrix")
merged = nn.Linear(3, 2)
with torch.no_grad():
    merged.weight.copy_(lora.merged_weight())
    merged.bias.copy_(lora.base.bias)
    mdiff = (merged(X) - lora(X)).abs().max().item()
    Wp = lora.merged_weight()
print("  merged weight W' = W + dW:")
for i in range(Wp.shape[0]):
    print(f"    row {i}: {fmt(Wp[i])}")
print(f"  merged layer vs adapted layer: max diff {mdiff:.6f}")
print("  one matrix again - inference pays zero extra latency")

# ---------- Part 4: the bill at real scale ----------
print("Part 4 - the bill at real scale: ONE 4096 x 4096 layer")
d = k = 4096
full = d * k
print(f"  full fine-tune: {full:,} trainable params")
print(f"    fp32 weights {full * 4 / 2**20:.0f} MiB - weights + "
      f"grads + Adam m/v = {full * 16 / 2**20:.0f} MiB per layer")
for r in (8, 16, 64):
    n = r * (d + k)
    print(f"  LoRA r={r:>2}: {n:,} trainable = {100 * n / full:.2f}% "
          f"of full - adapter file {n * 4 / 2**20:.2f} MiB")
print("  verdict: freeze the binary, train the patch, merge "
      "before shipping")
