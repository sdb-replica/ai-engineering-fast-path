"""Day 14 - Softmax & Cross-Entropy: turning scores into probabilities.

A classifier's raw scores (logits) can be anything - negative, huge, not summing
to 1. Softmax squashes them into a probability distribution; cross-entropy bills
you -log(p_true) for the answer you got wrong. And the punchline: the gradient
of the pair collapses to (prediction - target) - the same "error" pattern as
Day 6's regression gradient and Day 11's MLE.

Runs on the ~/ai-lab venv: torch CPU or MPS, no paid APIs.
Expected output is shown in the lesson HTML - numbers below should match to the
printed precision.
"""
import torch
import torch.nn.functional as F

torch.set_printoptions(precision=4)

print("=== 1. Logits -> probabilities: softmax by hand ===")
logits = torch.tensor([2.0, 1.0, 0.1])          # [bull, bear, chop] conviction
hand = torch.exp(logits) / torch.exp(logits).sum()
lib = F.softmax(logits, dim=0)
print("hand :", hand.tolist())
print("torch:", lib.tolist())
print(f"sum  : {float(lib.sum()):.6f}")             # a real distribution: sums to 1

print("\n=== 2. The overflow gotcha (and the one-line fix) ===")
big = torch.tensor([1000.0, 1001.0, 1002.0])
naive = torch.exp(big) / torch.exp(big).sum()    # exp(1002) = inf -> nan
print("naive softmax:", naive.tolist())
stable = F.softmax(big, dim=0)                   # subtracts max(z) internally
print("torch softmax :", stable.tolist())
print("softmax(z) == softmax(z - max(z)): shifting scores changes nothing")

print("\n=== 3. Cross-entropy: the loss = -log(p_true) ===")
probs = F.softmax(logits, dim=0)
true_class = 0                                   # bull was right
ce_hand = -torch.log(probs[true_class])
# F.cross_entropy takes RAW logits (it applies log_softmax inside) + class index
ce_lib = F.cross_entropy(logits.unsqueeze(0), torch.tensor([true_class]))
print(f"hand: -log({float(probs[true_class]):.4f}) = {float(ce_hand):.4f}")
print(f"lib : F.cross_entropy(raw_logits, target)  = {float(ce_lib):.4f}")
print("(minimizing this = maximizing likelihood: Day 11's MLE, one line)")

print("\n=== 4. The punchline gradient: dL/dz = p - y ===")
z = logits.clone().requires_grad_(True)
F.cross_entropy(z.unsqueeze(0), torch.tensor([true_class])).backward()
onehot = F.one_hot(torch.tensor([true_class]), num_classes=3).float()
print("autograd dL/dz:", [f"{v:.4f}" for v in z.grad.tolist()])
print("p - y        :", [f"{v:.4f}" for v in (probs - onehot).tolist()])
print("identical. Gradient = how-wrong x how-much-I-pushed: the Day 6 pattern.")

print("\n=== 5. Your turn: regime scores -> allocation weights ===")
features = torch.tensor([0.04, 1.25, 0.62])      # momentum, volume_z, vol_spike
W = torch.tensor([[ 1.20,  0.35, -0.80],          # bull row (Day 12: neurons)
                  [-1.05,  0.15,  0.60],          # bear row
                  [ 0.30, -0.90,  0.45]])         # chop row
b = torch.tensor([0.10, 0.05, 0.00])
regime_logits = W @ features + b                  # three neurons, one matmul
alloc = F.softmax(regime_logits, dim=0)
print("logits:", [f"{v:.4f}" for v in regime_logits.tolist()])
print(f"alloc : bull {float(alloc[0]):.1%} | bear {float(alloc[1]):.1%} | chop {float(alloc[2]):.1%}")
print(f"loss if bear wins (-log p): {float(-torch.log(alloc[1])):.4f}")
print("\nWIN: you turned raw conviction into a sized book - and priced the mistake.")
