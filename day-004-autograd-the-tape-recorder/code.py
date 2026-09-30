"""Day 4: Autograd -- the tape recorder.

Every op on a requires_grad tensor stamps a receipt (grad_fn).
z.backward() replays the tape in reverse, filling .grad on the
leaves. Grads accumulate -- you own zero_grad().
Run:  python3 code.py   (inside the ~/ai-lab venv)
"""
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("using device:", device)

print("\n--- 1. The tape: every op leaves a receipt ---")
x = torch.tensor(2.0, requires_grad=True, device=device)  # leaf
w = torch.tensor(3.0, requires_grad=True, device=device)  # leaf
b = torch.tensor(1.0, requires_grad=True, device=device)  # leaf

y = w * x              # 6.0 -- stamps MulBackward0
z = y + b              # 7.0 -- stamps AddBackward0
print("y.grad_fn:", type(y.grad_fn).__name__)
print("z.grad_fn:", type(z.grad_fn).__name__)
print("x is a leaf:", x.is_leaf, "| x.grad_fn:", x.grad_fn)

z.backward()           # replay the tape from z back to the leaves
print("dz/dx =", x.grad.item(), "(expected: w = 3)")
print("dz/dw =", w.grad.item(), "(expected: x = 2)")
print("dz/db =", b.grad.item(), "(expected: 1)")

print("\n--- 2. Grads accumulate -- you own the reset ---")
p = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device)
(p ** 2).sum().backward()        # dL/dp = 2p -> [2, 4, 6]
print("after backward #1:", p.grad)
(p ** 2).sum().backward()        # no reset: grads ADD -> [4, 8, 12]
print("after backward #2 (no zero):", p.grad)
p.grad.zero_()                   # the reset optimizers do every step
(p ** 2).sum().backward()
print("after zero_ + backward:", p.grad)

print("\n--- 3. Chain rule, the trading way ---")
# Portfolio value from two positions: v = w1*r1 + w2*r2.
# The gradient answers: which weight moves the value most per unit nudge?
r = torch.tensor([0.02, -0.01], device=device)        # today's returns: fixed data
w = torch.tensor([0.6, 0.4], requires_grad=True, device=device)
v = (w * r).sum()                                    # 0.0080
v.backward()                                         # dv/dw = r
print(f"value: {v.item():.4f}")
print("d value / d weights:", w.grad)
print("-> w[0] has twice the pull of w[1], and opposite sign")

print("\n--- 4. Turn the recorder off ---")
x = torch.tensor(2.0, requires_grad=True, device=device)
with torch.no_grad():          # eval / inference: skip the tape entirely
    y = x * 2
print("inside no_grad, y.grad_fn:", y.grad_fn)
z = x.detach() * 2             # detach: one tensor opts out, recorder keeps running
print("detached,        z.grad_fn:", z.grad_fn)
print("x.grad:", x.grad)       # None -- we never called backward here

print("\n--- bonus: the tape is exact ---")
# Finite-difference sanity check: autograd == numeric derivative.
x = torch.tensor(2.0, requires_grad=True, device=device)
w = torch.tensor(3.0, requires_grad=True, device=device)
b = torch.tensor(1.0, requires_grad=True, device=device)
(w * x + b).backward()
eps = 1e-4
fd = ((3.0 * (2.0 + eps) + 1.0) - (3.0 * (2.0 - eps) + 1.0)) / (2 * eps)
print(f"autograd dz/dx: {x.grad.item():.4f}   finite-difference: {fd:.4f}")
