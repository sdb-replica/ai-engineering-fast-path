# Day 1 of 100 — Build Your AI Lab

**Tonight you'll be able to:** run PyTorch on your Mac and prove your GPU works.

## The one idea

Every model you'll ever touch — from a tiny classifier to a frontier LLM — is just
**tensors** (multi-dimensional arrays) plus math on tensors. PyTorch is the workshop
where all of it happens.

Tonight isn't glamorous: we're provisioning the lab. Every lesson after this assumes
it's ready. Ninety-nine lessons of compounding start with one boring, load-bearing setup.

> **Analogy:** This is `terraform apply` for your learning environment. Nobody brags
> about it — nothing ships without it.

## See it

![Matrix multiply: two [2,2] tensors in, one [2,2] tensor out](diagram.svg)

Matrix multiply (`@`): two [2,2] tensors in, one [2,2] tensor out. This single
operation powers every neural network. (Full-color version in `lesson.html`.)

## Code it (~30 min)

On your MacBook, in Terminal. **Step 1 — create the lab:**

```bash
python3 -m venv ~/ai-lab
source ~/ai-lab/bin/activate
pip install --upgrade pip
pip install torch numpy
```

Plain `pip install torch` covers Apple Silicon (MPS GPU) and Intel Macs (CPU fallback).
No CUDA needed — ever — for this challenge.

**Step 2 — meet your first tensor.** Still in the venv, run `python3` and type
(or just run `code.py`):

```python
import torch
print(torch.__version__)
print("MPS (Apple GPU) available:", torch.backends.mps.is_available())

device = "mps" if torch.backends.mps.is_available() else "cpu"
a = torch.tensor([[1., 2.], [3., 4.]], device=device)
b = torch.tensor([[5., 6.], [7., 8.]], device=device)
print(a + b)
print(a @ b)  # @ = matrix multiply
```

**Expected output:**

```
2.x.x
MPS (Apple GPU) available: True
tensor([[ 6.,  8.],
        [10., 12.]])
tensor([[19., 22.],
        [43., 50.]])
```

## Today's win

Get the above running and see those two matrices print. You now have a working AI
lab — and you've executed the operation at the heart of all deep learning.

## Short on time? 20-minute version

Just the install plus the MPS check printing `True`. The tensor math can wait for
tomorrow.

## Tomorrow's teaser

Tensors are more than arrays — what *shape* actually means, and why getting it wrong
is 90% of beginner bugs.

---
*Streak: 1 day · Never miss twice.*
