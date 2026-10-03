# AI Engineering Fast Path

Sai's 100-day journey from distributed systems to AI engineering — one byte-size
lesson per day, about an hour each evening. Theory bite, visuals, runnable code,
one hands-on win.

Each day has its own folder:

- `lesson.html` — the visual lesson (diagrams, analogies, code with expected
  output). Open it in any browser; it works offline.
- `lesson.md` — the same lesson as Markdown (renders natively here on GitHub).
- `diagram.svg` — the day's diagram as a standalone image.
- `code.py` — the day's runnable code. Run it with the day's environment.

## Setup (once)

```bash
python3 -m venv ~/ai-lab
source ~/ai-lab/bin/activate
pip install --upgrade pip
pip install torch numpy
```

## Lessons

| Day | Lesson (visual) | Lesson (markdown) | Code |
|-----|-----------------|-------------------|------|
| 1 | [Build Your AI Lab](day-001-build-your-ai-lab/lesson.html) | [lesson.md](day-001-build-your-ai-lab/lesson.md) | [code.py](day-001-build-your-ai-lab/code.py) |
| 2 | [Tensors: The Universal Container](day-002-tensors-the-universal-container/lesson.html) | [lesson.md](day-002-tensors-the-universal-container/lesson.md) | [code.py](day-002-tensors-the-universal-container/code.py) |
| 3 | [Tensor Ops & Broadcasting](day-003-tensor-ops-broadcasting/lesson.html) | [lesson.md](day-003-tensor-ops-broadcasting/lesson.md) | [code.py](day-003-tensor-ops-broadcasting/code.py) |
| 4 | [Autograd: The Tape Recorder](day-004-autograd-the-tape-recorder/lesson.html) | [lesson.md](day-004-autograd-the-tape-recorder/lesson.md) | [code.py](day-004-autograd-the-tape-recorder/code.py) |
| 5 | [Loss Functions: How a Model Knows It's Wrong](day-005-loss-functions/lesson.html) | [lesson.md](day-005-loss-functions/lesson.md) | [code.py](day-005-loss-functions/code.py) |
| 6 | [Gradient Descent From Scratch](day-006-gradient-descent-from-scratch/lesson.html) | [lesson.md](day-006-gradient-descent-from-scratch/lesson.md) | [code.py](day-006-gradient-descent-from-scratch/code.py) |
| 7 | [The Training Loop Anatomy](day-007-training-loop-anatomy/lesson.html) | [lesson.md](day-007-training-loop-anatomy/lesson.md) | [code.py](day-007-training-loop-anatomy/code.py) |
