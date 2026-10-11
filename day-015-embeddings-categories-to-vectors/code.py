"""Day 15 - Embeddings: Categories to Vectors.

Neurons only eat numbers, so a category (a word, a ticker, a regime) has to
become a vector. One-hot does it - but sparsely, and every category ends up
equidistant from every other (no meaning). An embedding is a LEARNED lookup
table: each category id maps to a dense vector, and the vectors are trained by
gradient descent so that similar things land near each other.

Runs on the ~/ai-lab venv: torch CPU or MPS, no paid APIs.
Expected output is shown in the lesson HTML - numbers below should match to the
printed precision.
"""
import torch
import torch.nn.functional as F

torch.set_printoptions(precision=4)

vocab = ["bull", "bear", "chop", "spike", "crash", "rally"]
V, D = len(vocab), 4          # 6 categories, 4-dim vectors

print("=== 1. The one-hot problem: categories with no meaning ===")
idx = torch.tensor([0, 5])                       # bull, rally
oh = F.one_hot(idx, num_classes=V).float()
print("one-hot 'bull' :", oh[0].tolist())
print("one-hot 'rally':", oh[1].tolist())
print(f"dot(bull, rally) = {float(oh[0] @ oh[1]):.1f}   "
      f"dot(bull, bear) = {float(oh[0] @ F.one_hot(torch.tensor(1), V).float()):.1f}")
print("every category is exactly as far from every other. No meaning fits.")

print("\n=== 2. The fix: a lookup table of learned vectors ===")
W = torch.tensor([                               # the embedding TABLE (6 x 4)
    [ 0.90,  0.10, -0.30,  0.60],   # bull
    [-0.80,  0.20,  0.50, -0.40],   # bear
    [ 0.05, -0.05,  0.10, -0.05],   # chop
    [ 0.70,  0.60, -0.10,  0.20],   # spike
    [-0.60, -0.70,  0.30, -0.50],   # crash
    [ 0.85, -0.10, -0.25,  0.55]])  # rally
emb = torch.nn.Embedding.from_pretrained(W)      # a Parameter, not data
vecs = emb(idx)
print("emb([0, 5]) =")
for i, w in enumerate(vocab):
    if i in (0, 5):
        print(f"  {w:>6}:", [f"{v:.4f}" for v in emb.weight[i].tolist()])

print("\n=== 3. Lookup == one-hot @ W, minus the waste ===")
matmul_way = oh @ W                              # dense multiply: O(V * D)
gather_way = emb(idx)                            # row fetch: O(1) gather
print("max |one-hot@W - lookup| =", float((matmul_way - gather_way).abs().max()))
print("torch never does the matmul - it gathers the row, like a memtable read.")

print("\n=== 4. Meaning is geometry: cosine similarity ===")
def sim(a, b):
    return float(F.cosine_similarity(emb.weight[a].unsqueeze(0),
                                     emb.weight[b].unsqueeze(0)))
pairs = [("bull", "rally", sim(0, 5)), ("bull", "crash", sim(0, 4)),
         ("bull", "bear", sim(0, 1)), ("chop", "bear", sim(2, 1))]
for a, b, s in pairs:
    print(f"  cos({a:>5}, {b:>5}) = {s:+.4f}")
print("synonyms point the same way; opposites point away; chop points nowhere.")

print("\n=== 5. The table LEARNS: pull 'bull' toward 'rally' ===")
print(f"before: cos(bull, rally) = {sim(0, 5):+.4f}")
loss = (emb(torch.tensor([0])) - emb(torch.tensor([5]))).pow(2).sum()
loss.backward()
touched = torch.where(emb.weight.grad.abs().sum(dim=1) > 0)[0].tolist()
print("rows the gradient touched:", touched, "(only the looked-up rows)")
with torch.no_grad():                            # one manual SGD step, lr = 0.1
    emb.weight -= 0.1 * emb.weight.grad
print(f"after : cos(bull, rally) = {sim(0, 5):+.4f}")
print("Day 7's training loop writes into this table. That is how meaning forms.")

print("\nWIN: you turned categories into geometry - and trained the geometry.")
