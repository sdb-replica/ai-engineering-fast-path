"""Day 94 - Deep Dive: LoRA Hands-On at Runnable Scale.

Day 53 taught the patch on a hand-set 3 x 2 toy: freeze W,
train dW = B x A, merge. Tonight is the same three moves on
the real model you built on Day 42 - the 4-block char GPT -
fine-tuned to speak in a second voice: your trading notes.
Nothing here needs a download or an API: both corpora are
embedded in this file.

  Part 1  Rebuild Day 42's TinyGPT (classes verbatim) and
          count it from the tensors: 814,643 parameters.
          Day 42's formula card said 618,035 - it priced the
          MLP down-projection (512 -> 128) at C^2 instead of
          4C^2. A LoRA census forces you to enumerate every
          matrix anyway, so tonight every count is a count
          of tensors, not of formulas.
  Part 2  Re-pretrain the base on the Day 42 Shakespeare
          corpus (800 steps). No checkpoint file shipped
          with Day 42's script, so the base is re-made here;
          in a real job you load the checkpoint and skip
          straight to Part 3. Sample the base on a journal
          prompt: it answers in Shakespeare. Of course it
          does - that is all it has ever read.
  Part 3  Bolt Day 53's LoRALinear onto EVERY weight matrix
          in the blocks: all k/q/v and proj in attention,
          both MLP matrices - 60 adapters, r = 4, alpha = 8
          (scaling alpha/r = 2, Day 53's volume knob). Freeze
          everything else. Prove the Day 53 init invariant
          on the full model: B = 0, so adapted logits equal
          base logits to the digit.
  Part 4  Train ONLY the adapters on a 2,774-character
          trading-journal corpus (300 steps). The journal is
          written inside the Shakespeare alphabet on purpose
          - the vocab is Day 42's 51 characters, asserted in
          code - so every journal char has an embedding the
          base already trained. Afterwards, check that every
          frozen tensor is bit-for-bit unchanged.
  Part 5  Sample again on the same prompt, seed and
          temperature: the voice has moved. Then the
          deliverables - the adapter file is 55,296 fp32
          numbers (0.21 MiB), and merging W' = W + (alpha/r)
          B A into a fresh TinyGPT reproduces the adapted
          model's logits to float noise.

EXACT vs REFERENCE, as always (Day 42 rules): every count -
814,643 in the base, 55,296 trainable adapter numbers on
top of it, 60 adapters, byte sizes - is shape arithmetic,
twin-verified
with numpy, identical on any machine. The two init/merge
diffs are exact properties (0 at init; ~6e-8 after merge,
verified in float32 by the twin). The loss arcs and both
samples are REFERENCE numbers: your MacBook's RNG stream
and device differ, so your digits will wiggle and your
samples will not match word for word. What is guaranteed:
the base sample sounds like Shakespeare, the adapted one
sounds like the journal, and both loss arcs fall.

Runs on the ~/ai-lab venv: torch only (free), MPS if
available else CPU, no CUDA, no paid APIs, no downloads.
800 + 300 steps of this size is minutes on MPS, longer on
CPU - drop PRE_STEPS to 400 if impatient; the patch demo
does not care how polished the base is.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {device}")

# ---------- data: two corpora, one alphabet ----------
TEXT = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die -- to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream -- ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause -- there's the respect
That makes calamity of so long life.

Tomorrow, and tomorrow, and tomorrow,
Creeps in this petty pace from day to day,
To the last syllable of recorded time;
And all our yesterdays have lighted fools
The way to dusty death. Out, out, brief candle!
Life's but a walking shadow, a poor player
That struts and frets his hour upon the stage
And then is heard no more. It is a tale
Told by an idiot, full of sound and fury,
Signifying nothing.

Friends, Romans, countrymen, lend me your ears;
I come to bury Caesar, not to praise him.
The evil that men do lives after them;
The good is oft interred with their bones;
So let it be with Caesar. The noble Brutus
Hath told you Caesar was ambitious:
If it were so, it was a grievous fault,
And grievously hath Caesar answer'd it.

But, soft! what light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief,
That thou, her maid, art far more fair than she.

Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date;
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimm'd;
And every fair from fair sometime declines,
By chance or nature's changing course untrimm'd;
But thy eternal summer shall not fade,
Nor lose possession of that fair thou ow'st;
Nor shall death brag thou wander'st in his shade,
When in eternal lines to time thou grow'st:
So long as men can breathe or eyes can see,
So long lives this and this gives life to thee.

Now is the winter of our discontent
Made glorious summer by this sun of York;
And all the clouds that lour'd upon our house
In the deep bosom of the ocean buried.
"""

JOURNAL = """Trade notes -- the wheel book and the watchlist.
Monday. Sold MSTR calls into the morning rip. The premium was fat and the strike sits far above the last swing high. The plan is the usual one -- let it decay, buy back at half, roll only if the tape says so.
Tuesday. HOOD dipped under eighty-five and I added shares, not calls. Cash is a position too. The book stays calm if I do.
Wednesday. Sat on my hands. No trade is a trade. The scanner lit up all day and none of it matched the plan, so none of it matched my money.
Thursday. BMNR ran hard into the close. Trimmed a third into strength. Selling into a green day feels wrong and works anyway -- that is the whole trick of the wheel.
Friday. Flat week, small win. The premium came in, risk stayed flat. Boring is the business model.
Monday. CIFR pulled back to the shelf and held. Sold puts at the shelf with cash held back to take the shares if it breaks. If it breaks, I own a name I wanted at a price I named first.
Tuesday. The whole tape was red and the book barely moved. That is the hedge nobody brags about -- short premium and time.
Wednesday. HOOD broke the low and the notes said add, so I added. Rules over feelings, in that order, every time.
Thursday. MSTR calls bought back at half in nine days. Half of fat premium in nine days beats all of it in thirty. Freed the cash for the next shelf.
Friday. Reviewed the month. Wins came from the plan. Losses came from the two times I chased. Write it down so the next month reads it.
Monday. Bitcoin ripped and BMNR followed an hour later. The lag is the tell. Sold calls into the second rip, not the first.
Tuesday. AMD sat in a range and the range paid rent. Sold strangles small, far out, and let the calendar work.
Wednesday. A red open scared the chat and bored me. The plan has a line for red opens. Followed it. Added nothing, sold nothing, worried less.
Thursday. Rolled the CIFR puts out and down for a credit. Time is the product I sell. The strike moved, the thesis did not.
Friday. Took the HOOD shares off at the target. A seat is not a shrine. Cash back, eyes open, next shelf marked.
Monday. Calm tape, loud feed. Muted the feed, read the book. The book said hold, so I held.
Tuesday. BMNR fell to the trend line and bounced on the hour. Sold puts at the line. If the line fails I own it lower, and the notes say that is fine.
Wednesday. Half on all new sales into earnings. The print is a coin flip with a calendar invite. The wheel does not need coin flips.
Thursday. MSTR broke out and ran. My short calls are a cap, not a curse. If they take the shares, the wheel starts again lower. No anger in the notes.
Friday. Month two of the log. The pattern is plain -- small edges, taken often, and kept small so a bad week is a story and not a scar.
"""

chars = sorted(set(TEXT))          # vocab from Shakespeare alone,
vocab_size = len(chars)            # exactly as Day 42: V = 51
assert set(JOURNAL) <= set(chars), "journal left the alphabet"
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)
data = torch.tensor(encode(TEXT), dtype=torch.long)
jdata = torch.tensor(encode(JOURNAL), dtype=torch.long)

batch_size = 32        # sequences per step
block_size = 64        # context window T (Day 42)
n_embd = 128           # width C (Day 42)
n_head = 4             # heads per block, 32 dims each (Day 42)
n_layer = 4            # blocks deep (Day 42)
PRE_STEPS = 800        # base pretraining (Day 42 used 1200)
FT_STEPS = 300         # adapter training steps
R, ALPHA = 4, 8.0      # LoRA rank and alpha (scaling = 2.0)


def get_batch(src):
    """Random windows into src; y is x shifted by one (Day 42)."""
    ix = torch.randint(len(src) - block_size, (batch_size,))
    x = torch.stack([src[i:i + block_size] for i in ix])
    y = torch.stack([src[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ---------- the model: Day 42's classes, verbatim ----------
class Head(nn.Module):
    """One self-attention head with the causal mask (Day 38)."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size)
        self.query = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        _, T, _ = x.shape
        k = self.key(x)                                # (B, T, hs)
        q = self.query(x)                              # (B, T, hs)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0,
                              float("-inf"))           # no peeking
        wei = F.softmax(wei, dim=-1)
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    """Heads in parallel, outputs concatenated + projected (Day 39)."""

    def __init__(self):
        super().__init__()
        hs = n_embd // n_head
        self.heads = nn.ModuleList([Head(hs) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads],
                                   dim=-1))


class FeedForward(nn.Module):
    """The block's MLP: widen, bend, narrow (Days 21/40)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-norm block: norm -> attend -> add, norm -> mlp -> add
    (Day 40). Shape in = shape out: (B, T, C) throughout."""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention()
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))     # communicate
        x = x + self.ffwd(self.ln2(x))     # compute
        return x


class TinyGPT(nn.Module):
    """The whole stack: embed -> blocks -> norm -> head."""

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding(idx)                    # (B,T,C)
        pos = self.position_embedding(
            torch.arange(T, device=idx.device))            # (T,C)
        x = self.blocks(tok + pos)                         # (B,T,C)
        logits = self.lm_head(self.ln_f(x))                # (B,T,V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, vocab_size),
                                   targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0,
                 greedy=False):
        """Predict, append, repeat - autoregression in one loop.
        Temperature divides the logits before softmax (Day 14):
        lower = safer picks, higher = wilder. Greedy = argmax."""
        self.eval()
        for _ in range(max_new_tokens):
            window = idx[:, -block_size:]        # crop to context
            logits, _ = self(window)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            if greedy:
                nxt = probs.argmax(dim=-1, keepdim=True)
            else:
                nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


# ---------- Part 1: the base, rebuilt and recounted ----------
model = TinyGPT().to(device)
n_total = sum(p.numel() for p in model.parameters())
print("Part 1 - the base, rebuilt and recounted")
print("  TinyGPT from Day 42, identical config: C=128, "
      "4 blocks, 4 heads of 32, T=64")
print(f"  parameters, counted from the tensors: {n_total:,}")
print("  Day 42's formula card said 618,035 - it priced the "
      "MLP down-projection")
print("  (512 -> 128) at C^2; the matrix holds 4C^2 numbers. "
      "Count tensors, not cards.")

# ---------- Part 2: re-pretrain the base ----------
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
print(f"Part 2 - re-pretrain the base on Shakespeare "
      f"({PRE_STEPS} steps, AdamW lr=3e-3)")
model.train()
for step in range(PRE_STEPS):
    xb, yb = get_batch(data)
    _, loss = model(xb, yb)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 200 == 0 or step == PRE_STEPS - 1:
        print(f"  step {step:>4}  loss {loss.item():.4f}")

PROMPT = "Friday. "
probe = jdata[:block_size].unsqueeze(0).to(device)
with torch.no_grad():
    base_logits, _ = model(probe)
torch.manual_seed(0)
start = torch.tensor([encode(PROMPT)], dtype=torch.long,
                     device=device)
print(f"  base sample, prompt {PROMPT!r}, seed 0, "
      "temperature 0.8 (reference):")
print(decode(model.generate(start, 110, temperature=0.8)[0]
             .tolist()))

# ---------- Part 3: bolt on the adapters ----------
class LoRALinear(nn.Module):
    """Day 53's wrapper: a frozen nn.Linear plus a trainable
    low-rank patch. forward(x) = base(x) + (alpha/r) (x A^T) B^T.
    B starts at zero, so the patch contributes exactly nothing
    until training moves it. One change from Day 53, which ran
    CPU-only: A and B are born on the base weight's device."""

    def __init__(self, base, r, alpha=1.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)            # freeze W and b
        d_out, d_in = base.weight.shape
        self.r = r
        self.scaling = alpha / r
        dev = base.weight.device
        self.A = nn.Parameter(torch.zeros(r, d_in, device=dev))
        self.B = nn.Parameter(torch.zeros(d_out, r, device=dev))

    def forward(self, x):
        patch = (x @ self.A.T) @ self.B.T      # squeeze, then lift
        return self.base(x) + self.scaling * patch

    def merged_weight(self):
        """W' = W + scaling * B x A - the patch, folded in."""
        with torch.no_grad():
            return self.base.weight + self.scaling * (self.B @ self.A)


def wrap(lin):
    """Swap one nn.Linear for its LoRA twin: A drawn small and
    random (so the patch CAN learn), B left at exactly zero
    (so at init it HASN'T)."""
    w = LoRALinear(lin, R, ALPHA)
    nn.init.normal_(w.A, std=0.02)
    return w


torch.manual_seed(94)
n_wrapped = 0
for block in model.blocks:
    for head in block.attn.heads:
        head.key = wrap(head.key)
        head.query = wrap(head.query)
        head.value = wrap(head.value)
        n_wrapped += 3
    block.attn.proj = wrap(block.attn.proj)
    block.ffwd.net[0] = wrap(block.ffwd.net[0])
    block.ffwd.net[2] = wrap(block.ffwd.net[2])
    n_wrapped += 3
for name, p in model.named_parameters():
    p.requires_grad_(name.endswith(".A") or name.endswith(".B"))
n_train = sum(p.numel() for p in model.parameters()
              if p.requires_grad)
with torch.no_grad():
    adapted_logits, _ = model(probe)
init_diff = (adapted_logits - base_logits).abs().max().item()
print(f"Part 3 - bolt on the adapters (LoRA r={R}, "
      f"alpha={ALPHA:.0f}, scaling {ALPHA / R:.1f})")
print(f"  wrapped linears: {n_wrapped} - every k/q/v, proj "
      "and MLP matrix in all 4 blocks")
print(f"  trainable (adapters only): {n_train:,} new numbers "
      f"= {100 * n_train / n_total:.2f}% of the base count")
print(f"  frozen: {n_total:,} - every base tensor: embeddings, "
      "norms, LM head, all matrices")
print(f"  adapters at init: max |delta logits| vs base "
      f"= {init_diff:.6f}  (B = 0 -> exact)")

# ---------- Part 4: train ONLY the adapters ----------
frozen_before = {k: v.clone() for k, v in
                 model.state_dict().items()
                 if not k.endswith((".A", ".B"))}
adapter_params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(adapter_params, lr=1e-3)
print(f"Part 4 - train ONLY the adapters on the journal "
      f"({FT_STEPS} steps, AdamW lr=1e-3)")
print(f"  journal corpus: {len(JOURNAL):,} characters, same "
      f"{vocab_size}-char alphabet (asserted above)")
model.train()
for step in range(FT_STEPS):
    xb, yb = get_batch(jdata)
    _, loss = model(xb, yb)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0 or step == FT_STEPS - 1:
        print(f"  step {step:>4}  loss {loss.item():.4f}")
still = all(torch.equal(v, frozen_before[k]) for k, v in
            model.state_dict().items() if k in frozen_before)
print(f"  frozen tensors bit-for-bit unchanged: {still}")

# ---------- Part 5: sample, save the patch, merge it ----------
torch.manual_seed(0)
start = torch.tensor([encode(PROMPT)], dtype=torch.long,
                     device=device)
print("Part 5 - sample, save the patch, merge it")
print(f"  adapted sample, same prompt/seed/temperature "
      "(reference):")
print(decode(model.generate(start, 110, temperature=0.8)[0]
             .tolist()))
print(f"  the deliverable: one adapter file, {n_train:,} fp32 "
      f"numbers = {n_train * 4 / 2**20:.2f} MiB")
print(f"  (full fine-tune ships {n_total:,} numbers and a "
      f"{n_total * 16 / 2**20:.2f} MiB training state)")


def get_module(root, path):
    obj = root
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


lora_paths = [path for path, m in model.named_modules()
              if isinstance(m, LoRALinear)]
new_sd = {}
for k, v in model.state_dict().items():
    if any(k == p + suf for p in lora_paths
           for suf in (".A", ".B", ".base.weight", ".base.bias")):
        continue                     # adapter keys: replaced below
    new_sd[k] = v
for p in lora_paths:
    mod = get_module(model, p)
    new_sd[p + ".weight"] = mod.merged_weight()
    new_sd[p + ".bias"] = mod.base.bias
merged = TinyGPT().to(device)
merged.load_state_dict(new_sd)       # strict: keys+shapes must fit
with torch.no_grad():
    merged_logits, _ = merged(probe)
    adapted_logits, _ = model(probe)
mdiff = (merged_logits - adapted_logits).abs().max().item()
print(f"  merged model: fresh TinyGPT, W' = W + "
      f"{ALPHA / R:.1f} * B A folded into all {len(lora_paths)} "
      "slots")
print(f"  merged vs adapted logits on a journal window: "
      f"max diff {mdiff:.6f}")
print("  (a hair in the last digits is float rounding, not "
      "math - the twin puts it at ~6e-8)")
print("  verdict: one frozen base, one tiny patch per voice "
      "- merge before shipping")
