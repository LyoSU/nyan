"""Full fine-tune of an encoder on Jev's soft category labels; judged each epoch on the blind 500.

The epoch is chosen on the dev half; the test half is only read.
Usage: ft.py <name> <hf repo> <prefix> [epochs] [lr]
Env: BS (micro-batch, 8), ACCUM (2), MAXLEN (256), MAX_STEPS (0 = all; a smoke test otherwise).
Run with PYTORCH_MPS_HIGH_WATERMARK_RATIO set, so running out of GPU memory raises
instead of taking the whole machine down with it.
"""
import os as _os
# The checkout that holds data/ and models/: this one, unless NYAN_ROOT names another.
ROOT = _os.environ.get("NYAN_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import math, os, random, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/scripts")
from eval_jev import gold_labels, in_split, is_right, read_jsonl

name, repo, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 4
lr = float(sys.argv[5]) if len(sys.argv) > 5 else 3e-5
BS, ACCUM = int(os.environ.get("BS", 8)), int(os.environ.get("ACCUM", 2))
MAXLEN, MAX_STEPS = int(os.environ.get("MAXLEN", 256)), int(os.environ.get("MAX_STEPS", 0))
D = Path(f"{ROOT}/data/distill"); E = Path(f"{ROOT}/data/jev_eval")
CATS = ["not_news", "war", "politics", "economy", "incident", "other", "tech", "sports", "entertainment", "science"]
torch.manual_seed(0); random.seed(0)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

teach = [r for f in os.environ.get("TEACH", "teach_v2.jsonl").split(",") for r in read_jsonl(D / f)]
X = [prefix + r["text"][:1500] for r in teach]
Y = torch.tensor([[r["answers"]["category"]["probabilities"].get(c, 0.0) for c in CATS] for r in teach])
Y = Y / Y.sum(1, keepdim=True)
sample = read_jsonl(E / "sample.jsonl"); gold = gold_labels(E)
G = [prefix + p["text"][:1500] for p in sample]
split = np.array([in_split(p["url"], "test") for p in sample])

tok = AutoTokenizer.from_pretrained(repo)
enc = AutoModel.from_pretrained(repo).to(dev)
# The token table is most of a multilingual encoder (250k rows in e5) and AdamW
# keeps two more copies of whatever trains: frozen, it costs a third of the memory.
enc.get_input_embeddings().weight.requires_grad_(False)
trainable = [p for p in enc.parameters() if p.requires_grad]
print(f"trainable {sum(p.numel() for p in trainable)/1e6:.0f}M of {sum(p.numel() for p in enc.parameters())/1e6:.0f}M", flush=True)
head = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(enc.config.hidden_size, len(CATS))).to(dev)

def embed(texts):
    b = tok(texts, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(dev)
    h = enc(**b).last_hidden_state
    m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1)

steps = epochs * math.ceil(len(X) / (BS * ACCUM))
opt = torch.optim.AdamW([{"params": trainable, "lr": lr}, {"params": head.parameters(), "lr": 1e-3}], weight_decay=0.01, foreach=False)
warm = max(1, int(0.06 * steps))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / max(1, steps - warm)))

def evaluate():
    enc.eval(); head.eval(); probs = [None] * len(G)
    with torch.no_grad():
        order = sorted(range(len(G)), key=lambda i: len(G[i]))
        for k in range(0, len(order), 32):
            idx = order[k:k + 32]
            for i, row in zip(idx, F.softmax(head(embed([G[i] for i in idx])), -1).cpu().numpy()):
                probs[i] = row
    enc.train(); head.train()
    probs = np.stack(probs); pred = [CATS[i] for i in probs.argmax(1)]
    ok = np.array([is_right(pred[i], gold[sample[i]["url"]]) for i in range(len(G))])
    return probs, 100 * ok[~split].mean(), 100 * ok[split].mean(), 100 * ok.mean()

started = time.time(); best = -1.0; step = 0
for ep in range(epochs):
    perm = list(range(len(X))); random.shuffle(perm)
    for k in range(0, len(perm), BS):
        idx = perm[k:k + BS]
        loss = -(Y[idx].to(dev) * F.log_softmax(head(embed([X[i] for i in idx])), -1)).sum(-1).mean() / ACCUM
        loss.backward()
        if (k // BS + 1) % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); sched.step(); opt.zero_grad(); step += 1
            if step % 25 == 0:
                print(f"  step {step}/{steps} loss {loss.item() * ACCUM:.3f} ({time.time()-started:.0f}s, mps {(torch.mps.current_allocated_memory() if torch.backends.mps.is_available() else 0)/2**30:.1f} GB)", flush=True)
            if MAX_STEPS and step >= MAX_STEPS:
                print("smoke test done", flush=True); sys.exit(0)
    probs, d, t, a = evaluate()
    print(f"{name} epoch {ep+1}: dev {d:.1f}% test {t:.1f}% all {a:.1f}% ({time.time()-started:.0f}s)", flush=True)
    if d > best:
        best = d; np.save(D / f"ft_{name}_gold_probs.npy", probs)
print(f"{name} done, best dev {best:.1f}%", flush=True)
