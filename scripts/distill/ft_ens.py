"""Retrain the production student on labels from several judges and on fresh posts.

`ft_multi.py` learnt from one teacher and one week. This one starts from the
model in production and trains on what `build_ens.py` wrote: the old posts with
the decisions made since applied to them, and a later week of production posts
labelled by three other models where the student was unsure, by the student
itself where it was sure, and by the rubric patterns where they match.

The epoch is chosen on fresh posts kept for that (`dev`), and three sets are
only read: fresh posts the judges labelled that nothing trained on (`test`),
and the two halves of the blind 500. Each is printed before the first step too,
so the model being replaced is measured by the same code as its replacement.

Usage: ft_ens.py <name> [epochs] [lr]
Env: INIT (e5_multi_h3c), TRAIN (teach_ens.jsonl), BS 8, ACCUM 2, MAXLEN 256, LIMIT (0 = all).
"""
import os as _os
# The checkout that holds data/ and models/: this one, unless NYAN_ROOT names another.
ROOT = _os.environ.get("NYAN_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json, math, os, random, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, str(Path(__file__).parent))
from eval_jev import gold_labels, in_split, is_right, read_jsonl
from multi_label import CATS, TOPIC_KEYS, REGIONS, SCOPES, ROLES

name = sys.argv[1]
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
lr = float(sys.argv[3]) if len(sys.argv) > 3 else 2e-5
BS, ACCUM = int(os.environ.get("BS", 8)), int(os.environ.get("ACCUM", 2))
MAXLEN, LIMIT = int(os.environ.get("MAXLEN", 256)), int(os.environ.get("LIMIT", 0))
D = Path(f"{ROOT}/data/distill"); E = Path(f"{ROOT}/data/jev_eval")
torch.manual_seed(0); random.seed(0)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

CHOICES = {"topic": TOPIC_KEYS, "scope": SCOPES, "region": REGIONS, "role": ROLES, "significance": [1, 2, 3, 4, 5]}
BINARY = ["about_ukraine", "urgent", "clickbait"]
WEIGHT = {"category": 1.0, "topic": 0.5, "scope": 0.3, "region": 0.3, "role": 0.2, "significance": 0.3,
          "about_ukraine": 0.2, "urgent": 0.2, "clickbait": 0.2}

def targets(a: dict) -> dict:
    """One row of build_ens.py as tensors; -1 marks a field that is not known."""
    t = {"category": [float(a["category_p"].get(c, 0.0)) for c in CATS]}
    for f, opts in CHOICES.items():
        v = a.get(f)
        t[f] = opts.index(v) if v in opts else -1
    for f in BINARY:
        try: t[f] = min(1.0, max(0.0, float(a.get(f))))
        except (TypeError, ValueError): t[f] = -1.0
    return t

ck = torch.load(D / f"model_{os.environ.get('INIT', 'e5_multi_h3c')}.pt", map_location="cpu")
repo, prefix = ck["repo"], ck["prefix"]
teach = read_jsonl(D / os.environ.get("TRAIN", "teach_ens.jsonl"))
random.shuffle(teach)
if LIMIT: teach = teach[:LIMIT]
X = [prefix + r["text"][:1500] for r in teach]
T = [targets(r["labels"]) for r in teach]

# What is read and never trained on. `ok` is every category that counts as right.
READ = {}
for part in ("dev", "test"):
    rows = json.load(open(D / f"ens/{part}_labels.json"))
    READ[part] = ([prefix + r["text"][:1500] for r in rows], [set(r["ok"]) for r in rows], [r["category"] for r in rows])
sample = read_jsonl(E / "sample.jsonl"); gold = gold_labels(E)
for half in ("dev", "test"):
    rows = [p for p in sample if in_split(p["url"], "test") == (half == "test")]
    g = [gold[p["url"]] for p in rows]
    READ[f"blind_{half}"] = ([prefix + p["text"][:1500] for p in rows],
                             [{x["category"], x.get("also_ok")} - {None} for x in g], [x["category"] for x in g])

tok = AutoTokenizer.from_pretrained(repo)
enc = AutoModel.from_pretrained(repo); enc.load_state_dict(ck["encoder"]); enc.to(dev)
enc.get_input_embeddings().weight.requires_grad_(False)  # most of the model, and three copies in AdamW
trainable = [p for p in enc.parameters() if p.requires_grad]
H = enc.config.hidden_size
sizes = {"category": len(CATS), **{f: len(o) for f, o in CHOICES.items()}, **{f: 1 for f in BINARY}}
heads = torch.nn.ModuleDict({f: torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(H, n)) for f, n in sizes.items()})
heads.load_state_dict(ck["heads"]); heads.to(dev)
print(f"{len(teach)} posts; trainable {sum(p.numel() for p in trainable)/1e6:.0f}M", flush=True)

def embed(texts):
    b = tok(texts, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(dev)
    h = enc(**b).last_hidden_state
    m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1)

def loss_of(z, batch):
    y = torch.tensor([t["category"] for t in batch], device=dev)
    total = WEIGHT["category"] * -(y * F.log_softmax(heads["category"](z), -1)).sum(-1).mean()
    for f in CHOICES:
        y = torch.tensor([t[f] for t in batch], device=dev)
        if (y >= 0).any():
            total = total + WEIGHT[f] * F.cross_entropy(heads[f](z), y, ignore_index=-1)
    for f in BINARY:
        y = torch.tensor([t[f] for t in batch], device=dev); keep = y >= 0
        if keep.any():
            total = total + WEIGHT[f] * F.binary_cross_entropy_with_logits(heads[f](z).squeeze(-1)[keep], y[keep])
    return total

def read(texts):
    out = [None] * len(texts)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    for k in range(0, len(order), 32):
        idx = order[k:k + 32]
        p = F.softmax(heads["category"](embed([texts[i] for i in idx])), -1).cpu().numpy()
        for i, row in zip(idx, p): out[i] = row
    return np.stack(out)

def evaluate():
    enc.eval(); heads.eval(); lines = []; score = 0.0; probs = {}
    with torch.no_grad():
        for part, (texts, ok, first) in READ.items():
            p = read(texts); probs[part] = p
            pred = [CATS[i] for i in p.argmax(1)]
            right = np.mean([c in o for c, o in zip(pred, ok)]); strict = np.mean([c == f for c, f in zip(pred, first)])
            news = [i for i, o in enumerate(ok) if "not_news" not in o]; rest = [i for i, o in enumerate(ok) if o == {"not_news"}]
            lines.append(f"{part} {100*right:.1f}% (strict {100*strict:.1f}%, news dropped {sum(pred[i]=='not_news' for i in news)}/{len(news)}, "
                         f"not_news let through {sum(pred[i]!='not_news' for i in rest)}/{len(rest)})")
            if part == "dev": score = 100 * strict
    enc.train(); heads.train()
    return probs, score, "\n    ".join(lines)

probs, before, line = evaluate()
np.savez(D / f"probs_{name}_e0.npz", **probs)
print(f"{name} before training: {line}", flush=True)

steps = epochs * math.ceil(len(X) / (BS * ACCUM))
opt = torch.optim.AdamW([{"params": trainable, "lr": lr}, {"params": heads.parameters(), "lr": 1e-4}], weight_decay=0.01, foreach=False)
warm = max(1, int(0.06 * steps))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / max(1, steps - warm)))

started = time.time(); best = -1.0; step = 0
for ep in range(epochs):
    perm = list(range(len(X))); random.shuffle(perm)
    for k in range(0, len(perm), BS):
        idx = perm[k:k + BS]
        loss = loss_of(embed([X[i] for i in idx]), [T[i] for i in idx]) / ACCUM
        loss.backward()
        if (k // BS + 1) % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); sched.step(); opt.zero_grad(); step += 1
            if step % 100 == 0:
                print(f"  step {step}/{steps} loss {loss.item()*ACCUM:.3f} ({time.time()-started:.0f}s)", flush=True)
    probs, d, line = evaluate()
    np.savez(D / f"probs_{name}_e{ep+1}.npz", **probs)
    print(f"{name} epoch {ep+1} ({time.time()-started:.0f}s): {line}", flush=True)
    state = {"encoder": enc.state_dict(), "heads": heads.state_dict(), "repo": repo, "prefix": prefix}
    torch.save(state, D / f"model_{name}_e{ep+1}.pt")
    if d > best:
        best = d
        torch.save(state, D / f"model_{name}.pt")
print(f"{name} done, dev {before:.1f}% before, best {best:.1f}%", flush=True)
