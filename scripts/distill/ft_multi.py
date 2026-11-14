"""Fine-tune one encoder with a head per field, on the teacher's multi-field labels.

Category is judged on the blind 500; the other heads against the teacher's own
answers on those 500 posts (there is no blind gold for them). The epoch is chosen
on the dev half's category accuracy; the test half is only read.
Usage: ft_multi.py <name> <hf repo> <prefix> [epochs] [lr]
Env: TEACH (teach_multi_gpt-5.6-luna.jsonl), GOLD (gold_multi_gpt-5.6-luna.jsonl), BS 8, ACCUM 2, MAXLEN 256, LIMIT (0 = all).
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

name, repo, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 3
lr = float(sys.argv[5]) if len(sys.argv) > 5 else 3e-5
BS, ACCUM = int(os.environ.get("BS", 8)), int(os.environ.get("ACCUM", 2))
MAXLEN, LIMIT = int(os.environ.get("MAXLEN", 256)), int(os.environ.get("LIMIT", 0))
D = Path(f"{ROOT}/data/distill"); E = Path(f"{ROOT}/data/jev_eval")
torch.manual_seed(0); random.seed(0)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

CHOICES = {"topic": TOPIC_KEYS, "scope": SCOPES, "region": REGIONS, "role": ROLES, "significance": [1, 2, 3, 4, 5]}
BINARY = ["about_ukraine", "urgent", "clickbait"]
URGENT_FACTS = ("danger_now", "effect_24h", "breaking")
WEIGHT = {"category": 1.0, "topic": 0.5, "scope": 0.3, "region": 0.3, "role": 0.2, "significance": 0.3,
          "about_ukraine": 0.2, "urgent": 0.2, "clickbait": 0.2}

def targets(a: dict) -> dict:
    """The teacher's answer as tensors; -1 marks a field it left out or garbled."""
    p = [0.0] * len(CATS)
    conf = min(1.0, max(0.0, float(a.get("confidence", 1.0))))
    p[CATS.index(a["category"])] += conf
    second = a.get("second")
    if second in CATS and second != a["category"]:
        p[CATS.index(second)] += 1 - conf
    else:
        p = [x + (1 - conf) / len(CATS) for x in p]
    t = {"category": p}
    for f, opts in CHOICES.items():
        v = a.get(f)
        if f == "significance":
            try: v = int(round(float(v)))
            except (TypeError, ValueError): v = None
        t[f] = opts.index(v) if v in opts else -1
    if "urgent" not in a and any(k in a for k in URGENT_FACTS):
        # The v3 prompts ask three facts instead of urgency itself; without this
        # every v3 post read as "urgent left out" and the head never trained.
        a = {**a, "urgent": max(float(a.get(k) or 0) for k in URGENT_FACTS)}
    for f in BINARY:
        try: t[f] = min(1.0, max(0.0, float(a.get(f))))
        except (TypeError, ValueError): t[f] = -1.0
    return t

teach = read_jsonl(D / os.environ.get("TEACH", "teach_multi_gpt-5.6-luna.jsonl"))
random.shuffle(teach)
if LIMIT: teach = teach[:LIMIT]
X = [prefix + r["text"][:1500] for r in teach]
T = [targets(r["labels"]) for r in teach]
sample = read_jsonl(E / "sample.jsonl"); gold = gold_labels(E)
G = [prefix + p["text"][:1500] for p in sample]
split = np.array([in_split(p["url"], "test") for p in sample])
# The teacher's answers on the blind 500 under the same prompt as TEACH, or
# the heads are scored against a different definition of their own fields.
teacher_gold = {r["url"]: r for r in read_jsonl(D / os.environ.get("GOLD", "gold_multi_gpt-5.6-luna.jsonl"))}
TG = [targets(teacher_gold[p["url"]]) if teacher_gold.get(p["url"], {}).get("category") else None for p in sample]

tok = AutoTokenizer.from_pretrained(repo)
enc = AutoModel.from_pretrained(repo).to(dev)
enc.get_input_embeddings().weight.requires_grad_(False)  # most of the model, and three copies in AdamW
trainable = [p for p in enc.parameters() if p.requires_grad]
H = enc.config.hidden_size
sizes = {"category": len(CATS), **{f: len(o) for f, o in CHOICES.items()}, **{f: 1 for f in BINARY}}
heads = torch.nn.ModuleDict({f: torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(H, n)) for f, n in sizes.items()}).to(dev)
print(f"{len(teach)} posts; trainable {sum(p.numel() for p in trainable)/1e6:.0f}M", flush=True)

def embed(texts):
    b = tok(texts, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(dev)
    h = enc(**b).last_hidden_state
    m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1)

def loss_of(z, batch):
    total = 0.0
    y = torch.tensor([t["category"] for t in batch], device=dev)
    total = total + WEIGHT["category"] * -(y * F.log_softmax(heads["category"](z), -1)).sum(-1).mean()
    for f in CHOICES:
        y = torch.tensor([t[f] for t in batch], device=dev)
        if (y >= 0).any():
            total = total + WEIGHT[f] * F.cross_entropy(heads[f](z), y, ignore_index=-1)
    for f in BINARY:
        y = torch.tensor([t[f] for t in batch], device=dev); keep = y >= 0
        if keep.any():
            total = total + WEIGHT[f] * F.binary_cross_entropy_with_logits(heads[f](z).squeeze(-1)[keep], y[keep])
    return total

steps = epochs * math.ceil(len(X) / (BS * ACCUM))
opt = torch.optim.AdamW([{"params": trainable, "lr": lr}, {"params": heads.parameters(), "lr": 1e-3}], weight_decay=0.01, foreach=False)
warm = max(1, int(0.06 * steps))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / max(1, steps - warm)))

def evaluate():
    enc.eval(); heads.eval(); out = {f: [None] * len(G) for f in sizes}
    with torch.no_grad():
        order = sorted(range(len(G)), key=lambda i: len(G[i]))
        for k in range(0, len(order), 32):
            idx = order[k:k + 32]; z = embed([G[i] for i in idx])
            for f in sizes:
                v = heads[f](z)
                v = (torch.sigmoid(v.squeeze(-1)) if f in BINARY else F.softmax(v, -1)).cpu().numpy()
                for i, row in zip(idx, v): out[f][i] = row
    enc.train(); heads.train()
    pred = [CATS[int(np.argmax(p))] for p in out["category"]]
    ok = np.array([is_right(pred[i], gold[sample[i]["url"]]) for i in range(len(G))])
    news = [i for i in range(len(G)) if "not_news" not in (gold[sample[i]["url"]]["category"], gold[sample[i]["url"]].get("also_ok"))]
    line = f"category dev {100*ok[~split].mean():.1f}% test {100*ok[split].mean():.1f}%, news dropped {sum(pred[i]=='not_news' for i in news)}/{len(news)}"
    tests = [i for i in range(len(G)) if split[i] and TG[i]]
    agree = []
    for f in CHOICES:
        hits = [int(np.argmax(out[f][i])) == TG[i][f] for i in tests if TG[i][f] >= 0]
        agree.append(f"{f} {100*np.mean(hits):.0f}%")
    sig = [(float(np.dot(out["significance"][i], [1, 2, 3, 4, 5])), TG[i]["significance"] + 1) for i in tests if TG[i]["significance"] >= 0]
    a, b = map(np.array, zip(*sig))
    agree.append(f"significance MAE {np.abs(a-b).mean():.2f}")
    for f in BINARY:
        xs = [(float(out[f][i]), TG[i][f]) for i in tests if TG[i][f] >= 0]
        a, b = map(np.array, zip(*xs)); agree.append(f"{f} r={np.corrcoef(a, b)[0, 1]:.2f}")
    return out, 100 * ok[~split].mean(), line + "\n    vs teacher (test half): " + ", ".join(agree)

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
                print(f"  step {step}/{steps} loss {loss.item()*ACCUM:.3f} ({time.time()-started:.0f}s, mps {(torch.mps.current_allocated_memory() if torch.backends.mps.is_available() else 0)/2**30:.1f} GB)", flush=True)
    out, d, line = evaluate()
    print(f"{name} epoch {ep+1} ({time.time()-started:.0f}s): {line}", flush=True)
    state = {"encoder": enc.state_dict(), "heads": heads.state_dict(), "repo": repo, "prefix": prefix}
    # Every epoch too: the dev half is 244 posts, one post moves it 0.4 points,
    # so the pick by dev alone is worth checking against the others afterwards.
    torch.save(state, D / f"model_{name}_e{ep+1}.pt")
    if d > best:
        best = d
        torch.save(state, D / f"model_{name}.pt")
print(f"{name} done, best dev {best:.1f}%", flush=True)
