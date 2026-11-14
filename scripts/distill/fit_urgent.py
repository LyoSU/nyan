"""Fit the urgent head of a trained multi-head student on its frozen encoder.

The v3 prompts ask three facts instead of urgency (danger_now, effect_24h,
breaking), and ft_multi.py read every v3 post as "urgent left out", so that
head kept its random start. It is a single linear layer on the pooled
embedding, so it is fitted here on the finished encoder's embeddings of the
teacher's posts, with urgent = the largest of the three facts, and written
back into the checkpoint. Nothing else in it changes.

Usage: fit_urgent.py <name> [posts]   (reads and rewrites model_<name>.pt; keeps a .bak)
Env: TEACH (teach_multi_gpt-5.6-luna_v3b.jsonl), GOLD (gold_multi_gpt-5.6-luna_v3b.jsonl)
"""
import os as _os
# The checkout that holds data/ and models/: this one, unless NYAN_ROOT names another.
ROOT = _os.environ.get("NYAN_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json, os, random, shutil, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/scripts")
from eval_jev import in_split, read_jsonl

name = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 16000
D = Path(f"{ROOT}/data/distill"); E = Path(f"{ROOT}/data/jev_eval")
FACTS = ("danger_now", "effect_24h", "breaking")
MAXLEN = 256
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
random.seed(0); torch.manual_seed(0)

def urgent(a: dict) -> float | None:
    if "urgent" in a:
        return float(a["urgent"])
    if any(k in a for k in FACTS):
        return max(float(a.get(k) or 0) for k in FACTS)
    return None

path = D / f"model_{name}.pt"
ck = torch.load(path, map_location="cpu")
repo, prefix = ck["repo"], ck["prefix"]
tok = AutoTokenizer.from_pretrained(repo)
enc = AutoModel.from_pretrained(repo); enc.load_state_dict(ck["encoder"]); enc.to(dev).eval()

@torch.no_grad()
def embed(texts: list[str]) -> np.ndarray:
    order =sorted(range(len(texts)), key=lambda i: len(texts[i]))
    res = [None] * len(texts)
    for k in range(0, len(order), 32):
        idx = order[k:k + 32]
        b = tok([prefix + texts[i][:1500] for i in idx], padding=True, truncation=True,
                max_length=MAXLEN, return_tensors="pt").to(dev)
        h = enc(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        z = ((h * m).sum(1) / m.sum(1)).float().cpu().numpy()
        for j, i in enumerate(idx): res[i] = z[j]
    return np.stack(res)

teach = [r for r in read_jsonl(D / os.environ.get("TEACH", "teach_multi_gpt-5.6-luna_v3b.jsonl"))
         if urgent(r["labels"]) is not None]
random.shuffle(teach); teach = teach[:N]
t0 = time.time()
X = torch.from_numpy(embed([r["text"] for r in teach]))
y = torch.tensor([min(1.0, max(0.0, urgent(r["labels"]))) for r in teach])
print(f"{len(teach)} posts embedded in {time.time()-t0:.0f}s, urgent share {y.mean():.3f}", flush=True)

# 90/10 split to pick the epoch count; then the head is the linear layer itself.
cut = int(0.9 * len(X))
key = "urgent.1"
lin = torch.nn.Linear(X.shape[1], 1)
opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-4)
best, best_state = 1e9, None
for epoch in range(60):
    perm = torch.randperm(cut)
    for k in range(0, cut, 256):
        i = perm[k:k + 256]
        loss = F.binary_cross_entropy_with_logits(lin(X[i]).squeeze(-1), y[i])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        val = F.binary_cross_entropy_with_logits(lin(X[cut:]).squeeze(-1), y[cut:]).item()
    if val < best:
        best, best_state = val, {k: v.clone() for k, v in lin.state_dict().items()}
lin.load_state_dict(best_state)
print(f"held-out BCE {best:.4f}", flush=True)

# Against the teacher on the blind 500's test half, under both prompts.
sample = read_jsonl(E / "sample.jsonl")
tests = [p for p in sample if in_split(p["url"], "test")]
Z = torch.from_numpy(embed([p["text"] for p in tests]))
with torch.no_grad():
    old = torch.sigmoid(F.linear(Z, ck["heads"][f"{key}.weight"], ck["heads"][f"{key}.bias"]).squeeze(-1)).numpy()
    new = torch.sigmoid(lin(Z).squeeze(-1)).numpy()
for gold_name in [os.environ.get("GOLD", "gold_multi_gpt-5.6-luna_v3b.jsonl"), "gold_multi_gpt-5.6-luna.jsonl"]:
    gold = {r["url"]: r for r in read_jsonl(D / gold_name)}
    pairs = [(i, urgent(gold[p["url"]])) for i, p in enumerate(tests)
             if p["url"] in gold and urgent(gold[p["url"]]) is not None]
    idx, g = map(np.array, zip(*pairs))
    print(f"vs {gold_name}: r old {np.corrcoef(old[idx], g)[0, 1]:.2f} -> new {np.corrcoef(new[idx], g)[0, 1]:.2f}"
          f" (n {len(idx)})", flush=True)

shutil.copy(path, path.with_suffix(".pt.bak"))
ck["heads"][f"{key}.weight"] = lin.weight.detach().clone()
ck["heads"][f"{key}.bias"] = lin.bias.detach().clone()
torch.save(ck, path)
print(f"wrote {path} (previous kept as {path.with_suffix('.pt.bak').name})", flush=True)
