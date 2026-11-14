"""The v1 student with v3c's significance and urgency, on one encoder.

v3c reads significance and urgency the way the v3 prompt asks (three facts
instead of "urgent", a scale with room at the top), and its category is a little
worse than v1's on the blind 500. Two encoders in production would double the
annotator's CPU, so this fits only those two heads again, as linear layers on
v1's frozen encoder, from the v3c teacher's labels. v1's category head and
encoder are untouched: the category it ships is the one production has now.

Then it prints what the choice needs:
- the new heads against the v3c teacher on the blind 500's test half, next to
  v3c's own fine-tuned heads, and v1's for reference;
- the shadow ranker's thresholds carried over by quantile, on 2500 posts none of
  the models trained on: the v1 cut at 3.8 flags a share of posts, and the new
  cut is the one that flags the same share;
- the owner's 25 answers: category, and how many each model calls urgent (the
  owner called none of them urgent).

Writes model_e5_multi_h3c.pt: v1 with the two heads replaced.
Usage: hybrid_heads.py [train posts]
"""
import os as _os
# The checkout that holds data/ and models/: this one, unless NYAN_ROOT names another.
ROOT = _os.environ.get("NYAN_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import glob, json, random, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, str(Path(__file__).parent))
from eval_jev import gold_labels, in_split, is_right, read_jsonl
from multi_label import CATS
from students import Model

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16000
D = Path(f"{ROOT}/data/distill"); E = Path(f"{ROOT}/data/jev_eval"); R = D / "review"
FACTS = ("danger_now", "effect_24h", "breaking")
SCALE = np.array([1, 2, 3, 4, 5], dtype=float)
random.seed(0); torch.manual_seed(0)

def urgent(a: dict) -> float | None:
    if "urgent" in a:
        return float(a["urgent"])
    if any(k in a for k in FACTS):
        return max(float(a.get(k) or 0) for k in FACTS)
    return None

def significance(a: dict) -> int | None:
    try: v = int(round(float(a.get("significance"))))
    except (TypeError, ValueError): return None
    return v if 1 <= v <= 5 else None

# The posts every model is read on: the blind 500, then 2000 unseen production
# posts (the hand-run drift check and the first automatic one), then the owner's 25.
sample = read_jsonl(E / "sample.jsonl")
unseen = read_jsonl(D / "fresh.jsonl") + [r for f in sorted(glob.glob(str(D / "drift/*/sample.jsonl"))) for r in read_jsonl(Path(f))]
owner = {Path(f).stem: json.load(open(f)) for f in sorted(glob.glob(str(R / "answers_dump/answers/*.json")))}
seed = {p: json.load(open(R / f"seed/{p}.json")) for p in owner}
READ = [p["text"] for p in sample] + [r["text"] for r in unseen] + [seed[p]["text"] for p in owner]
nb, nu = len(sample), len(unseen)
print(f"reading {nb} blind + {nu} unseen + {len(owner)} owner posts", flush=True)

teach = [r for r in read_jsonl(D / "teach_multi_gpt-5.6-luna_v3c.jsonl")
         if urgent(r["labels"]) is not None and significance(r["labels"]) is not None]
random.shuffle(teach); teach = teach[:N]

t0 = time.time()
v1 = Model("e5_multi")
Za_train = v1.embed([r["text"] for r in teach]); Za = v1.embed(READ)
print(f"v1 encoder: {len(teach) + len(READ)} posts in {time.time()-t0:.0f}s", flush=True)
del v1.enc; (torch.backends.mps.is_available() and torch.mps.empty_cache())
t0 = time.time()
v3c = Model("e5_multi_v3c")
Zb = v3c.embed(READ)
print(f"v3c encoder: {len(READ)} posts in {time.time()-t0:.0f}s", flush=True)

def fit(X: np.ndarray, y: np.ndarray, out: int) -> torch.nn.Linear:
    """A linear head, the epoch picked on a held-out tenth."""
    X, y = torch.from_numpy(X), torch.from_numpy(y)
    cut = int(0.9 * len(X)); lin = torch.nn.Linear(X.shape[1], out)
    loss_of = (lambda z, t: F.cross_entropy(z, t.long())) if out > 1 else (lambda z, t: F.binary_cross_entropy_with_logits(z.squeeze(-1), t.float()))
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-4)
    best, state = 1e9, None
    for _ in range(60):
        perm = torch.randperm(cut)
        for k in range(0, cut, 256):
            i = perm[k:k + 256]; loss = loss_of(lin(X[i]), y[i])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): val = loss_of(lin(X[cut:]), y[cut:]).item()
        if val < best: best, state = val, {k: v.clone() for k, v in lin.state_dict().items()}
    lin.load_state_dict(state)
    return lin

sig_lin = fit(Za_train, np.array([significance(r["labels"]) - 1 for r in teach]), 5)
urg_lin = fit(Za_train, np.array([min(1.0, max(0.0, urgent(r["labels"]))) for r in teach], dtype=np.float32), 1)
new = dict(v1.ck["heads"])
new["significance.1.weight"], new["significance.1.bias"] = sig_lin.weight.detach().clone(), sig_lin.bias.detach().clone()
new["urgent.1.weight"], new["urgent.1.bias"] = urg_lin.weight.detach().clone(), urg_lin.bias.detach().clone()

out = {
    "v1": {"sig": v1.head("significance", Za) @ SCALE, "urg": v1.head("urgent", Za), "cat": v1.head("category", Za)},
    "v3c": {"sig": v3c.head("significance", Zb) @ SCALE, "urg": v3c.head("urgent", Zb), "cat": v3c.head("category", Zb)},
    "h3c": {"sig": v1.head("significance", Za, new) @ SCALE, "urg": v1.head("urgent", Za, new), "cat": v1.head("category", Za)},
}

def pick(p: np.ndarray, threshold: float = 0.7) -> str:
    ranked = [CATS[i] for i in np.argsort(-p)]
    return ranked[1] if ranked[0] == "not_news" and p[CATS.index("not_news")] < threshold else ranked[0]

# 1. against the v3c teacher, blind 500 test half
tg = {r["url"]: r for r in read_jsonl(D / "gold_multi_gpt-5.6-luna_v3c.jsonl")}
tests = [i for i, p in enumerate(sample) if in_split(p["url"], "test") and tg.get(p["url"], {}).get("category")]
ts = np.array([significance(tg[sample[i]["url"]]) or 0 for i in tests], dtype=float)
tu = np.array([urgent(tg[sample[i]["url"]]) or 0 for i in tests], dtype=float)
ok = ts > 0
print(f"\n1. vs the v3c teacher, blind test half (n {ok.sum()}; teacher urgent share {tu.mean():.3f})")
for m, o in out.items():
    s, u = o["sig"][tests], o["urg"][tests]
    hit = (u > 0.5) & (tu > 0.5)
    print(f"  {m:4} significance MAE {np.abs(s[ok] - ts[ok]).mean():.2f} r {np.corrcoef(s[ok], ts[ok])[0, 1]:.2f} | "
          f"urgent r {np.corrcoef(u, tu)[0, 1]:.2f}, >0.5 on {(u > 0.5).sum()}, precision {hit.sum()}/{(u > 0.5).sum()} recall {hit.sum()}/{(tu > 0.5).sum()}")

# 2. the shadow's thresholds carried over by quantile, on the unseen posts
cal = slice(nb, nb + nu)
news = np.array([pick(p) != "not_news" for p in out["v1"]["cat"][cal]])
print(f"\n2. thresholds by quantile on {nu} unseen posts ({news.sum()} news by v1)")
for scope, mask in (("all", np.ones(nu, bool)), ("news", news)):
    x = out["v1"]["sig"][cal][mask]
    major, minor = (x >= 3.8).mean(), (x <= 1.8).mean()
    line = f"  {scope:4} v1: major>=3.8 {100*major:.1f}%, minor<=1.8 {100*minor:.1f}%, urgent>0.5 {100*(out['v1']['urg'][cal][mask] > 0.5).mean():.1f}%"
    for m in ("v3c", "h3c"):
        y = out[m]["sig"][cal][mask]
        tmaj, tmin = float(np.quantile(y, 1 - major)), float(np.quantile(y, minor))
        same_maj = ((x >= 3.8) & (y >= tmaj)).sum() / max(1, (x >= 3.8).sum())
        line += (f"\n       {m}: major>={tmaj:.2f} (same posts {100*same_maj:.0f}%), minor<={tmin:.2f}, "
                 f"at the old cuts major {100*(y >= 3.8).mean():.1f}% minor {100*(y <= 1.8).mean():.1f}%, "
                 f"urgent>0.5 {100*(out[m]['urg'][cal][mask] > 0.5).mean():.1f}%")
    print(line)

# 3. the owner's 25
RULED = {"p017": "not_news", "p018": "economy"}
ow = slice(nb + nu, None); ids = list(owner)
print(f"\n3. owner's {len(ids)} answers (owner urgent: {sum(bool(owner[p].get('urgent')) for p in ids)})")
for m, o in out.items():
    cats = [pick(p) for p in o["cat"][ow]]
    right = sum(c in ({RULED[p]} if p in RULED else {owner[p]["category"], *owner[p].get("also_ok", [])}) for p, c in zip(ids, cats))
    urg = [p for p, u in zip(ids, o["urg"][ow]) if u > 0.5]
    print(f"  {m:4} category {right}/{len(ids)}, called urgent {len(urg)} {urg}")

# the blind 500 category, to show h3c's is v1's
gold = gold_labels(E); split = np.array([in_split(p["url"], "test") for p in sample])
for m, o in out.items():
    hit = np.array([is_right(pick(p), gold[s["url"]]) for p, s in zip(o["cat"][:nb], sample)])
    print(f"  {m:4} blind category dev {100*hit[~split].mean():.1f}% test {100*hit[split].mean():.1f}% (threshold 0.7)")

torch.save({**v1.ck, "heads": new}, D / "model_e5_multi_h3c.pt")
print(f"\nwrote model_e5_multi_h3c.pt")
