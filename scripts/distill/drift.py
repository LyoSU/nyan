"""Is the student still keeping up with its teacher? One command, about a cent.

Pulls N random posts from the last H hours of production (read-only), skipping
any the student trained on or an earlier check already used, and any whose text
repeats a training post: channels copy each other and repeat their own templates,
and on the first check those copies were 30% of the sample and agreed 99% of the
time, which made 91% read as 94%; runs the shipped
student model on them; asks the teacher the same questions; and prints how often
the two agree, next to the joblib head production used before. Every run is kept
under drift/<timestamp>/ and summarised in drift/log.jsonl, so the last line of
the report is a trend, not a single number.

Usage: OPENLUX_KEY=... drift.py [--n 500]   (or TEACHER_VIA=codex: the codex CLI, no key) [--hours 24] [--prompt v1]
       [--model models/student_e5_multi] [--mongo-config configs/mongo_config.json]
Exits 2 when agreement is more than 3 points below the first check, so a
scheduler can tell a drifted model from a finished run.

--prompt is the teacher prompt the shipped student was trained on (v1 now; v3
once the v3 student ships): comparing against a different prompt measures the
prompts, not drift.
"""
import os as _os
# The checkout that holds data/ and models/: this one, unless NYAN_ROOT names another.
ROOT = _os.environ.get("NYAN_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse, json, os, random, re, sys, time
from collections import Counter
from pathlib import Path

ROOT = Path(ROOT); D = ROOT / "data/distill"; OUT = D / "drift"
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--hours", type=float, default=24)
ap.add_argument("--prompt", default="v1")
ap.add_argument("--model", default=str(ROOT / "models/student_e5_multi"))
ap.add_argument("--mongo-config", default=str(ROOT / "configs/mongo_config.json"))
ap.add_argument("--teach", default="teach_multi_gpt-5.6-luna.jsonl", help="the training file whose texts are skipped")
ap.add_argument("--threshold", type=float, default=None, help="not_news threshold; default: the shipped config's")
args = ap.parse_args()
if not os.environ.get("OPENLUX_KEY") and os.environ.get("TEACHER_VIA") != "codex":
    sys.exit("neither OPENLUX_KEY nor TEACHER_VIA=codex is set: the teacher cannot be asked")

stamp = time.strftime("%Y%m%d-%H%M")
run = OUT / stamp; run.mkdir(parents=True, exist_ok=True)
# multi_label reads these at import: a cache file of this run's own, and the prompt.
os.environ["TAG"] = f"_drift_{stamp}"; os.environ["PROMPT"] = args.prompt
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(Path(__file__).parent)]
import multi_label as ml
from nyan.mongo import get_annotated_documents_collection
from nyan.student import Student

# 1. sample
seen = {json.loads(l)["url"] for l in open(D / "pool.jsonl")}
if (D / "fresh.jsonl").exists():  # the first, hand-run check
    seen |= {json.loads(l)["url"] for l in open(D / "fresh.jsonl")}
for old in OUT.glob("*/sample.jsonl"):
    if old.parent != run:
        seen |= {json.loads(l)["url"] for l in open(old)}
def norm(text: str) -> str:
    """A post's first 200 letters and digits: the same for a copy with other spacing or emoji."""
    return re.sub(r"\W+", "", text.lower())[:200]

trained = {norm(json.loads(l)["text"]) for l in open(D / args.teach)} if (D / args.teach).exists() else set()
collection = get_annotated_documents_collection(args.mongo_config)
query = {"pub_time": {"$gte": int(time.time() - args.hours * 3600)}, "patched_text": {"$exists": True},
         "$expr": {"$gte": [{"$strLenCP": {"$ifNull": ["$patched_text", ""]}}, 40]}}
docs = [d for d in collection.find(query, {"url": 1, "channel_id": 1, "pub_time": 1, "patched_text": 1, "category": 1})
        if d["url"] not in seen]
fresh = [d for d in docs if norm(d["patched_text"]) not in trained]
print(f"{len(docs) - len(fresh)} of {len(docs)} unseen posts repeat a training text and are skipped", flush=True)
docs = fresh
random.Random(stamp).shuffle(docs); docs = docs[:args.n]
if len(docs) < 50:
    sys.exit(f"only {len(docs)} unseen posts in the last {args.hours:g}h")
print(f"{len(docs)} posts from the last {args.hours:g}h", flush=True)

# 2. student
config = json.load(open(ROOT / "configs/annotator_config.json"))["student"]
config["path"] = args.model
if args.threshold is not None:
    config["not_news_threshold"] = args.threshold
student = Student.load(config)
if student is None:
    sys.exit(f"no student model at {args.model}")
probs = student.predict([d["patched_text"] for d in docs])

# 3. teacher
answers = ml.label([d["patched_text"] for d in docs])

rows = []
for d, p, a in zip(docs, probs, answers):
    rows.append({"url": d["url"], "channel_id": d.get("channel_id"), "text": d["patched_text"][:1500],
                 "prod": d.get("category"), "student": {"category": student.pick(p["category"]), **student.record(p)},
                 "teacher": a})
ml.write_jsonl(run / "sample.jsonl", rows)

# 4. report
pairs = [r for r in rows if r["teacher"]]
T = [r["teacher"]["category"] for r in pairs]; S = [r["student"]["category"] for r in pairs]; P = [r["prod"] for r in pairs]
agree = sum(s == t for s, t in zip(S, T)) / len(pairs); prod = sum(p == t for p, t in zip(P, T)) / len(pairs)
news = [i for i, t in enumerate(T) if t != "not_news"]
dropped = sum(S[i] == "not_news" for i in news); kept = sum(s != "not_news" and t == "not_news" for s, t in zip(S, T))
heads = {}
for f in ["topic", "scope", "region", "role"]:
    xs = [(r["teacher"].get(f), r["student"][f]) for r in pairs if r["teacher"].get(f) is not None]
    heads[f] = round(sum(x == y for x, y in xs) / len(xs), 3) if xs else None
recall = {c: (sum(S[i] == c for i in range(len(T)) if T[i] == c), n) for c, n in Counter(T).items() if n >= 10}

print(f"\ncategory vs teacher: student {100*agree:.1f}%, prod head {100*prod:.1f}%  (n={len(pairs)}, teacher filtered {ml.filtered[0]}, ${ml.cost():.4f})")
print(f"news dropped {dropped}/{len(news)}, not_news let through {kept}/{len(T) - len(news)}")
print("heads: " + ", ".join(f"{f} {100*v:.0f}%" for f, v in heads.items() if v is not None))
print("recall by teacher category: " + ", ".join(f"{c} {k}/{n}" for c, (k, n) in sorted(recall.items(), key=lambda x: -x[1][1])))
print("disagreements (teacher, student): " + ", ".join(f"{t}->{s} {k}" for (t, s), k in Counter((t, s) for t, s in zip(T, S) if t != s).most_common(8)))

summary = {"run": stamp, "n": len(pairs), "hours": args.hours, "prompt": args.prompt, "teacher": ml.MODEL, "model": student.name,
           "threshold": student.config.not_news_threshold, "deduped": True, "agree": round(agree, 3), "prod_agree": round(prod, 3),
           "news_dropped": dropped, "news": len(news), "not_news_kept": kept, "heads": heads, "cost": round(ml.cost(), 4)}
with open(OUT / "log.jsonl", "a") as w:
    w.write(json.dumps(summary) + "\n")
# Agreement is with one teacher: a run that asked another model measures the two
# models as well as the drift, so the trend and the alarm stay within a teacher.
FIRST = "gpt-5.6-luna"  # the teacher the shipped student learnt from; runs before 2026-11-15 did not record it
history = [h for h in map(json.loads, open(OUT / "log.jsonl")) if h.get("teacher", FIRST) == ml.MODEL]
print(f"\ntrend ({ml.MODEL}): " + "  ".join(f"{h['run'][:8]} {100*h['agree']:.1f}%" for h in history[-8:]))
# The first check (2026-09-22) kept the copies of training posts; on the posts
# that were not copies it agreed on 91.1%, and that is what later runs compare to.
deduped = [h for h in history if h.get("deduped")]
baseline = deduped[0]["agree"] if len(deduped) > 1 or ml.MODEL != FIRST else 0.911
if agree < baseline - 0.03:
    print(f"!! {100*(baseline - agree):.1f} points below the first check: time to relabel fresh posts and retrain")
    sys.exit(2)
