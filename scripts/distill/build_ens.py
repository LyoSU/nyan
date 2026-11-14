"""Put together what `ft_ens.py` trains on and what it is read on.

Training rows, one per line of `teach_ens.jsonl`, each with the category as a
distribution and the other fields where they are known:
  old     the 40 000 posts the first student learnt from: category, topic, scope,
          region, role, about_ukraine and clickbait from the v1 teacher,
          significance and urgency from v3c (which is what production's heads
          were fitted on). A post the rubric patterns take is not_news whatever
          the teacher said: that is how a decision made after the labelling
          reaches the model.
  judged  fresh posts the student was unsure of, labelled by two or three other
          models: the mean of their answers. Written twice, because they are a
          seventh of the rows and the reason for the retraining.
  pseudo  fresh posts the student was sure of: its own distribution.
  rubric  fresh posts the patterns take: not_news.
On fresh posts the other fields are the production model's own answers, so the
heads nobody relabelled stay where they are; the topic is dropped when it
belongs to a category the judges did not choose.

Read, never trained on: `ens/dev_labels.json` (1000 fresh posts, for choosing
the epoch) and `ens/test_labels.json` (1000 others). `category` is what most
judges said, `ok` is every category any of them said.

Usage: build_ens.py <judge dir>,<judge dir>[,...]   (names under judges/out/<set>/)
"""
import os as _os
ROOT = _os.environ.get("NYAN_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json, re, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # nyan from this checkout, the data from ROOT
from eval_jev import read_jsonl, write_jsonl
from multi_label import CATS
from nyan.rubrics import RubricDetector

D = Path(f"{ROOT}/data/distill"); J = D / "judges"; ENS = D / "ens"
JUDGES = sys.argv[1].split(",")
FACTS = ("danger_now", "effect_24h", "breaking")
detector = RubricDetector(json.load(open(Path(__file__).resolve().parents[2] / "configs/annotator_config.json"))["rubric_detector"])

def answers(set_: str, judge: str) -> dict[str, dict]:
    out = {}
    for f in sorted((J / "out" / set_ / judge).glob("p*.txt")):
        for m in re.finditer(r"\{[^{}]*\}", f.read_text()):
            try: r = json.loads(m.group())
            except ValueError: continue
            if isinstance(r, dict) and r.get("category") in CATS and "id" in r: out[str(r["id"])] = r
    return out

def spread(a: dict) -> dict[str, float]:
    """One answer as a distribution: its confidence on the category, the rest on the second."""
    try: conf = min(1.0, max(0.0, float(a.get("confidence", 1.0))))
    except (TypeError, ValueError): conf = 1.0
    p = dict.fromkeys(CATS, 0.0); p[a["category"]] += conf
    second = a.get("second")
    if second in CATS and second != a["category"]: p[second] += 1 - conf
    else:
        for c in CATS: p[c] += (1 - conf) / len(CATS)
    return p

def mean(ps: list[dict]) -> dict[str, float]:
    return {c: round(sum(p[c] for p in ps) / len(ps), 4) for c in CATS}

NOT_NEWS = {c: float(c == "not_news") for c in CATS}
rows, said = [], Counter()

# old
v3c = {r["url"]: r["labels"] for r in read_jsonl(D / "teach_multi_gpt-5.6-luna_v3c.jsonl")}
for r in read_jsonl(D / "teach_multi_gpt-5.6-luna.jsonl"):
    a = r["labels"]
    if a.get("category") not in CATS: continue
    lab = {f: a.get(f) for f in ("topic", "scope", "region", "role", "about_ukraine", "clickbait")}
    lab["category_p"] = spread(a)
    b = v3c.get(r["url"]) or {}
    try: lab["significance"] = int(round(float(b.get("significance"))))
    except (TypeError, ValueError): lab["significance"] = None
    lab["urgent"] = max(float(b.get(k) or 0) for k in FACTS) if any(k in b for k in FACTS) else None
    if detector(r["text"]):
        said["old, a pattern overrules " + a["category"]] += a["category"] != "not_news"
        lab["category_p"] = NOT_NEWS
        if not str(lab["topic"]).startswith("not_news."): lab["topic"] = None
    rows.append({"url": r["url"], "text": r["text"], "src": "old", "labels": lab}); said["old"] += 1

def fresh(post: dict, heads: dict | None, p: dict, src: str) -> dict:
    lab = {"category_p": p}
    h = heads or {}
    for f in ("topic", "scope", "region", "role", "significance", "about_ukraine", "urgent", "clickbait"):
        lab[f] = h.get(f)
    top = max(p, key=p.get)
    if not str(lab["topic"]).startswith(top + "."): lab["topic"] = None
    return {"url": post["url"], "text": post["text"], "src": src, "labels": lab}

def judged(set_: str, posts: list[dict]) -> dict[int, list[dict]]:
    lab = {j: answers(set_, j) for j in JUDGES}
    print(set_, {j: len(lab[j]) for j in JUDGES})
    return {p["n"]: [lab[j][str(p["n"])] for j in JUDGES if str(p["n"]) in lab[j]] for p in posts}

def read_set(posts: list[dict], votes: dict, name: str) -> None:
    out = []
    for p in posts:
        v = votes[p["n"]]
        if len(v) < len(JUDGES): continue
        top, n = Counter(a["category"] for a in v).most_common(1)[0]
        if n * 2 <= len(v): continue  # no two agree: nothing to be right about
        out.append({"url": p["url"], "text": p["text"], "category": top, "ok": sorted({a["category"] for a in v})})
    json.dump(out, open(ENS / f"{name}_labels.json", "w"), ensure_ascii=False)
    print(f"{name}: {len(out)} of {len(posts)} posts read, all judges agree on {sum(len(r['ok']) == 1 for r in out)}")

ens = json.load(open(J / "ens8500.json")); votes = judged("ens8500", ens)
dev_urls = {p["url"] for p in json.load(open(ENS / "dev.json"))}
read_set([p for p in ens if p["url"] in dev_urls], votes, "dev")
for p in ens:
    if p["url"] in dev_urls or len(votes[p["n"]]) < 2: continue
    row = fresh(p, p["heads"], mean([spread(a) for a in votes[p["n"]]]), "judged")
    rows += [row, row]; said["judged"] += 1

nov = json.load(open(J / "nov2000.json")); votes = judged("nov2000", nov)
read_set([p for p in nov if p["part"] == "random"], votes, "test")
prod = {}
for line in open(D / "prod_20261114.jsonl"):
    r = json.loads(line); prod[r["url"]] = r.get("student")
for p in nov:
    if p["part"] == "random" or len(votes[p["n"]]) < 2: continue
    row = fresh(p, prod.get(p["url"]), mean([spread(a) for a in votes[p["n"]]]), "judged")
    rows += [row, row]; said["judged"] += 1

for p in json.load(open(ENS / "pseudo.json")):
    total = sum(p["scores"].values())
    rows.append(fresh(p, p["heads"], {c: round(p["scores"].get(c, 0.0) / total, 4) for c in CATS}, "pseudo")); said["pseudo"] += 1
for p in json.load(open(ENS / "rubric.json")):
    rows.append(fresh(p, p["heads"], NOT_NEWS, "rubric")); said["rubric"] += 1

held = {r["url"] for part in ("dev", "test") for r in json.load(open(ENS / f"{part}_labels.json"))} | dev_urls | {p["url"] for p in nov if p["part"] == "random"}
rows = [r for r in rows if r["url"] not in held]
write_jsonl(D / "teach_ens.jsonl", rows)
print(len(rows), "rows:", dict(said))
print("category of the rows:", Counter(max(r["labels"]["category_p"], key=r["labels"]["category_p"].get) for r in rows).most_common())
