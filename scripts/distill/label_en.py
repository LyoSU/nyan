"""Label more pool posts with the cheap English category question.

Skips posts already in teach_v2.jsonl; stops cleanly when the account runs dry,
since every answer is cached as it arrives.
"""
import os as _os
# The checkout that holds data/ and models/: this one, unless NYAN_ROOT names another.
ROOT = _os.environ.get("NYAN_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys
from pathlib import Path
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/scripts")
from eval_jev import Jev, read_jsonl, write_jsonl
from nyan.jev import STATE_NOTE, load_questions
D = Path(f"{ROOT}/data/distill")
N = int(sys.argv[1])
q = load_questions(str(D / "scripts/cat_en.json"))
done = {r["i"] for r in read_jsonl(D / "teach_v2.jsonl")}
pool = read_jsonl(D / "pool.jsonl")[:40000]
todo = [(i, p) for i, p in enumerate(pool) if i not in done][:N]
jev = Jev(D / "cache.jsonl")
rows, tokens = [], 0
try:
    for k in range(0, len(todo), 500):
        chunk = todo[k:k + 500]
        res = jev.map([({"note": STATE_NOTE, "post_text": p["text"][:1500]}, q) for _, p in chunk], workers=16)
        for (i, p), r in zip(chunk, res):
            rows.append({"i": i, "url": p["url"], "language": p.get("language"), "text": p["text"], "answers": r["answers"]})
            tokens += r["usage"].get("input_tokens", 0)
        print(f"{len(rows)} labelled, ${tokens * 0.042 / 1e6:.2f}", flush=True)
finally:
    write_jsonl(D / "teach_en.jsonl", rows)
