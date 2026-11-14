"""Try int8 variants of onnx_<name>/model.onnx and score each on the blind 500."""
import os as _os
# The checkout that holds data/ and models/: this one, unless NYAN_ROOT names another.
ROOT = _os.environ.get("NYAN_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, time
from pathlib import Path
import numpy as np, onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoTokenizer
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/scripts")
from eval_jev import gold_labels, in_split, is_right, read_jsonl
import json
name = sys.argv[1]
D = Path(f"{ROOT}/data/distill"); E = Path(f"{ROOT}/data/jev_eval"); OUT = D / f"onnx_{name}"
meta = json.load(open(OUT / "meta.json")); CATS = meta["choices"]["category"]; prefix = meta["prefix"]
tok = AutoTokenizer.from_pretrained(OUT)
sample = read_jsonl(E / "sample.jsonl"); gold = gold_labels(E)
texts = [prefix + p["text"][:1500] for p in sample]
split = np.array([in_split(p["url"], "test") for p in sample])
order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
batches = []
for k in range(0, len(order), 16):
    idx = order[k:k + 16]
    b = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=256, return_tensors="np")
    batches.append((idx, b["input_ids"].astype(np.int64), b["attention_mask"].astype(np.int64)))
def run(path):
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    cat = [None] * len(texts); emb = [None] * len(texts); t0 = time.time()
    for idx, ids, m in batches:
        e, c = s.run(["embedding", "category"], {"input_ids": ids, "attention_mask": m})
        for j, i in enumerate(idx): cat[i], emb[i] = c[j], e[j]
    return cat, emb, time.time() - t0
ref_cat, ref_emb, _ = run(OUT / "model.onnx")
variants = {"matmul": dict(op_types_to_quantize=["MatMul"]),
            "gather": dict(op_types_to_quantize=["Gather"]),
            "matmul_nopc": dict(op_types_to_quantize=["MatMul"], per_channel=False, reduce_range=False),
            "matmul_u8": dict(op_types_to_quantize=["MatMul"], weight_type=QuantType.QUInt8)}
for v, kw in variants.items():
    p = OUT / f"q_{v}.onnx"
    kw = {"weight_type": QuantType.QInt8, "per_channel": True, **kw}
    quantize_dynamic(OUT / "model.onnx", p, **kw)
    cat, emb, tt = run(p)
    pred = [CATS[int(np.argmax(c))] for c in cat]; rp = [CATS[int(np.argmax(c))] for c in ref_cat]
    ok = np.array([is_right(pred[i], gold[sample[i]["url"]]) for i in range(len(texts))])
    cos = np.mean([np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b) for a, b in zip(emb, ref_emb)])
    print(f"{v:12} {p.stat().st_size/1e6:5.0f} MB  dev {100*ok[~split].mean():.1f} test {100*ok[split].mean():.1f}  same {100*np.mean([a==b for a,b in zip(pred,rp)]):.1f}%  cos {cos:.4f}  {1000*tt/len(texts):.0f} ms/post", flush=True)
