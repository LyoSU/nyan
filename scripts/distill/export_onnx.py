"""Export a multi-head student (ft_multi.py checkpoint) to ONNX, fp32 and int8,
and check both against the torch model on the blind 500.

The graph takes input_ids and attention_mask and returns the pooled embedding
plus probabilities for every head (softmax for choices, sigmoid for binaries),
so the runtime needs only the tokenizer and meta.json.

Usage: export_onnx.py <name>   (reads model_<name>.pt, writes onnx_<name>/)
"""
import os as _os
# The checkout that holds data/ and models/: this one, unless NYAN_ROOT names another.
ROOT = _os.environ.get("NYAN_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json, sys, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, str(Path(__file__).parent))
from eval_jev import gold_labels, in_split, is_right, read_jsonl
from multi_label import CATS, TOPIC_KEYS, REGIONS, SCOPES, ROLES

name = sys.argv[1]
D = Path(f"{ROOT}/data/distill"); E = Path(f"{ROOT}/data/jev_eval")
OUT = D / f"onnx_{name}"; OUT.mkdir(exist_ok=True)
MAXLEN = 256
CHOICES = {"category": CATS, "topic": TOPIC_KEYS, "scope": SCOPES, "region": REGIONS, "role": ROLES, "significance": [1, 2, 3, 4, 5]}
BINARY = ["about_ukraine", "urgent", "clickbait"]
HEADS = list(CHOICES) + BINARY

ck = torch.load(D / f"model_{name}.pt", map_location="cpu")
repo, prefix = ck["repo"], ck["prefix"]
tok = AutoTokenizer.from_pretrained(repo)
enc = AutoModel.from_pretrained(repo); enc.load_state_dict(ck["encoder"])
H = enc.config.hidden_size
heads = torch.nn.ModuleDict({f: torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(H, len(CHOICES[f]) if f in CHOICES else 1)) for f in HEADS})
heads.load_state_dict(ck["heads"])

class Student(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.enc, self.heads = enc, heads
    def forward(self, input_ids, attention_mask):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        z = (h * m).sum(1) / m.sum(1)
        out = [z]
        for f in HEADS:
            v = self.heads[f](z)
            out.append(torch.softmax(v, -1) if f in CHOICES else torch.sigmoid(v.squeeze(-1)))
        return tuple(out)

model = Student().eval()
NAMES = ["embedding"] + HEADS
fp32, int8, emb8 = OUT / "model.onnx", OUT / "model_int8.onnx", OUT / "model_emb8.onnx"
dummy = tok([prefix + "Тест", prefix + "Ще один довший тест для форми"], padding=True, return_tensors="pt")
t0 = time.time()
torch.onnx.export(model, (dummy["input_ids"], dummy["attention_mask"]), fp32, dynamo=False, opset_version=17,
                  input_names=["input_ids", "attention_mask"], output_names=NAMES,
                  dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                                **{n: {0: "batch"} for n in NAMES}})
print(f"fp32 exported in {time.time()-t0:.0f}s, {fp32.stat().st_size/1e6:.0f} MB", flush=True)

from onnxruntime.quantization import QuantType, quantize_dynamic
quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul", "Gather"])
print(f"int8 {int8.stat().st_size/1e6:.0f} MB", flush=True)
# What ships: int8 in the embedding table only, lossless on the blind 500 (quant_variants.py).
quantize_dynamic(fp32, emb8, weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["Gather"])
print(f"emb8 {emb8.stat().st_size/1e6:.0f} MB", flush=True)

tok.save_pretrained(OUT)
json.dump({"name": name, "repo": repo, "prefix": prefix, "max_length": MAXLEN, "max_chars": 1500, "outputs": NAMES,
           "choices": CHOICES, "binary": BINARY}, open(OUT / "meta.json", "w"), ensure_ascii=False, indent=1)

# parity on the blind 500
import onnxruntime as ort
sample = read_jsonl(E / "sample.jsonl"); gold = gold_labels(E)
texts = [prefix + p["text"][:1500] for p in sample]
split = np.array([in_split(p["url"], "test") for p in sample])
order = sorted(range(len(texts)), key=lambda i: len(texts[i]))

def run(fn):
    res = {n: [None] * len(texts) for n in NAMES}; t0 = time.time()
    for k in range(0, len(order), 16):
        idx = order[k:k + 16]
        b = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=MAXLEN, return_tensors="np")
        outs = fn(b["input_ids"].astype(np.int64), b["attention_mask"].astype(np.int64))
        for n, o in zip(NAMES, outs):
            for j, i in enumerate(idx): res[n][i] = np.asarray(o[j])
    return res, time.time() - t0

def torch_fn(ids, mask):
    with torch.no_grad(): return [o.numpy() for o in model(torch.from_numpy(ids), torch.from_numpy(mask))]

def ort_fn(path):
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return lambda ids, mask: s.run(NAMES, {"input_ids": ids, "attention_mask": mask})

def acc(res):
    pred = [CATS[int(np.argmax(p))] for p in res["category"]]
    ok = np.array([is_right(pred[i], gold[sample[i]["url"]]) for i in range(len(texts))])
    return pred, f"dev {100*ok[~split].mean():.1f}% test {100*ok[split].mean():.1f}%"

ref, tt = run(torch_fn); rpred, line = acc(ref)
print(f"torch   {line}  ({1000*tt/len(texts):.0f} ms/post)", flush=True)
for label, path in [("onnx32", fp32), ("int8", int8), ("emb8", emb8)]:
    res, tt = run(ort_fn(path)); pred, line = acc(res)
    same = np.mean([a == b for a, b in zip(pred, rpred)])
    diff = max(float(np.abs(res["category"][i] - ref["category"][i]).max()) for i in range(len(texts)))
    cos = np.mean([np.dot(res["embedding"][i], ref["embedding"][i]) / np.linalg.norm(res["embedding"][i]) / np.linalg.norm(ref["embedding"][i]) for i in range(len(texts))])
    print(f"{label:7} {line}  same category {100*same:.1f}%, max prob diff {diff:.3f}, embedding cos {cos:.4f}  ({1000*tt/len(texts):.0f} ms/post)", flush=True)
