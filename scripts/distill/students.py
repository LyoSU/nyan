"""The trained multi-head students, read in torch: one encoder, and each head applied to its embeddings."""
import os as _os
# The checkout that holds data/ and models/: this one, unless NYAN_ROOT names another.
ROOT = _os.environ.get("NYAN_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

D = Path(f"{ROOT}/data/distill")
SCALE = np.array([1, 2, 3, 4, 5], dtype=float)
MAXLEN = 256
dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


class Model:
    def __init__(self, name: str):
        ck = torch.load(D / f"model_{name}.pt", map_location="cpu")
        self.ck, self.prefix = ck, ck["prefix"]
        self.tok = AutoTokenizer.from_pretrained(ck["repo"])
        self.enc = AutoModel.from_pretrained(ck["repo"]); self.enc.load_state_dict(ck["encoder"]); self.enc.to(dev).eval()

    @torch.no_grad()
    def embed(self, texts: list[str]) -> np.ndarray:
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        res = [None] * len(texts)
        for k in range(0, len(order), 32):
            idx = order[k:k + 32]
            b = self.tok([self.prefix + texts[i][:1500] for i in idx], padding=True, truncation=True,
                         max_length=MAXLEN, return_tensors="pt").to(dev)
            h = self.enc(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            z = ((h * m).sum(1) / m.sum(1)).float().cpu().numpy()
            for j, i in enumerate(idx): res[i] = z[j]
        return np.stack(res)

    def head(self, f: str, Z: np.ndarray, weights: dict | None = None) -> np.ndarray:
        w = weights or self.ck["heads"]
        v = F.linear(torch.from_numpy(Z), w[f"{f}.1.weight"].float(), w[f"{f}.1.bias"].float())
        return (torch.sigmoid(v.squeeze(-1)) if f in ("about_ukraine", "urgent", "clickbait") else F.softmax(v, -1)).numpy()
