"""The smallest proof that additive diarization conditioning works (PLAN.md §8 step 2).

Task: speaker-attributed recognition of simulated rooms. Input per 80 ms frame
is a frozen Whisper-tiny encoder feature of the *mixture* (the stand-in for
Mimi tokens), optionally plus DiarizationConditioning(activity). Target is the
serialized speaker-attributed word sequence, "<spk0> w w w <spk2> w w ...",
in start-time order, trained with CTC. Two identical models are trained, one
with the conditioning and one without; the dev-set cpWER decides.

Why this is a fair test of the plan's mechanism and not a toy in the
pejorative sense: the encoder is frozen and hears the mixture, so the only
way to know *which* speaker's words to emit and *where* a turn starts is the
conditioning — exactly the job it has in the full model.

    python research/toy_ablation.py features   # cache Whisper-tiny features for every room
    python research/toy_ablation.py train --cond 1 --steps 3000
    python research/toy_ablation.py train --cond 0 --steps 3000
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conditioning import DiarizationConditioning, corrupt_activity  # noqa: E402

ROOT = Path(__file__).resolve().parent / "data"
ROOMS = sorted(glob.glob(str(ROOT / "rooms" / "room*")))
SR, FRAME_S, K = 16000, 0.08, 4
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def read_wav(p):
    with wave.open(p, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


# ------------------------------------------------------------------ features
def features():
    """Whisper-tiny encoder, 50 Hz -> mean-pooled to 12.5 Hz, cached as float16."""
    from transformers import WhisperFeatureExtractor, WhisperModel
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
    enc = WhisperModel.from_pretrained("openai/whisper-tiny").encoder.to(DEV).eval()
    for d in ROOMS:
        out = Path(d) / "feats.npy"
        if out.exists():
            continue
        x = read_wav(str(Path(d) / "audio.wav"))
        feats = []
        for i in range(0, x.size, 30 * SR):                      # Whisper windows of 30 s
            seg = x[i:i + 30 * SR]
            n_frames = int(round(seg.size / SR / FRAME_S))
            inp = fe(seg, sampling_rate=SR, return_tensors="pt").input_features.to(DEV)
            with torch.no_grad():
                h = enc(inp).last_hidden_state[0]                 # (1500, 384) = 30 s at 50 Hz
            h = h[: n_frames * 4].reshape(n_frames, 4, -1).mean(1)
            feats.append(h.cpu().numpy().astype(np.float16))
        np.save(out, np.concatenate(feats))
    print("features cached for", len(ROOMS), "rooms")


# --------------------------------------------------------------------- data
class Vocab:
    def __init__(self, rooms):
        words = set()
        for d in rooms:
            for line in open(Path(d) / "transcript.jsonl"):
                s = json.loads(line)
                if s["text"] != "[backchannel]":
                    words.update(s["text"].split())
        self.tags = [f"<spk{k}>" for k in range(K)]
        self.itos = ["<blank>"] + self.tags + sorted(words)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, d):
        segs = sorted((json.loads(l) for l in open(Path(d) / "transcript.jsonl")), key=lambda s: s["t0"])
        ids = []
        for s in segs:
            if s["text"] == "[backchannel]":
                continue
            ids.append(self.stoi[self.tags[s["slot"]]])
            ids += [self.stoi[w] for w in s["text"].split() if w in self.stoi]
        return ids

    def ref_by_slot(self, d):
        out = {}
        for l in open(Path(d) / "transcript.jsonl"):
            s = json.loads(l)
            if s["text"] != "[backchannel]":
                out.setdefault(s["slot"], []).extend(s["text"].split())
        return out


def load_room(d):
    f = torch.from_numpy(np.load(Path(d) / "feats.npy").astype(np.float32))
    a = torch.from_numpy(np.load(Path(d) / "activity.npy"))
    T = min(f.shape[0], a.shape[0])
    act = torch.zeros(T, K, dtype=torch.bool); act[:, : a.shape[1]] = a[:T]
    return f[:T], act


# -------------------------------------------------------------------- model
class Recognizer(nn.Module):
    def __init__(self, vocab, d_in=384, d=256, layers=4, cond=True):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.cond = DiarizationConditioning(d, K) if cond else None
        self.pos = nn.Embedding(4096, d)
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, vocab)

    def forward(self, feats, active):
        h = self.proj(feats) + self.pos(torch.arange(feats.shape[1], device=feats.device))[None]
        if self.cond is not None:
            h = h + self.cond(active)
        return self.head(self.enc(h))                            # (B,T,V) logits


def ctc_decode(logits, vocab):
    ids = logits.argmax(-1).tolist(); out, prev = [], 0
    for i in ids:
        if i != prev and i != 0:
            out.append(vocab.itos[i])
        prev = i
    return out


def cp_wer(ref_by_slot, hyp_tokens, vocab):
    """Speaker-attributed WER with the *known* slot mapping (tags are slot ids)."""
    import jiwer
    hyp = {}; cur = None
    for t in hyp_tokens:
        if t in vocab.tags:
            cur = vocab.tags.index(t)
        elif cur is not None:
            hyp.setdefault(cur, []).append(t)
    err = n = 0
    for k, ref in ref_by_slot.items():
        h = hyp.get(k, [])
        if not h:
            err += len(ref)
        else:
            o = jiwer.process_words(" ".join(ref), " ".join(h)); err += o.substitutions + o.deletions + o.insertions
        n += len(ref)
    for k, h in hyp.items():
        if k not in ref_by_slot:
            err += len(h)
    return err / max(n, 1)


def train(cond: bool, steps: int, bs: int = 8, lr: float = 3e-4, seed: int = 0, corrupt: bool = True):
    torch.manual_seed(seed); random.seed(seed)
    tr, dv = ROOMS[:-40], ROOMS[-40:]
    vocab = Vocab(ROOMS)
    data = {d: load_room(d) for d in ROOMS}
    tgts = {d: vocab.encode(d) for d in ROOMS}
    m = Recognizer(len(vocab.itos), cond=cond).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    ctc = nn.CTCLoss(zero_infinity=True)
    g = torch.Generator().manual_seed(seed)
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        batch = random.sample(tr, bs)
        T = max(data[d][0].shape[0] for d in batch)
        f = torch.zeros(bs, T, 384); a = torch.zeros(bs, T, K, dtype=torch.bool); lens = []
        for i, d in enumerate(batch):
            x, act = data[d]; f[i, : x.shape[0]] = x; a[i, : act.shape[0]] = act; lens.append(x.shape[0])
        if corrupt and cond:
            a = corrupt_activity(a, generator=g)
        y = [torch.tensor(tgts[d]) for d in batch]
        logp = m(f.to(DEV), a.to(DEV)).log_softmax(-1).transpose(0, 1)      # (T,B,V)
        loss = ctc(logp.cpu().float(), torch.cat(y), torch.tensor(lens), torch.tensor([len(t) for t in y]))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        if step % 100 == 0 or step == steps:
            print(f"[cond={int(cond)}] step {step} loss {float(loss):.3f} ({time.perf_counter() - t0:.0f}s)", flush=True)
    m.eval(); scores = []
    with torch.no_grad():
        for d in dv:
            x, act = data[d]
            hyp = ctc_decode(m(x[None].to(DEV), act[None].to(DEV))[0], vocab)
            scores.append(cp_wer(vocab.ref_by_slot(d), hyp, vocab))
    res = {"cond": cond, "steps": steps, "dev_cpwer": round(float(np.mean(scores)), 4), "n_dev": len(dv),
           "params": sum(p.numel() for p in m.parameters()), "train_rooms": len(tr), "wall_s": round(time.perf_counter() - t0)}
    print(json.dumps(res))
    out = ROOT / f"toy_result_cond{int(cond)}.json"; out.write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--cond", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "features":
        features()
    elif a.cmd == "train":
        train(bool(a.cond), a.steps, a.bs)
