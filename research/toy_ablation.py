"""The smallest proof that additive diarization conditioning works (PLAN.md §8 step 2).

Task: speaker-attributed recognition of simulated rooms. Input per 20 ms frame
is a frozen Whisper-tiny encoder feature of the *mixture* (50 Hz, the encoder's
native rate; the stand-in for codec tokens), optionally plus
DiarizationConditioning(activity upsampled 12.5 -> 50 Hz, the same x4 repeat
Dixtral uses). Target is the serialized speaker-attributed *character*
sequence per 30 s window, "<spk0>some words<spk2>more words", in start-time
order, trained with CTC. Identical models are trained with cond = 0 / 1 / 2;
dev-set cpWER (by slot tag, word level) decides.

v1 of this file used word-level CTC at 12.5 Hz over a 9k-word vocabulary and
learned nothing in 2.5k steps (dev cpWER 97.6 % for the conditioned arm) -
too little data for a word vocabulary from scratch. Character CTC on frozen
encoder features is the standard small-data setup and does learn.

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
from conditioning import DiarizationConditioning, PerLayerStateConditioning, corrupt_activity  # noqa: E402

ROOT = Path(__file__).resolve().parent / "data"
ROOMS = sorted(glob.glob(str(ROOT / "rooms" / "room*")))
SR, FRAME_S, K = 16000, 0.08, 4
FPS, WIN_S = 50, 30                       # encoder frames per second; window length
WIN = FPS * WIN_S                         # 1500 frames per window
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def read_wav(p):
    with wave.open(p, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


# ------------------------------------------------------------------ features
def features():
    """Whisper-tiny encoder at its native 50 Hz, cached as float16 (T50, 384)."""
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
            feats.append(h[: n_frames * 4].cpu().numpy().astype(np.float16))
        np.save(out, np.concatenate(feats))
    print("features cached for", len(ROOMS), "rooms")


# --------------------------------------------------------------------- data
class Vocab:
    """Characters + one tag per slot. Tags are single symbols so CTC targets stay short."""

    def __init__(self, rooms=None):
        self.tags = [f"<spk{k}>" for k in range(K)]
        self.itos = ["<blank>"] + self.tags + [" ", "'"] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def segs(self, d):
        return sorted((json.loads(l) for l in open(Path(d) / "transcript.jsonl")), key=lambda s: s["t0"])

    def encode_window(self, d, w):
        """Serialized target for window w: segments starting in [w*WIN_S, (w+1)*WIN_S)."""
        ids = []
        for s in self.segs(d):
            if s["text"] == "[backchannel]" or not (w * WIN_S <= s["t0"] < (w + 1) * WIN_S):
                continue
            ids.append(self.stoi[self.tags[s["slot"]]])
            ids += [self.stoi[c] for c in s["text"].lower() if c in self.stoi]
        return ids

    def ref_by_slot(self, d, w):
        out = {}
        for s in self.segs(d):
            if s["text"] != "[backchannel]" and w * WIN_S <= s["t0"] < (w + 1) * WIN_S:
                out.setdefault(s["slot"], []).extend(s["text"].lower().split())
        return out


def load_windows(d):
    """-> [(feats (WIN,384), active (WIN,K) at 50 Hz, window index)]"""
    f = torch.from_numpy(np.load(Path(d) / "feats.npy").astype(np.float32))
    a = torch.from_numpy(np.load(Path(d) / "activity.npy"))
    act = torch.zeros(a.shape[0], K, dtype=torch.bool); act[:, : a.shape[1]] = a
    act50 = act.repeat_interleave(4, dim=0)                    # 12.5 Hz -> 50 Hz, x4 repeat
    T = min(f.shape[0], act50.shape[0]); out = []
    for w in range(T // WIN):
        out.append((f[w * WIN:(w + 1) * WIN], act50[w * WIN:(w + 1) * WIN], w))
    return out


# -------------------------------------------------------------------- model
class Recognizer(nn.Module):
    """cond: 0 = none, 1 = additive input conditioning, 2 = additive + per-layer state affine."""

    def __init__(self, vocab, d_in=384, d=256, layers=4, cond=1):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.cond = DiarizationConditioning(d, K) if cond else None
        self.per_layer = PerLayerStateConditioning(d, layers) if cond == 2 else None
        self.pos = nn.Embedding(4096, d)
        self.layers = nn.ModuleList(nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
                                    for _ in range(layers))
        self.head = nn.Linear(d, vocab)

    def forward(self, feats, active):
        h = self.proj(feats) + self.pos(torch.arange(feats.shape[1], device=feats.device))[None]
        if self.cond is not None:
            h = h + self.cond(active)
        state = PerLayerStateConditioning.frame_state(active) if self.per_layer is not None else None
        for i, layer in enumerate(self.layers):
            if self.per_layer is not None:
                h = self.per_layer(h, i, state)
            h = layer(h)
        return self.head(h)                                      # (B,T,V) logits


def ctc_decode(logits, vocab):
    ids = logits.argmax(-1).tolist(); out, prev = [], 0
    for i in ids:
        if i != prev and i != 0:
            out.append(vocab.itos[i])
        prev = i
    return out


def cp_wer(ref_by_slot, hyp_tokens, vocab):
    """Speaker-attributed WER with the *known* slot mapping (tags are slot ids); chars -> words."""
    import jiwer
    hyp = {}; cur = None; buf = []
    def flush():
        if cur is not None and buf:
            hyp.setdefault(cur, []).extend("".join(buf).split())
    for t in hyp_tokens:
        if t in vocab.tags:
            flush(); cur = vocab.tags.index(t); buf = []
        else:
            buf.append(t)
    flush()
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


def train(cond: int, steps: int, bs: int = 8, lr: float = 3e-4, seed: int = 0, corrupt: bool = True,
          rooms_dir: str = "rooms", ckpt_every: int = 500):
    torch.manual_seed(seed); random.seed(seed)
    rooms = sorted(glob.glob(str(ROOT / rooms_dir / "room*")))
    n_dev = max(40, len(rooms) // 10)
    tr, dv = rooms[:-n_dev], rooms[-n_dev:]
    vocab = Vocab()
    win_tr = [(d, *x) for d in tr for x in load_windows(d)]
    win_dv = [(d, *x) for d in dv for x in load_windows(d)]
    tgts = {(d, w): vocab.encode_window(d, w) for d, _, _, w in win_tr + win_dv}
    win_tr = [x for x in win_tr if len(tgts[(x[0], x[3])]) > 0]
    print(f"[cond={cond}] {len(win_tr)} train windows, {len(win_dv)} dev windows, vocab {len(vocab.itos)}", flush=True)
    m = Recognizer(len(vocab.itos), cond=cond).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    ctc = nn.CTCLoss(zero_infinity=True)
    g = torch.Generator().manual_seed(seed)
    tag = f"{rooms_dir}_cond{cond}_s{steps}"
    ckpt = ROOT / f"toy_ckpt_{tag}.pt"
    start = 1
    if ckpt.exists():                                            # resume after a crash / sleep
        st = torch.load(ckpt, map_location=DEV)
        m.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start = st["step"] + 1
        print(f"[cond={cond}] resumed from step {st['step']}", flush=True)
    t0 = time.perf_counter()
    for step in range(start, steps + 1):
        batch = random.sample(win_tr, bs)
        f = torch.stack([x[1] for x in batch]); a = torch.stack([x[2] for x in batch]); lens = [WIN] * bs
        if corrupt and cond:
            a = corrupt_activity(a, jitter_frames=12, max_lag=48, generator=g)   # 50 Hz frames
        y = [torch.tensor(tgts[(x[0], x[3])]) for x in batch]
        logp = m(f.to(DEV), a.to(DEV)).log_softmax(-1).transpose(0, 1)      # (T,B,V)
        loss = ctc(logp.cpu().float(), torch.cat(y), torch.tensor(lens), torch.tensor([len(t) for t in y]))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        if step % 100 == 0 or step == steps:
            print(f"[cond={int(cond)}] step {step} loss {float(loss):.3f} ({time.perf_counter() - t0:.0f}s)", flush=True)
        if step % ckpt_every == 0 or step == steps:
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, ckpt)
    m.eval(); err = n = 0; plain_err = 0
    with torch.no_grad():
        for d, x, act, w in win_dv:
            ref = vocab.ref_by_slot(d, w)
            if not ref:
                continue
            hyp = ctc_decode(m(x[None].to(DEV), act[None].to(DEV))[0], vocab)
            nref = sum(len(v) for v in ref.values())
            err += cp_wer(ref, hyp, vocab) * nref; n += nref
            import jiwer
            words = "".join(t for t in hyp if t not in vocab.tags).split(); refw = [w_ for v in ref.values() for w_ in v]
            o = jiwer.process_words(" ".join(refw), " ".join(words) or "-"); plain_err += o.substitutions + o.deletions + o.insertions
    res = {"cond": cond, "steps": steps, "rooms": rooms_dir, "dev_cpwer": round(err / max(n, 1), 4),
           "dev_wer_speaker_agnostic": round(plain_err / max(n, 1), 4), "n_dev": len(dv),
           "params": sum(p.numel() for p in m.parameters()), "train_rooms": len(tr), "wall_s": round(time.perf_counter() - t0)}
    print(json.dumps(res))
    out = ROOT / f"toy_result_{tag}.json"; out.write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--cond", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--rooms", default="rooms", help="subdirectory of research/data with room*/")
    a = ap.parse_args()
    if a.cmd == "features":
        ROOMS[:] = sorted(glob.glob(str(ROOT / a.rooms / "room*")))
        features()
    elif a.cmd == "train":
        train(a.cond, a.steps, a.bs, rooms_dir=a.rooms)
