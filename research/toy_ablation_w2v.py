"""Conditioning proof on a *pretrained* listener (PLAN.md §8 job 2/4, v3).

v1/v2 of the toy trained a CTC recognizer from scratch on frozen Whisper-tiny
features and never learned the acoustics (dev WER > 100 %), so they could not
measure conditioning. v3 keeps "what was said" fixed and pretrained and asks
only the question conditioning exists to answer - *who* said it and *where
turns start*:

  listener   facebook/wav2vec2-base-960h (CTC, LibriSpeech ~3 % WER), frozen
  injection  cond=1: DiarizationConditioning added to the encoder input
             (after feature_projection), i.e. the plan's additive variant
             cond=2: cond=1 + PerLayerStateConditioning inside every encoder
             layer (the DiCoW/FDDT-style variant)
             cond=0: nothing (the model cannot know who is speaking)
  head       the CTC vocabulary gains four <spk k> symbols; only the new
             rows of lm_head, the conditioning modules, and (for stability)
             the final layer norm train.
  target     "<spk1>SOME|WORDS|<spk0>MORE|WORDS" per 30 s window, wav2vec2's
             uppercase alphabet with | as the word delimiter.
  metric     dev cpWER by slot tag (word level, known mapping) and the
             speaker-agnostic WER as a sanity check that the listener is intact.

    python research/toy_ablation_w2v.py --cond 1 --steps 1500 --rooms rooms
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditioning import DiarizationConditioning, PerLayerStateConditioning, corrupt_activity  # noqa: E402

ROOT = Path(__file__).resolve().parent / "data"
SR, FRAME_S, K, WIN_S = 16000, 0.08, 4, 30
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL = "facebook/wav2vec2-base-960h"


def read_wav(p):
    with wave.open(str(p), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


# --------------------------------------------------------------------- data
class Windows:
    def __init__(self, rooms, tok):
        self.items = []                       # (audio (WIN_S*SR,), activity (T80, K), target ids, ref_by_slot)
        self.tok = tok
        for d in rooms:
            x = read_wav(Path(d) / "audio.wav")
            a = np.load(Path(d) / "activity.npy")
            act = np.zeros((a.shape[0], K), bool); act[:, : a.shape[1]] = a[:, :K]
            segs = sorted((json.loads(l) for l in open(Path(d) / "transcript.jsonl")), key=lambda s: s["t0"])
            for w in range(int(x.size / SR) // WIN_S):
                t0, t1 = w * WIN_S, (w + 1) * WIN_S
                ref, ids, ordered = {}, [], []
                for s in segs:
                    if s["text"] == "[backchannel]" or not (t0 <= s["t0"] < t1):
                        continue
                    words = s["text"].upper().split()
                    ref.setdefault(s["slot"], []).extend(words); ordered += words
                    ids.append(tok.tag_ids[s["slot"]])
                    ids += tok.encode("|".join(words)) + [tok.delim]
                if not ids:
                    continue
                self.items.append((x[t0 * SR:t1 * SR], act[int(t0 / FRAME_S):int(t1 / FRAME_S)], ids, ref, ordered))


class Tok:
    """wav2vec2's char vocab + four speaker tags appended."""

    def __init__(self, tokenizer):
        self.base = tokenizer
        self.n_base = len(tokenizer)
        self.tag_ids = [self.n_base + k for k in range(K)]
        self.tags = [f"<spk{k}>" for k in range(K)]
        self.delim = tokenizer.convert_tokens_to_ids("|")
        self.blank = tokenizer.pad_token_id

    def encode(self, s):
        return [i for i in self.base.convert_tokens_to_ids(list(s)) if i != self.base.unk_token_id]

    def decode(self, ids):
        out, prev = [], None
        for i in ids:
            if i != prev and i != self.blank:
                out.append(self.tags[i - self.n_base] if i >= self.n_base else self.base.convert_ids_to_tokens(int(i)))
            prev = i
        return out


# -------------------------------------------------------------------- model
class Conditioned(nn.Module):
    def __init__(self, cond: int):
        super().__init__()
        from transformers import Wav2Vec2ForCTC
        self.m = Wav2Vec2ForCTC.from_pretrained(MODEL)
        self.m.freeze_feature_encoder()
        for p in self.m.parameters():
            p.requires_grad = False
        d = self.m.config.hidden_size
        old = self.m.lm_head
        self.head = nn.Linear(d, old.out_features + K)
        with torch.no_grad():
            self.head.weight[: old.out_features] = old.weight; self.head.bias[: old.out_features] = old.bias
            self.head.weight[old.out_features:].normal_(0, 0.02); self.head.bias[old.out_features:] = -2.0
        # only the four new tag rows may change: mask gradients of the pretrained rows
        n0 = old.out_features
        self.head.weight.register_hook(lambda g: torch.cat([torch.zeros_like(g[:n0]), g[n0:]]))
        self.head.bias.register_hook(lambda g: torch.cat([torch.zeros_like(g[:n0]), g[n0:]]))
        self.cond_in = DiarizationConditioning(d, K) if cond else None
        self.per_layer = PerLayerStateConditioning(d, self.m.config.num_hidden_layers) if cond == 2 else None
        self._act50 = None
        enc = self.m.wav2vec2.encoder
        enc.register_forward_pre_hook(self._pre_encoder, with_kwargs=True)
        if self.per_layer is not None:
            for i, layer in enumerate(enc.layers):
                layer.register_forward_pre_hook(self._make_layer_hook(i), with_kwargs=True)

    def _pre_encoder(self, module, args, kwargs):
        h = kwargs.get("hidden_states", args[0] if args else None)
        if self.cond_in is not None and self._act50 is not None:
            act = self._act50[:, : h.shape[1]]
            if act.shape[1] < h.shape[1]:
                act = torch.cat([act, torch.zeros(act.shape[0], h.shape[1] - act.shape[1], K, dtype=act.dtype, device=act.device)], 1)
            h = h + self.cond_in(act)
            self._state = PerLayerStateConditioning.frame_state(act)
        if "hidden_states" in kwargs:
            kwargs["hidden_states"] = h; return args, kwargs
        return (h,) + tuple(args[1:]), kwargs

    def _make_layer_hook(self, i):
        def hook(module, args, kwargs):
            h = kwargs.get("hidden_states", args[0] if args else None)
            h = self.per_layer(h, i, self._state)
            if "hidden_states" in kwargs:
                kwargs["hidden_states"] = h; return args, kwargs
            return (h,) + tuple(args[1:]), kwargs
        return hook

    def forward(self, audio, act80):
        # act80 (B, T80, K) -> 50 Hz for the conv-frame timeline (20 ms hop, ~1499 frames per 30 s)
        self._act50 = act80.repeat_interleave(4, dim=1).to(audio.device)
        x = (audio - audio.mean(1, keepdim=True)) / (audio.std(1, keepdim=True) + 1e-5)
        h = self.m.wav2vec2(x).last_hidden_state
        return self.head(h)                                                      # (B, T50, V+K)


def cp_wer(ref, hyp_tokens, tok, ordered=None):
    import jiwer
    hyp, cur, buf = {}, None, []
    def flush():
        if cur is not None and buf:
            hyp.setdefault(cur, []).extend("".join(buf).replace("|", " ").split())
    for t in hyp_tokens:
        if t in tok.tags:
            flush(); cur = tok.tags.index(t); buf = []
        else:
            buf.append(t)
    flush()
    err = n = 0
    for k, r in ref.items():
        h = hyp.get(k, [])
        if not h:
            err += len(r)
        else:
            o = jiwer.process_words(" ".join(r), " ".join(h)); err += o.substitutions + o.deletions + o.insertions
        n += len(r)
    for k, h in hyp.items():
        if k not in ref:
            err += len(h)
    plain = "".join(t for t in hyp_tokens if t not in tok.tags).replace("|", " ").split()
    refw = ordered if ordered is not None else [w for v in ref.values() for w in v]   # time order for the agnostic WER
    o = jiwer.process_words(" ".join(refw), " ".join(plain) or "-")
    return err, n, o.substitutions + o.deletions + o.insertions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", type=int, default=1); ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=4); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rooms", default="rooms"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rooms", type=int, default=320)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    from transformers import Wav2Vec2CTCTokenizer
    tok = Tok(Wav2Vec2CTCTokenizer.from_pretrained(MODEL))
    rooms = sorted(glob.glob(str(ROOT / a.rooms / "room*")))[: a.max_rooms]
    n_dev = max(30, len(rooms) // 10)
    tr, dv = Windows(rooms[:-n_dev], tok), Windows(rooms[-n_dev:], tok)
    print(f"[cond={a.cond}] {len(tr.items)} train windows, {len(dv.items)} dev windows", flush=True)
    m = Conditioned(a.cond).to(DEV)
    params = [p for p in m.parameters() if p.requires_grad]
    print(f"[cond={a.cond}] trainable params {sum(p.numel() for p in params) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    ctc = nn.CTCLoss(blank=tok.blank, zero_infinity=True)
    g = torch.Generator().manual_seed(a.seed)
    tag = f"w2v_{a.rooms}_cond{a.cond}_s{a.steps}"
    ckpt = ROOT / f"toy_ckpt_{tag}.pt"; start = 1
    if ckpt.exists():
        st = torch.load(ckpt, map_location=DEV); m.load_state_dict(st["model"], strict=False)
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); start = st["step"] + 1
        print(f"[cond={a.cond}] resumed from step {st['step']}", flush=True)
    t0 = time.perf_counter(); m.train()
    for step in range(start, a.steps + 1):
        batch = random.sample(tr.items, a.bs)
        audio = torch.from_numpy(np.stack([b[0] for b in batch])).to(DEV)
        act = torch.from_numpy(np.stack([b[1] for b in batch]))
        if a.cond:
            act = corrupt_activity(act, generator=g)
        logits = m(audio, act)
        logp = logits.log_softmax(-1).transpose(0, 1)                          # (T,B,V)
        y = [torch.tensor(b[2]) for b in batch]
        loss = ctc(logp.float().cpu(), torch.cat(y), torch.full((a.bs,), logp.shape[0], dtype=torch.long),
                   torch.tensor([len(t) for t in y]))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        if step % 50 == 0 or step == a.steps:
            print(f"[cond={a.cond}] step {step} loss {float(loss):.3f} ({time.perf_counter() - t0:.0f}s)", flush=True)
        if step % 250 == 0 or step == a.steps:
            torch.save({"model": {k: v for k, v in m.state_dict().items() if not k.startswith("m.")},
                        "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, ckpt)
    m.eval(); err = n = plain = 0
    with torch.no_grad():
        for audio, act, _, ref, ordered in dv.items:
            logits = m(torch.from_numpy(audio)[None].to(DEV), torch.from_numpy(act)[None])
            e, k, pe = cp_wer(ref, tok.decode(logits[0].argmax(-1).tolist()), tok, ordered); err += e; n += k; plain += pe
    res = {"cond": a.cond, "steps": a.steps, "rooms": a.rooms, "dev_cpwer": round(err / max(n, 1), 4),
           "dev_wer_speaker_agnostic": round(plain / max(n, 1), 4), "dev_windows": len(dv.items),
           "train_windows": len(tr.items), "trainable_params": sum(p.numel() for p in params),
           "wall_s": round(time.perf_counter() - t0)}
    print(json.dumps(res), flush=True)
    (ROOT / f"toy_result_{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
