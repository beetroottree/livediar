"""Stage-1 gate: Dixtral (DiCoW encoder + Voxtral 3B) driven by *Sortformer* masks on AMI.

Reads the cached Sortformer probabilities from bench/ami/cache, runs the same
TurnTracker the live server uses (cap 4), converts each slot's activity into
Dixtral's STNO masks at 50 Hz, transcribes every slot window by window, and
scores cpWER against the AMI references with the benchmark's own scorer — so
the number is directly comparable with the Whisper pipeline's cpWER on the
same meeting.

    PYTHONPATH=.:bench:research/dixtral_repo/src research/.venv-dixtral/bin/python \
        research/dixtral_ami.py IS1009a --win 120

Windows: Dixtral generates at most 1024 tokens per call, so long meetings are
cut into `--win` second windows (multiple of 30 s); masks are clipped per
window; texts are concatenated per slot in time order.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "research" / "dixtral_repo" / "src"))
import ami_bench as B                     # noqa: E402
from livediar.turns import TurnTracker   # noqa: E402

FPS, CHUNK_FRAMES, SR = 50, 1500, 16000
MODEL = os.environ.get("DIXTRAL_MODEL", "BUT-FIT/Dixtral")
BASE = "mistralai/Voxtral-Mini-3B-2507"


def slot_segments(m):
    """Per-slot (start, end) segments from cached Sortformer probs via the live tracker."""
    z = np.load(B.CACHE / f"{m}.probs.npz"); P, T = z["probs"], int(z["chunk_len"])
    tr = TurnTracker(n_spk=4, max_spk=4); open_t, segs = {}, []
    for k in range(0, P.shape[0], T):
        _, events, _ = tr.update(P[k:k + T], k * B.FRAME_S)
        for ev in events:
            if ev.event == "start":
                open_t[ev.spk] = ev.t
            elif ev.spk in open_t:
                segs.append((open_t.pop(ev.spk), ev.t, ev.spk))
    end = P.shape[0] * B.FRAME_S
    segs += [(t, end, s) for s, t in open_t.items()]
    return sorted(segs), end


def stno(diar: torch.Tensor, k: int) -> torch.Tensor:
    """[K, T] binary -> [T, 4] silence/target/non-target/overlap (Dixtral demo, verbatim logic)."""
    others = torch.ones(diar.shape[0], dtype=torch.bool); others[k] = False
    sil = (1 - diar).prod(0)
    anyone_else = (1 - diar[others]).prod(0)
    target = diar[k] * anyone_else
    non_target = (1 - diar[k]) * (1 - anyone_else)
    overlap = diar[k] - target
    return torch.stack([sil, target, non_target, overlap], 0).T


def chunks(st: torch.Tensor, n: int) -> torch.Tensor:
    total = n * CHUNK_FRAMES
    if st.shape[0] < total:
        pad = torch.zeros(total - st.shape[0], 4); pad[:, 0] = 1.0
        st = torch.cat([st, pad], 0)
    return st[:total].reshape(n, CHUNK_FRAMES, 4).permute(0, 2, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meeting"); ap.add_argument("--win", type=int, default=120)
    ap.add_argument("--device", default="auto"); ap.add_argument("--max-new", type=int, default=1024)
    a = ap.parse_args()
    dev = torch.device(a.device if a.device != "auto" else
                       ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    from transformers import AutoModel, AutoProcessor
    t0 = time.perf_counter()
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16).to(dev).eval()
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model.set_tokenizer(proc.tokenizer)
    print(f"[{a.meeting}] model loaded on {dev} in {time.perf_counter() - t0:.0f}s", flush=True)

    m = a.meeting
    pcm = B.read_wav(m)
    segs, end = slot_segments(m)
    K = 4
    hyp = {k: [] for k in range(K)}
    n_win = math.ceil(end / a.win)
    t_gen = time.perf_counter()
    for w in range(n_win):
        ws, we = w * a.win, min((w + 1) * a.win, end)
        audio = pcm[int(ws * SR):int(we * SR)]
        if audio.size < SR:
            continue
        T = len(audio) // (SR // FPS)
        diar = torch.zeros(K, T)
        present = set()
        for s0, s1, k in segs:
            lo, hi = max(s0, ws), min(s1, we)
            if hi - lo > 0.05:
                diar[k, round((lo - ws) * FPS):round((hi - ws) * FPS)] = 1.0; present.add(k)
        present = sorted(present)
        if not present:
            continue
        prompt = proc.apply_transcription_request(language="en", sampling_rate=SR, audio=[audio],
                                                  model_id=BASE, format=["WAV"])
        n_chunks = prompt["input_features"].shape[0]
        stno_batch = torch.cat([chunks(stno(diar, k), n_chunks) for k in present], 0).to(dev, dtype=torch.bfloat16)
        batch = {}
        for key, v in prompt.items():
            if isinstance(v, torch.Tensor):
                v = v.repeat(len(present), *([1] * (v.dim() - 1)))
                v = v.to(dev, dtype=torch.bfloat16) if v.is_floating_point() else v.to(dev)
            batch[key] = v
        with torch.no_grad():
            gen = model.generate(**batch, stno_mask=stno_batch, max_new_tokens=a.max_new)
        L = prompt["input_ids"].shape[1]
        for i, k in enumerate(present):
            txt = proc.tokenizer.decode(gen[i, L:], skip_special_tokens=True).strip()
            hyp[k].append(txt)
        el = time.perf_counter() - t_gen
        print(f"[{m}] window {w + 1}/{n_win} ({ws:.0f}-{we:.0f}s) speakers {present} | {el:.0f}s elapsed, "
              f"rtf {el / we:.2f}", flush=True)

    ref = {}
    for s0, s1, spk, word in B.ref_words(m):
        ref.setdefault(spk, []).append(word)
    ref = {k: B.norm(" ".join(v)) for k, v in ref.items()}
    hyp_n = {k: B.norm(" ".join(v)) for k, v in hyp.items() if v}
    cpw, n_ref = B.cp_wer(ref, hyp_n)
    res = {"meeting": m, "model": MODEL, "masks": "sortformer+tracker(cap4)", "win_s": a.win,
           "cpwer": round(cpw, 4), "ref_words": n_ref, "hyp_words": sum(len(v) for v in hyp_n.values()),
           "gen_wall_s": round(time.perf_counter() - t_gen), "rtf": round((time.perf_counter() - t_gen) / end, 3),
           "device": str(dev)}
    out = B.RES / f"{m}.dixtral.json"; out.write_text(json.dumps(res, indent=1))
    (B.RES / f"{m}.dixtral.hyp.txt").write_text("\n".join(f"slot{k}: {' '.join(v)}" for k, v in hyp.items() if v))
    print(f"[{m}/dixtral] cpWER {cpw * 100:.1f}%  (rtf {res['rtf']})", flush=True)


if __name__ == "__main__":
    main()
