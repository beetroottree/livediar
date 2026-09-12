"""Room simulator: exact multi-speaker labels from single-speaker corpora (PLAN.md §4.2).

Given a pool of single-speaker utterances (path, speaker id, transcript), it
samples a room of 2..6 speakers, draws a turn-taking schedule from a small
Markov model with AMI-like statistics (turn length, gap, overlap rate,
backchannels), applies per-speaker gain and an optional room impulse
response, mixes, and emits:

    audio.wav          16 kHz mono mixture
    activity.npy       (T, K) bool, 80 ms frames   <- exact conditioning target
    transcript.jsonl   serialized-output-training order: {t0, t1, slot, spk, text}
    meta.json          speakers, gains, schedule parameters

The schedule model is deliberately simple and *parametrised*: fit the
parameters to AMI turn statistics (bench/ami has the references) and the
simulator matches the real overlap rate instead of the usual LibriMix-style
fully overlapped mixtures, which are far from meetings.

Usage:
    python research/simulate.py pool.jsonl out_dir --n 100 --seed 0
pool.jsonl lines: {"path": "...wav", "spk": "id", "text": "..."}   (16 kHz mono)
"""

from __future__ import annotations

import argparse
import json
import random
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR, FRAME_S = 16000, 0.08


@dataclass
class Schedule:
    turn_s: tuple = (0.8, 6.0)      # log-uniform turn length range (AMI median ~2 s)
    gap_s: tuple = (-1.5, 1.2)      # negative gap = overlap at the handover
    overlap_p: float = 0.35         # probability the next turn starts before this one ends
    # defaults give ~8-12% overlapped frames on the sample pool; fit to AMI (PLAN.md §4.2)
    backchannel_p: float = 0.12     # short "mm-hm" from a listener inside a long turn
    backchannel_s: tuple = (0.2, 0.6)
    self_continue_p: float = 0.35   # same speaker keeps the floor after a pause


def read(path):
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def write(path, x):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def simulate(pool, out: Path, n_spk: int, dur_s: float, sched: Schedule, rng: random.Random,
             rir: np.ndarray | None = None):
    by_spk = {}
    for u in pool:
        by_spk.setdefault(u["spk"], []).append(u)
    spks = rng.sample([s for s in by_spk if len(by_spk[s]) >= 3], n_spk)
    gains = [10 ** (rng.uniform(-6, 0) / 20) for _ in spks]        # up to 6 dB apart
    T = int(dur_s / FRAME_S)
    mix = np.zeros(int(dur_s * SR) + SR * 8, np.float32)
    activity = np.zeros((T + 100, n_spk), bool)
    segs = []
    t, cur = 0.0, rng.randrange(n_spk)
    while t < dur_s:
        u = rng.choice(by_spk[spks[cur]])
        a = read(u["path"]) * gains[cur]
        want = np.exp(rng.uniform(np.log(sched.turn_s[0]), np.log(sched.turn_s[1])))
        a = a[: int(want * SR)]
        s0 = int(t * SR); mix[s0:s0 + a.size] += a
        f0, f1 = int(t / FRAME_S), int((t + a.size / SR) / FRAME_S) + 1
        activity[f0:f1, cur] = True
        segs.append({"t0": round(t, 2), "t1": round(t + a.size / SR, 2), "slot": cur, "spk": spks[cur],
                     "text": u["text"]})
        # backchannel from someone else inside this turn
        if a.size / SR > 2.0 and rng.random() < sched.backchannel_p and n_spk > 1:
            o = rng.choice([k for k in range(n_spk) if k != cur])
            b = read(rng.choice(by_spk[spks[o]])["path"]) * gains[o]
            b = b[: int(rng.uniform(*sched.backchannel_s) * SR)]
            bt = t + rng.uniform(0.5, a.size / SR - 0.5)
            b0 = int(bt * SR); mix[b0:b0 + b.size] += b
            activity[int(bt / FRAME_S):int((bt + b.size / SR) / FRAME_S) + 1, o] = True
            segs.append({"t0": round(bt, 2), "t1": round(bt + b.size / SR, 2), "slot": o, "spk": spks[o],
                         "text": "[backchannel]"})
        gap = rng.uniform(*sched.gap_s) if rng.random() < sched.overlap_p else abs(rng.uniform(0, sched.gap_s[1]))
        t = t + a.size / SR + gap
        if not (rng.random() < sched.self_continue_p):
            cur = rng.choice([k for k in range(n_spk) if k != cur]) if n_spk > 1 else cur
    n = int(dur_s * SR)
    mix = mix[:n]
    if rir is not None:
        mix = np.convolve(mix, rir)[:n]
    mix += np.random.default_rng(rng.randrange(1 << 30)).normal(0, 10 ** (rng.uniform(-60, -35) / 20), n)
    out.mkdir(parents=True, exist_ok=True)
    write(out / "audio.wav", mix)
    np.save(out / "activity.npy", activity[:T])
    (out / "transcript.jsonl").write_text("\n".join(json.dumps(s) for s in sorted(segs, key=lambda s: s["t0"])))
    (out / "meta.json").write_text(json.dumps({"speakers": spks, "gains_db": [round(20 * np.log10(g), 1) for g in gains],
                                              "n_spk": n_spk, "dur_s": dur_s, "schedule": sched.__dict__}))
    ov = (activity[:T].sum(1) > 1).mean()
    return {"dur_s": dur_s, "n_spk": n_spk, "overlap_frac": round(float(ov), 3), "segments": len(segs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dur", type=float, default=120.0)
    ap.add_argument("--spk", default="2,6", help="min,max speakers per room")
    a = ap.parse_args()
    pool = [json.loads(l) for l in open(a.pool)]
    rng = random.Random(a.seed); lo, hi = map(int, a.spk.split(","))
    for i in range(a.n):
        r = simulate(pool, Path(a.out) / f"room{i:05d}", rng.randint(lo, hi), a.dur, Schedule(), rng)
        print(i, r)


if __name__ == "__main__":
    main()
