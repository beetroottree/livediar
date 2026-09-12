"""Build the Moshi-LoRA stage-1 pilot dataset (PLAN.md §8 job 6) in moshi-finetune's format.

moshi-finetune expects, per example: a stereo wav (left = Moshi's channel,
right = the user's channel) at 24 kHz, a sibling .json with
{"alignments": [[word, [start, end], "SPEAKER_MAIN"], ...]}, and a .jsonl
listing {"path", "duration"}. The text stream is trained on the main
speaker's words at their timestamps.

Stage-1 repurposing ("room listening"): the *room mixture* goes on the user
channel, Moshi's channel is silence, and the alignments are the room's
speaker-attributed words - every word tagged SPEAKER_MAIN so it lands on the
text stream, with a "<spk k>" tag word inserted at each turn start so the
monologue is who-said-what. A sibling .activity.npy holds the (T, K) 80 ms
activity matrix the conditioning module reads (research/conditioning.py).

Sources:
  simulated rooms  research/data/rooms*/room*/{audio.wav, activity.npy, transcript.jsonl}
  AMI train/dev    bench/ami/wav/<m>.wav + references (exact per-word alignments from NXT)

    python research/pilot_data.py rooms research/data/rooms2k research/pilot_data/rooms
    python research/pilot_data.py ami   research/pilot_data/ami   # needs AMI train wavs downloaded
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
SR_IN, SR_OUT, FRAME_S = 16000, 24000, 0.08


def read_wav(p):
    with wave.open(str(p), "rb") as w:
        assert w.getframerate() == SR_IN and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def write_stereo_24k(p, room16k):
    room = resample_poly(room16k, 3, 2).astype(np.float32)          # 16 k -> 24 k
    left = np.zeros_like(room)                                        # Moshi silent in stage 1
    st = np.stack([left, room], 1)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR_OUT)
        w.writeframes((np.clip(st, -1, 1) * 32767).astype(np.int16).tobytes())
    return room.size / SR_OUT


def alignments_from_segments(segs, words_per_seg=None):
    """segs: [(t0, t1, slot, text)] -> moshi-finetune alignments with <spk k> tag words.
    Word timestamps are spread uniformly across the segment when not given."""
    out = []
    for t0, t1, slot, text in sorted(segs):
        ws = text.split()
        if not ws:
            continue
        out.append([f"<spk{slot}>", [round(t0, 3), round(min(t0 + 0.08, t1), 3)], "SPEAKER_MAIN"])
        step = (t1 - t0) / len(ws)
        for i, w in enumerate(ws):
            out.append([w, [round(t0 + i * step, 3), round(t0 + (i + 1) * step, 3)], "SPEAKER_MAIN"])
    return out


def from_rooms(src, dst):
    dst = Path(dst); dst.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in sorted(glob.glob(f"{src}/room*")):
        d = Path(d); name = d.name
        segs = []
        for line in open(d / "transcript.jsonl"):
            s = json.loads(line)
            if s["text"] != "[backchannel]":
                segs.append((s["t0"], s["t1"], s["slot"], s["text"].lower()))
        dur = write_stereo_24k(dst / f"{name}.wav", read_wav(d / "audio.wav"))
        (dst / f"{name}.json").write_text(json.dumps({"alignments": alignments_from_segments(segs)}))
        act = np.load(d / "activity.npy")
        np.save(dst / f"{name}.activity.npy", act)
        rows.append({"path": f"{name}.wav", "duration": round(dur, 3)})
    (dst / "data.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"rooms -> {dst}: {len(rows)} files, {sum(r['duration'] for r in rows) / 3600:.1f} h")


def from_ami(dst, split="train"):
    """AMI train/dev meetings with exact NXT word alignments and reference activity.
    Slots are assigned by arrival order of the reference speakers (K <= 4)."""
    import ami_bench as B
    dst = Path(dst); dst.mkdir(parents=True, exist_ok=True)
    rows = []
    rttm_dir = B.D / "setup" / "only_words" / "rttms" / split
    for rttm in sorted(rttm_dir.glob("*.rttm")):
        m = rttm.stem
        wav = B.D / "wav" / f"{m}.wav"
        if not wav.exists():
            continue
        words = B.ref_words(m)
        order = []
        for _, _, spk, _ in words:
            if spk not in order:
                order.append(spk)
        slot = {s: i for i, s in enumerate(order[:4])}
        pcm = read_wav(wav)
        T = int(pcm.size / SR_IN / FRAME_S) + 1
        act = np.zeros((T, 4), bool)
        for line in open(rttm):
            p = line.split()
            if p and p[0] == "SPEAKER" and p[7] in slot:
                s0, dur = float(p[3]), float(p[4])
                act[int(s0 / FRAME_S):int((s0 + dur) / FRAME_S) + 1, slot[p[7]]] = True
        # word-level alignments straight from NXT, with a tag word at each speaker change
        al, last = [], None
        for s0, s1, spk, w in words:
            if spk not in slot:
                continue
            if spk != last:
                al.append([f"<spk{slot[spk]}>", [round(s0, 3), round(s0 + 0.08, 3)], "SPEAKER_MAIN"]); last = spk
            al.append([w.lower(), [round(s0, 3), round(max(s1, s0 + 0.02), 3)], "SPEAKER_MAIN"])
        dur = write_stereo_24k(dst / f"{m}.wav", pcm)
        (dst / f"{m}.json").write_text(json.dumps({"alignments": al}))
        np.save(dst / f"{m}.activity.npy", act)
        rows.append({"path": f"{m}.wav", "duration": round(dur, 3)})
    (dst / "data.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"ami/{split} -> {dst}: {len(rows)} meetings, {sum(r['duration'] for r in rows) / 3600:.1f} h")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("kind", choices=["rooms", "ami"])
    ap.add_argument("a"); ap.add_argument("b", nargs="?")
    x = ap.parse_args()
    if x.kind == "rooms":
        from_rooms(x.a, x.b)
    else:
        from_ami(x.a, split=x.b or "train")
