"""AMI test-set benchmark for livediar: DER + speaker-attributed WER, with ground truth.

Data (bench/ami/):
  wav/<m>.wav                     AMI Mix-Headset (16 kHz mono), 16 test meetings
  setup/only_words/rttms/test     pyannote AMI-diarization-setup references
  setup/uems/test                 scoring regions
  manual/words/<m>.<A-D>.words.xml  NXT word-level transcripts (speaker-attributed)
  manual/corpusResources/meetings.xml  agent letter -> global speaker name

Stages:
  python bench/ami_bench.py diarize <meeting>          # Sortformer once -> cache/<m>.probs.npz
  python bench/ami_bench.py eval <meeting> <condition> # from cache: DER + cpWER -> results/<m>.<cond>.json
  python bench/ami_bench.py report                     # aggregate -> results/RESULTS.md

Conditions: none (raw mix captions), words (default overlap attribution),
            enrolled (words + voices enrolled from 10 s of each speaker's clean speech).
Everything runs through the same TurnTracker / Captioner / Whisper code the live
server uses; only the Sortformer step is replayed from cache.
"""

import asyncio
import itertools
import json
import re
import sys
import time
import wave
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
D = ROOT / "bench" / "ami"
CACHE, RES = D / "cache", D / "results"
CACHE.mkdir(exist_ok=True), RES.mkdir(exist_ok=True)
SR, FRAME_S = 16000, 0.08
MEETINGS = ["EN2002a", "EN2002b", "EN2002c", "EN2002d", "ES2004a", "ES2004b", "ES2004c", "ES2004d",
            "IS1009a", "IS1009b", "IS1009c", "IS1009d", "TS3003a", "TS3003b", "TS3003c", "TS3003d"]


# ----------------------------------------------------------------- references
def read_wav(m):
    with wave.open(str(D / "wav" / f"{m}.wav"), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1, "expected 16 kHz mono"
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


def agent_map(m):
    """agent letter -> global speaker name (matches the RTTM references)."""
    root = ET.parse(D / "manual" / "corpusResources" / "meetings.xml").getroot()
    for meet in root.iter():
        if meet.get("observation") == m:
            return {s.get("nxt_agent"): s.get("global_name") for s in meet if s.get("nxt_agent")}
    raise KeyError(m)


def ref_words(m):
    """[(start, end, speaker_global_name, word)] sorted by start; vocal sounds dropped."""
    out = []
    amap = agent_map(m)
    for agent, name in amap.items():
        f = D / "manual" / "words" / f"{m}.{agent}.words.xml"
        if not f.exists():
            continue
        for el in ET.parse(f).getroot():
            if el.tag.endswith("}w") or el.tag == "w":
                if el.get("starttime") is None or not (el.text or "").strip():
                    continue
                if el.get("punc") == "true":
                    continue
                out.append((float(el.get("starttime")), float(el.get("endtime") or el.get("starttime")),
                            name, el.text.strip()))
    return sorted(out)


def ref_rttm(m):
    from pyannote.core import Annotation, Segment
    ann = Annotation(uri=m)
    for line in (D / "setup" / "only_words" / "rttms" / "test" / f"{m}.rttm").read_text().splitlines():
        p = line.split()
        if p and p[0] == "SPEAKER":
            ann[Segment(float(p[3]), float(p[3]) + float(p[4]))] = p[7]
    return ann


def uem(m):
    from pyannote.core import Segment, Timeline
    tl = Timeline(uri=m)
    for line in (D / "setup" / "uems" / "test" / f"{m}.uem").read_text().splitlines():
        p = line.split()
        if len(p) >= 4:
            tl.add(Segment(float(p[2]), float(p[3])))
    return tl


# ---------------------------------------------------------------- diarization
def diarize(m):
    import os
    import torch
    # Several workers run at once; torch's default thread pool (= all cores) per
    # process oversubscribes the machine. Measured: 5 workers x default threads
    # gave 1.88x realtime per worker vs 0.64x for a lone process.
    torch.set_num_threads(int(os.environ.get("LIVEDIAR_THREADS", "4")))
    from livediar.engine_sortformer import SortformerEngine
    from livediar.presets import PRESETS
    pcm = read_wav(m)
    eng = SortformerEngine(PRESETS["low"])
    blocks, wall = [], 0.0
    t_all = time.perf_counter()
    for i in range(0, pcm.size, SR // 2):
        for st in eng.feed(pcm[i:i + SR // 2]):
            blocks.append(st.probs); wall += st.step_wall_s
    for st in eng.finalize():
        blocks.append(st.probs); wall += st.step_wall_s
    P = np.concatenate(blocks)
    np.savez_compressed(CACHE / f"{m}.probs.npz", probs=P, chunk_len=PRESETS["low"].chunk_len,
                        step_wall_s=wall, total_wall_s=time.perf_counter() - t_all, audio_s=pcm.size / SR)
    print(f"[{m}] diarized {pcm.size / SR / 60:.1f} min in {time.perf_counter() - t_all:.0f}s "
          f"(rtf {wall / (pcm.size / SR):.3f}) -> {P.shape}")


class CachedEngine:
    """Replays cached Sortformer probabilities with the engine interface."""
    kind, n_spk = "cached", 4

    def __init__(self, probs, chunk_len):
        from livediar.engine_mock import StepOut
        self.StepOut, self.P, self.T = StepOut, probs, int(chunk_len)
        self.buf_s, self.k = 0.0, 0

    def feed(self, pcm):
        self.buf_s += pcm.size / SR
        out = []
        hop = self.T * FRAME_S
        while (self.k + 1) * hop <= self.buf_s and self.k * self.T < self.P.shape[0]:
            blk = self.P[self.k * self.T:(self.k + 1) * self.T]
            out.append(self.StepOut(self.k * hop, blk, 0.0)); self.k += 1
        return out

    def finalize(self):
        out = []
        if self.k * self.T < self.P.shape[0]:
            out.append(self.StepOut(self.k * self.T * FRAME_S, self.P[self.k * self.T:], 0.0))
            self.k = 10 ** 9
        return out


# --------------------------------------------------------------------- scoring
_norm = None


def norm(text):
    global _norm
    if _norm is None:
        from whisper_normalizer.english import EnglishTextNormalizer
        _norm = EnglishTextNormalizer()
    toks = [t for t in text.split() if not t.endswith("-")]     # drop word fragments ("th-")
    return _norm(" ".join(toks)).split()


def wer_counts(ref, hyp):
    """(errors, ref_len) with jiwer; handles empty sides."""
    if not ref and not hyp:
        return 0, 0
    if not ref:
        return len(hyp), 0
    if not hyp:
        return len(ref), len(ref)
    import jiwer
    o = jiwer.process_words(" ".join(ref), " ".join(hyp))
    return o.substitutions + o.deletions + o.insertions, len(ref)


def cp_wer(ref_by_spk, hyp_by_spk):
    """Concatenated minimum-permutation WER. Extra hyp speakers count as insertions."""
    R, H = list(ref_by_spk), list(hyp_by_spk)
    n_ref = sum(len(v) for v in ref_by_spk.values())
    best = None
    slots = R + [None] * max(0, len(H) - len(R))
    for perm in itertools.permutations(slots, len(H)):
        err = 0
        used = set(p for p in perm if p is not None)
        for h, r in zip(H, perm):
            e, _ = wer_counts(ref_by_spk.get(r, []) if r else [], hyp_by_spk[h]); err += e
        for r in R:                       # reference speakers left unmapped: all deletions
            if r not in used:
                err += len(ref_by_spk[r])
        best = err if best is None else min(best, err)
    return (best or 0) / max(n_ref, 1), n_ref


def id_wer(ref_by_spk, hyp_by_name):
    """Fixed-identity WER: hypothesis keyed by the *identified* name (enrolled condition)."""
    err, n = 0, 0
    for r, words in ref_by_spk.items():
        e, k = wer_counts(words, hyp_by_name.get(r, [])); err += e; n += k
    for h, words in hyp_by_name.items():
        if h not in ref_by_spk:
            err += len(words)
    return err / max(n, 1)


def der(m, hyp_segments):
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate
    hyp = Annotation(uri=m)
    for s0, s1, spk in hyp_segments:
        hyp[Segment(s0, s1)] = f"spk{spk}"
    metric = DiarizationErrorRate(collar=0.25)
    d = metric(ref_rttm(m), hyp, uem=uem(m), detailed=True)   # components are seconds
    tot = float(d["total"]) or 1e-9
    return {"der": float(d["diarization error rate"]), "miss": float(d["missed detection"]) / tot,
            "fa": float(d["false alarm"]) / tot, "conf": float(d["confusion"]) / tot, "total": tot}


# ------------------------------------------------------------------- evaluate
def enrollment_audio(m, pcm, secs=10.0):
    """Per reference speaker: up to `secs` of their non-overlapped speech (from the mix)."""
    ann = ref_rttm(m)
    out = {}
    for spk in ann.labels():
        own = ann.label_timeline(spk)
        others = ann.subset([spk], invert=True).get_timeline()
        clean = own.extrude(others.support())
        chunks, got = [], 0.0
        for seg in clean:
            if seg.duration < 1.0:
                continue
            a = pcm[int(seg.start * SR):int(seg.end * SR)]
            chunks.append(a); got += seg.duration
            if got >= secs:
                break
        if got >= 3.0:
            out[spk] = np.concatenate(chunks)
    return out


def evaluate(m, cond):
    from livediar.asr import ActivityRing, AudioRing, Captioner, build_asr, build_overlap
    from livediar.turns import TurnTracker
    z = np.load(CACHE / f"{m}.probs.npz")
    pcm = read_wav(m)
    eng = CachedEngine(z["probs"], z["chunk_len"])
    tracker = TurnTracker(n_spk=4, max_spk=4)
    asr = build_asr("whisper", None, "en")
    mode, bank, sep = build_overlap("none" if cond == "none" else "words")
    if cond == "enrolled":
        enr = enrollment_audio(m, pcm)
        for name, audio in enr.items():
            bank.add_person(name, name, audio)
        print(f"[{m}] enrolled {list(enr)} ({[round(a.size / SR, 1) for a in enr.values()]} s)")
    ring = AudioRing(keep_s=pcm.size / SR + 1); ring.push(pcm)
    caps = {}
    idents = []

    async def emit(msg):
        if msg["type"] == "caption":
            caps[msg["id"]] = msg
        elif msg["type"] == "identify":
            idents.append(msg)

    cap = Captioner(asr, ring, emit, activity=ActivityRing(4), mode=mode, bank=bank, separator=sep)
    open_t, segs = {}, []
    t_all = time.perf_counter()

    async def run():
        cap.start()
        for i in range(0, pcm.size, SR // 2):
            for st in eng.feed(pcm[i:i + SR // 2]):
                _, events, smoothed = tracker.update(st.probs, st.t0)
                cap.on_frames(smoothed, tracker.last_active_frames)
                for ev in events:
                    if ev.event == "start":
                        open_t[ev.spk] = ev.t
                    elif ev.spk in open_t:
                        segs.append((open_t.pop(ev.spk), ev.t, ev.spk))
                cap.on_events(events, st.t0 + st.probs.shape[0] * FRAME_S)
        for st in eng.finalize():
            _, events, smoothed = tracker.update(st.probs, st.t0)
            cap.on_frames(smoothed, tracker.last_active_frames)
            for ev in events:
                if ev.event == "start":
                    open_t[ev.spk] = ev.t
                elif ev.spk in open_t:
                    segs.append((open_t.pop(ev.spk), ev.t, ev.spk))
            cap.on_events(events, st.t0 + st.probs.shape[0] * FRAME_S)
        await cap.finish(pcm.size / SR)

    asyncio.run(run())
    end = pcm.size / SR
    for s, t in open_t.items():
        segs.append((t, end, s))

    # ---- diarization score (independent of captions)
    d = der(m, segs)

    # ---- transcript scores
    finals = sorted((c for c in caps.values() if c["final"] and c["text"]), key=lambda c: c["t0"])
    hyp_by_slot, hyp_by_name = {}, {}
    for c in finals:
        w = norm(c["text"])
        hyp_by_slot.setdefault(c["spk"], []).extend(w)
        hyp_by_name.setdefault(c.get("name") or f"slot{c['spk']}", []).extend(w)
    ref_by_spk = {}
    for s0, s1, spk, word in ref_words(m):
        ref_by_spk.setdefault(spk, []).append(word)
    ref_by_spk = {k: norm(" ".join(v)) for k, v in ref_by_spk.items()}
    cpw, n_ref = cp_wer(ref_by_spk, hyp_by_slot)
    res = {"meeting": m, "condition": cond, "audio_min": round(end / 60, 1),
           "der": d, "cpwer": round(cpw, 4), "ref_words": n_ref,
           "hyp_words": sum(len(v) for v in hyp_by_slot.values()),
           "captions": len(finals), "overlap_s": round(cap.overlap_s, 1),
           "diar_rtf": round(float(z["step_wall_s"]) / end, 3), "asr_rtf": round(cap.rtf, 3),
           "eval_wall_s": round(time.perf_counter() - t_all, 1),
           "talk_s": [round(x, 1) for x in tracker.talk_s], "turns": tracker.turns}
    if cond == "enrolled":
        res["idwer"] = round(id_wer(ref_by_spk, hyp_by_name), 4)
        res["identified"] = {str(i["spk"]): i["name"] for i in idents}
        res["verified_captions"] = sum(1 for c in finals if c.get("verified"))
        # Identity policies scored offline from the same captions:
        #   caption  = per-caption verification overrides the slot mapping (live default)
        #   slot     = final slot->person mapping only
        #   strong   = override only with >= 2.5 s clean audio and margin >= 0.15
        final_map = {str(i["spk"]): i["name"] for i in idents}
        def hyp_for(policy):
            h = {}
            for c in finals:
                slot_name = final_map.get(str(c["spk"]))
                if policy == "slot":
                    nm = slot_name
                elif policy == "strong":
                    nm = c.get("name") if (c.get("verified") and c.get("clean_s", 0) >= 2.5
                                          and c.get("margin", 0) >= 0.15) else slot_name
                else:
                    nm = c.get("name")
                h.setdefault(nm or f"slot{c['spk']}", []).extend(norm(c["text"]))
            return h
        res["idwer_policy"] = {p: round(id_wer(ref_by_spk, hyp_for(p)), 4) for p in ("caption", "slot", "strong")}
        (RES / f"{m}.{cond}.caps.json").write_text(json.dumps(
            [{k: c.get(k) for k in ("spk", "t0", "t1", "text", "name", "verified", "sim", "margin", "clean_s", "slot_person")}
             for c in finals]))
    (RES / f"{m}.{cond}.json").write_text(json.dumps(res, indent=1))
    (RES / f"{m}.{cond}.hyp.txt").write_text("\n".join(
        f"[{c['t0']:8.2f}] {c.get('name') or 'slot' + str(c['spk'])}: {c['text']}" for c in finals))
    print(f"[{m}/{cond}] DER {d['der'] * 100:.1f}%  cpWER {cpw * 100:.1f}%"
          + (f"  idWER {res['idwer'] * 100:.1f}% (slot {res['idwer_policy']['slot'] * 100:.1f}%, "
             f"strong {res['idwer_policy']['strong'] * 100:.1f}%)" if cond == "enrolled" else "")
          + f"  ({res['eval_wall_s']}s)")
    return res


# --------------------------------------------------------------------- report
def report():
    rows = [json.loads(p.read_text()) for p in sorted(RES.glob("*.json"))]
    conds = ["none", "words", "enrolled"]
    lines = ["# livediar on the AMI test set (pyannote split, Mix-Headset, 16 meetings)", "",
             "Sortformer v2.1 streaming, preset `low` (1.04 s), cap 4 speakers, CPU; Whisper "
             "large-v3-turbo (MLX) captions; TitaNet-large voice profiles. DER: collar 0.25 s, "
             "overlap scored, pyannote `only_words` references + UEM. cpWER: concatenated "
             "minimum-permutation WER over speakers, Whisper English normalisation, AMI word "
             "fragments dropped. idWER (enrolled): hypothesis keyed by the *identified* name, no "
             "permutation - wrong identity counts as error.", ""]
    # per-condition aggregate
    lines += ["| condition | meetings | audio | DER | miss | FA | conf | cpWER | idWER |", "|---|---|---|---|---|---|---|---|---|"]
    for c in conds:
        rs = [r for r in rows if r["condition"] == c]
        if not rs:
            continue
        tot = sum(r["der"]["total"] for r in rs)
        agg = {k: sum(r["der"][k] * r["der"]["total"] for r in rs) / max(tot, 1e-9) for k in ("der", "miss", "fa", "conf")}
        nref = sum(r["ref_words"] for r in rs)
        cpw = sum(r["cpwer"] * r["ref_words"] for r in rs) / max(nref, 1)
        idw = (sum(r["idwer"] * r["ref_words"] for r in rs) / max(nref, 1)) if c == "enrolled" else None
        lines.append(f"| {c} | {len(rs)} | {sum(r['audio_min'] for r in rs) / 60:.1f} h | {agg['der'] * 100:.1f}% | "
                     f"{agg['miss'] * 100:.1f}% | {agg['fa'] * 100:.1f}% | {agg['conf'] * 100:.1f}% | {cpw * 100:.1f}% | "
                     + (f"{idw * 100:.1f}%" if idw is not None else "–") + " |")
    lines += ["", "NVIDIA's published DER for this model/config on AMI test IHM: 16.67% (model card; "
              "their scoring setup may differ).", "", "## Per meeting", "",
              "| meeting | min | DER | cpWER none | cpWER words | cpWER enrolled | idWER enrolled | identified |", "|---|---|---|---|---|---|---|---|"]
    for m in MEETINGS:
        by = {r["condition"]: r for r in rows if r["meeting"] == m}
        if not by:
            continue
        any_ = next(iter(by.values()))
        e = by.get("enrolled", {})
        lines.append(f"| {m} | {any_['audio_min']} | {any_['der']['der'] * 100:.1f}% | "
                     + " | ".join((f"{by[c]['cpwer'] * 100:.1f}%" if c in by else "–") for c in conds)
                     + f" | {(e.get('idwer', 0) * 100):.1f}% | {len(e.get('identified', {}))}/4 |" if e else
                     f"| {m} | {any_['audio_min']} | {any_['der']['der'] * 100:.1f}% | "
                     + " | ".join((f"{by[c]['cpwer'] * 100:.1f}%" if c in by else "–") for c in conds) + " | – | – |")
    rs = [r for r in rows if r["condition"] == "words"] or rows
    if rs:
        lines += ["", f"Compute: diarization RTF {np.mean([r['diar_rtf'] for r in rs]):.3f} (CPU), "
                  f"captions RTF {np.mean([r['asr_rtf'] for r in rs]):.3f} (GPU), on an Apple M5 Pro."]
    (RES / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "diarize":
        diarize(sys.argv[2])
    elif cmd == "eval":
        evaluate(sys.argv[2], sys.argv[3])
    elif cmd == "report":
        report()
