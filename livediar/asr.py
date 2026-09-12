"""Speaker-attributed captions on top of the diarizer.

The diarizer decides *who*; this module decides *what they said*. Every turn
the TurnTracker closes is single-speaker by construction, so we transcribe
that audio slice and attribute it exactly - no word-to-speaker matching.
Long turns get a partial caption every PARTIAL_S seconds (re-transcribing
the uncommitted tail), and a chunk is committed early once it grows past
COMMIT_S so latency and cost stay bounded.

Backends
--------
whisper : mlx-whisper (Apple GPU). Multilingual with per-segment language
          auto-detection, so English/Mandarin code-switching works without
          configuration. Default model: mlx-community/whisper-large-v3-turbo.
none    : captions disabled.

Overlap caveat: when two people talk at once, both turns contain the mixed
audio and Whisper transcribes the dominant voice for each. A separation-aware
backend (NVIDIA multi-talker Parakeet, English only) is the upgrade path.
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

SR = 16000
FRAME_S = 0.08
PARTIAL_S = 3.0      # emit a provisional caption for an open turn this often
COMMIT_S = 14.0      # commit an open turn's text in chunks no longer than this
MERGE_GAP_S = 1.2    # same speaker resumes within this -> one caption, not two
SHORT_S = 2.5        # slices shorter than this use the session's majority language
ENROLL_S = 1.5       # clean (non-overlapped) slices at least this long train voice profiles
EXTRACT_MIN_S = 0.6  # non-dominant stretches shorter than this are not worth un-mixing
EXTRACT_MIN_P = 0.6  # Whisper word confidence floor for words taken from a separated stream
MIN_SEG_S = 0.45     # ignore slices shorter than this (clicks, laughs)
PAD_S = 0.15         # audio padding around the diarizer's turn boundaries
RING_S = 240.0       # seconds of audio kept server-side for slicing

_HALLUCINATIONS = re.compile(
    r"^(thank you\.?|thanks for watching\.?|you\.?|bye\.?|\.|-|…)$", re.I)


# ---------------------------------------------------------------- backends
class WhisperASR:
    """mlx-whisper with language detection restricted to an allowed set.

    Whisper's free auto-detect is unreliable on 1-3 s slices (it will happily
    guess Dutch for a mumbled English fragment). We run detection ourselves,
    keep only `langs` as candidates, and for short slices fall back to the
    language the session has mostly been in. `language="en"` pins it.
    """
    kind = "whisper"

    def __init__(self, model: str = "mlx-community/whisper-large-v3-turbo",
                 language: Optional[str] = None, langs: Optional[List[str]] = None):
        import mlx_whisper  # deferred: Apple-silicon only
        self._mw = mlx_whisper
        self.model = model
        self.language = language or None            # pinned language, or None
        self.langs = [l.strip() for l in (langs or ["en", "zh"]) if l.strip()]
        self.counts: Dict[str, int] = {}            # session language prior
        self._detector = None
        try:                                        # optional: restricted detection
            from mlx_whisper.load_models import load_model
            from mlx_whisper.audio import log_mel_spectrogram, pad_or_trim, N_FRAMES
            from mlx_whisper.decoding import detect_language
            m = load_model(model)
            def detect(audio):
                mel = pad_or_trim(log_mel_spectrogram(audio, n_mels=m.dims.n_mels), N_FRAMES)
                _, probs = detect_language(m, mel)
                probs = probs[0] if isinstance(probs, list) else probs
                cand = {k: probs.get(k, 0.0) for k in self.langs}
                return max(cand, key=cand.get) if cand else None
            self._detector = detect
        except Exception as e:  # noqa: BLE001
            print(f"[livediar] restricted language detection unavailable ({e}); using whisper auto")
        # warm up (loads weights, compiles kernels) so the first caption is fast
        self._mw.transcribe(np.zeros(SR, dtype=np.float32), path_or_hf_repo=model,
                            language=self.language or self.langs[0], fp16=True)

    def _pick_language(self, audio: np.ndarray) -> Optional[str]:
        if self.language:
            return self.language
        majority = max(self.counts, key=self.counts.get) if self.counts else None
        if audio.size < SHORT_S * SR and majority:
            return majority
        if self._detector is not None:
            try:
                return self._detector(audio)
            except Exception:  # noqa: BLE001
                pass
        return None

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None,
                   words: bool = False) -> dict:
        """-> {"text", "language", "words": [{"word","start","end"}] (slice-relative s)}"""
        audio = audio.astype(np.float32)
        lang = language or self._pick_language(audio)
        # Whisper flips between simplified and traditional characters at random;
        # a simplified-Chinese prompt pins it to simplified (swap for traditional if needed).
        prompt = "以下是普通话的句子。" if lang == "zh" else None
        r = self._mw.transcribe(
            audio, path_or_hf_repo=self.model, language=lang, initial_prompt=prompt,
            fp16=True, condition_on_previous_text=False, no_speech_threshold=0.6,
            temperature=(0.0, 0.2, 0.4), word_timestamps=words)
        segs = r.get("segments") or []
        keep = [s for s in segs if s.get("no_speech_prob", 0) < 0.75
                and s.get("compression_ratio", 0) < 2.4
                and s.get("avg_logprob", 0) > -1.2]
        text = " ".join(s["text"].strip() for s in keep).strip()
        if _HALLUCINATIONS.match(text):
            text, keep = "", []
        got = r.get("language") or lang
        if text and got:
            self.counts[got] = self.counts.get(got, 0) + 1
        wl = [{"word": w["word"], "start": float(w["start"]), "end": float(w["end"]),
               "p": float(w.get("probability", 1.0))}
              for s in keep for w in (s.get("words") or [])] if words else []
        return {"text": text, "language": got, "words": wl}

    @property
    def info(self) -> dict:
        return {"asr": self.kind, "asr_model": self.model.split("/")[-1],
                "language": self.language or ("auto " + "/".join(self.langs))}


def build_overlap(mode: str):
    """-> (mode, bank, separator).

    The voice-profile bank (TitaNet) is loaded for every mode but 'none': it
    drives overlap dominance and speaker identification. The separator only
    for 'extract'. Falls back gracefully if a model can't load.
    """
    if mode == "none":
        return mode, None, None
    bank = sep = None
    try:
        from .overlap import SpeakerBank
        bank = SpeakerBank()
    except Exception as e:  # noqa: BLE001
        print(f"[livediar] voice profiles unavailable ({type(e).__name__}: {e})")
    if mode == "extract":
        try:
            from .overlap import Separator
            sep = Separator()
        except Exception as e:  # noqa: BLE001
            print(f"[livediar] separator unavailable ({type(e).__name__}: {e}); using 'words'")
            mode = "words"
    return mode, bank, sep


def build_asr(kind: str, model: Optional[str] = None, language: Optional[str] = None,
              langs: Optional[List[str]] = None):
    if kind in ("none", None, ""):
        return None
    if kind == "whisper":
        return WhisperASR(model or "mlx-community/whisper-large-v3-turbo", language, langs)
    raise ValueError(f"unknown asr backend {kind!r}")


# ----------------------------------------------------------------- ring buffer
class AudioRing:
    """Rolling float32 buffer addressed by absolute stream time."""

    def __init__(self, keep_s: float = RING_S):
        self.buf = np.zeros(0, dtype=np.float32)
        self.start = 0          # absolute sample index of buf[0]
        self.keep = int(keep_s * SR)

    def push(self, pcm: np.ndarray):
        self.buf = np.concatenate([self.buf, pcm.astype(np.float32)])
        if self.buf.size > self.keep:
            drop = self.buf.size - self.keep
            self.buf = self.buf[drop:]
            self.start += drop

    @property
    def end_s(self) -> float:
        return (self.start + self.buf.size) / SR

    def slice(self, t0: float, t1: float) -> np.ndarray:
        a = max(int(t0 * SR), self.start) - self.start
        b = min(int(t1 * SR), self.start + self.buf.size) - self.start
        return self.buf[max(a, 0):max(b, 0)]


class ActivityRing:
    """Per-frame (80 ms) diarizer output aligned to the AudioRing's clock."""

    def __init__(self, n_spk: int = 4, keep_s: float = RING_S):
        self.n = n_spk
        self.probs = np.zeros((0, n_spk), dtype=np.float32)
        self.active = np.zeros((0, n_spk), dtype=bool)
        self.start = 0                          # absolute frame index of row 0
        self.keep = int(keep_s / FRAME_S)

    def push(self, probs: np.ndarray, active: np.ndarray):
        self.probs = np.concatenate([self.probs, probs.astype(np.float32)])
        self.active = np.concatenate([self.active, active.astype(bool)])
        if self.probs.shape[0] > self.keep:
            drop = self.probs.shape[0] - self.keep
            self.probs, self.active = self.probs[drop:], self.active[drop:]
            self.start += drop

    def window(self, t0: float, t1: float):
        """(probs, active, first_frame_time) covering [t0, t1); may be empty."""
        a = max(int(t0 / FRAME_S), self.start) - self.start
        b = min(int(np.ceil(t1 / FRAME_S)), self.start + self.probs.shape[0]) - self.start
        a, b = max(a, 0), max(b, 0)
        return self.probs[a:b], self.active[a:b], (self.start + a) * FRAME_S


def _norm_word(w: str) -> str:
    return re.sub(r"[^\w]", "", w.lower())


# ------------------------------------------------------------------ captioner
@dataclass
class _Open:
    spk: int
    t0: float            # start of the uncommitted region
    cap_id: int          # caption id currently being built
    last_partial: float  # stream time of the last partial we scheduled
    ended_at: Optional[float] = None   # diarizer said "end" here; waiting for a resume


@dataclass
class Captioner:
    """Turns TurnTracker events + an AudioRing into caption jobs.

    `emit(msg)` is awaited with dicts:
      {"type":"caption","id":int,"spk":int,"t0":float,"t1":float,
       "text":str,"final":bool,"lang":str|None}
    Partial and final captions share an id; the client replaces in place.
    """
    asr: object
    ring: AudioRing
    emit: Callable
    activity: Optional[ActivityRing] = None
    mode: str = "words"               # none | words | extract
    bank: object = None               # overlap.SpeakerBank (extract mode)
    separator: object = None          # overlap.Separator (extract mode)
    overlap_s: float = 0.0            # audio seconds handled by word-level attribution
    extracted_s: float = 0.0          # audio seconds un-mixed for a non-dominant speaker
    open: Dict[int, _Open] = field(default_factory=dict)
    pending: List[dict] = field(default_factory=list)   # side messages (identify) to emit
    next_id: int = 1
    queue: Optional[asyncio.Queue] = None
    worker: Optional[asyncio.Task] = None
    asr_wall_s: float = 0.0
    asr_audio_s: float = 0.0

    def start(self):
        self.queue = asyncio.Queue()
        self.worker = asyncio.create_task(self._run())

    def on_frames(self, probs: np.ndarray, active: np.ndarray):
        if self.activity is not None:
            self.activity.push(probs, active)

    # ---- overlap analysis -------------------------------------------------
    def _dominance(self, spk: int, t0: float, t1: float):
        """Per-frame: (times, others_active, spk_dominant) over [t0, t1)."""
        if self.activity is None or self.mode == "none":
            return None
        P, A, ft = self.activity.window(t0, t1)
        if P.shape[0] == 0:
            return None
        others = A.copy(); others[:, spk] = False
        others_active = others.any(axis=1)
        other_p = np.where(others, P, 0.0).max(axis=1)
        dominant = P[:, spk] >= other_p
        times = ft + np.arange(P.shape[0]) * FRAME_S
        return times, others_active, dominant

    @staticmethod
    def _runs(mask: np.ndarray, times: np.ndarray, min_s: float):
        """Contiguous True runs of `mask` as (start, end) seconds, >= min_s."""
        out, i, n = [], 0, mask.size
        while i < n:
            if mask[i]:
                j = i
                while j < n and mask[j]:
                    j += 1
                s, e = times[i], times[j - 1] + FRAME_S
                if e - s >= min_s:
                    out.append((s, e))
                i = j
            else:
                i += 1
        return out

    def _clean_audio(self, audio, base, others_active, times):
        clean = np.ones(audio.size, dtype=bool)
        for ti, t in enumerate(times):
            if others_active[ti]:
                i0, i1 = int((t - base) * SR), int((t - base + FRAME_S) * SR)
                clean[max(i0, 0):max(i1, 0)] = False
        return audio[clean]

    def _learn(self, spk, pieces):
        """Online slot profile + identification against enrolled people.

        Returns (person id or None, sim, margin, clean_s). Identification uses
        only clean audio >= 1 s, so an overlapped stretch can never re-label a slot.
        """
        clean_s = pieces.size / SR
        if self.bank is None or pieces.size < 1.0 * SR:
            return None, 0.0, 0.0, clean_s
        try:
            if pieces.size >= ENROLL_S * SR:
                self.bank.enroll(spk, pieces)
            pid, sim, margin = self.bank.identify(pieces)
            if pid is not None:
                changed = self.bank.vote(spk, pid, clean_s)
                if changed is not None:
                    self.pending.append({"type": "identify", "spk": spk, "person": changed,
                                         "name": self.bank.people[changed]["name"],
                                         "sim": round(sim, 3)})
            return pid, sim, margin, clean_s
        except Exception as e:  # noqa: BLE001
            print(f"[livediar] profile update failed: {e}")
            return None, 0.0, 0.0, clean_s

    def _run_dominant(self, spk, other, s, e, prob_dom):
        """Is `spk` the louder voice in the overlapped stretch [s, e)?

        Profiles (TitaNet) when available: the mix embeds closer to the louder
        voice. Sortformer probabilities saturate for both talkers, so they are
        only the fallback.
        """
        if self.bank is None:
            return prob_dom
        mix = self.ring.slice(s, e)
        if mix.size < 0.6 * SR:
            return prob_dom
        try:
            e_mix = self.bank.embed(mix)
        except Exception:  # noqa: BLE001
            return prob_dom
        ps = self.bank.profile(spk)
        po = self.bank.profile(other) if other is not None else None
        if ps is not None and po is not None:
            return float(e_mix @ ps) >= float(e_mix @ po)
        if ps is not None:
            return float(e_mix @ ps) >= 0.5
        if po is not None:
            return float(e_mix @ po) < 0.35
        return prob_dom

    def _transcribe_slice(self, spk, t0, t1, audio, final):
        """Word-level attribution + optional extraction. Runs in a worker thread."""
        dom = self._dominance(spk, t0, t1)
        base = t0 - PAD_S
        if dom is None or not dom[1].any():           # clean slice: plain transcription
            r = self.asr.transcribe(audio)
            r["overlap"] = 0.0; r["recovered"] = False
            r["person"], r["sim"], r["margin"], r["clean_s"] = (
                self._learn(spk, audio) if (final and r["text"]) else (None, 0.0, 0.0, 0.0))
            return r
        times, others_active, prob_dominant = dom
        overlap_s = float(others_active.sum()) * FRAME_S
        person, sim, margin, clean_s = None, 0.0, 0.0, 0.0
        if final:
            person, sim, margin, clean_s = self._learn(spk, self._clean_audio(audio, base, others_active, times))

        # overlapped stretches, each judged once for dominance
        P, A, _ = self.activity.window(t0, t1)
        dominant = prob_dominant.copy()
        stretches = []                                 # (s, e, other, spk_dominant)
        for (s, e) in self._runs(others_active, times, EXTRACT_MIN_S):
            m = (times >= s) & (times < e)
            others_mass = np.where(A[m], P[m], 0.0).sum(axis=0); others_mass[spk] = -1
            other = int(np.argmax(others_mass)) if others_mass.max() > 0 else None
            d = self._run_dominant(spk, other, s, e, bool(prob_dominant[m].mean() >= 0.5))
            dominant[m] = d
            stretches.append((s, e, other, d))

        r = self.asr.transcribe(audio, words=True)
        lang = r.get("language")

        def frame_stats(ws, we):                       # overlap / dominance over a word span
            m = (times + FRAME_S > ws) & (times < we)
            if not m.any():
                return False, True
            return bool(others_active[m].any()), bool(dominant[m].mean() >= 0.5)

        kept, dropped = [], []
        for w in r["words"]:
            ws, we = base + w["start"], base + w["end"]
            ov, dm = frame_stats(ws, we)
            (kept if (not ov or dm) else dropped).append({**w, "start": ws, "end": we})

        recovered = False
        if self.mode == "extract" and self.separator is not None and self.bank is not None:
            for (s, e, other, d) in stretches:
                if d:
                    continue                           # we are the louder voice: mix words are ours
                s, e = max(s - 0.25, 0.0), e + 0.25
                mix = self.ring.slice(s, e)
                if mix.size < EXTRACT_MIN_S * SR:
                    continue
                try:
                    est = self.separator.separate(mix, key=(int(s * SR), int(e * SR)))
                    k = self.bank.pick(spk, est, [other] if other is not None else [])
                except Exception as ex:  # noqa: BLE001
                    print(f"[livediar] extraction failed: {type(ex).__name__}: {ex}")
                    continue
                if k < 0:
                    continue
                rr = self.asr.transcribe(est[:, k], language=lang, words=True)
                taken = {_norm_word(d_["word"]) for d_ in dropped if s <= d_["start"] < e}
                new = [{**w, "start": s + w["start"], "end": s + w["end"]} for w in rr["words"]
                       if _norm_word(w["word"]) and _norm_word(w["word"]) not in taken
                       and w["p"] >= EXTRACT_MIN_P]
                if len(new) >= 2:
                    kept.extend(new); recovered = True
                    if final:
                        self.extracted_s += e - s
        kept.sort(key=lambda w: w["start"])
        sep = "" if lang == "zh" else " "
        text = sep.join(w["word"].strip() for w in kept).strip()
        if final:
            self.overlap_s += overlap_s
        return {"text": text, "language": lang, "overlap": round(overlap_s, 2), "recovered": recovered,
                "person": person, "sim": sim, "margin": margin, "clean_s": clean_s}

    async def _run(self):
        while True:
            job = await self.queue.get()
            if job is None:
                return
            cap_id, spk, t0, t1, final = job
            audio = self.ring.slice(t0 - PAD_S, t1 + PAD_S)
            if audio.size < MIN_SEG_S * SR:
                continue
            tw = time.perf_counter()
            try:
                r = await asyncio.to_thread(self._transcribe_slice, spk, t0, t1, audio, final)
            except Exception as e:  # noqa: BLE001 - never kill the socket over a caption
                print(f"[livediar] asr failed: {type(e).__name__}: {e}")
                continue
            self.asr_wall_s += time.perf_counter() - tw
            self.asr_audio_s += audio.size / SR
            while self.pending:
                await self.emit(self.pending.pop(0))
            if not r["text"] and not final:
                continue
            # identity: verified on this slice's own clean audio, else the slot's mapping
            pid = r.get("person")
            if pid is None and self.bank is not None:
                pid = self.bank.slot_person.get(spk)
            name = self.bank.people[pid]["name"] if (pid and self.bank and pid in self.bank.people) else None
            await self.emit({"type": "caption", "id": cap_id, "spk": spk,
                             "t0": round(t0, 2), "t1": round(t1, 2),
                             "text": r["text"], "final": final, "lang": r.get("language"),
                             "overlap": r.get("overlap", 0.0), "recovered": r.get("recovered", False),
                             "person": pid, "name": name, "verified": r.get("person") is not None,
                             "sim": round(float(r.get("sim") or 0), 3),
                             "margin": round(float(r.get("margin") or 0), 3),
                             "clean_s": round(float(r.get("clean_s") or 0), 2),
                             "slot_person": self.bank.slot_person.get(spk) if self.bank else None})

    def _schedule(self, o: _Open, t1: float, final: bool):
        self.queue.put_nowait((o.cap_id, o.spk, o.t0, t1, final))

    def _close(self, o: _Open, t1: float):
        self.open.pop(o.spk, None)
        self._schedule(o, t1, final=True)

    def on_events(self, events, now: float):
        """Feed TurnTracker events; `now` is the stream time of the latest frame.

        A turn "end" does not close the caption immediately: if the same
        speaker starts again within MERGE_GAP_S (a breath, a comma) the
        caption simply continues, which gives Whisper sentence-length audio
        instead of fragments. Another speaker starting closes it at once.
        """
        for ev in events:
            if ev.event == "start":
                o = self.open.get(ev.spk)
                if o is not None and o.ended_at is not None:
                    o.ended_at = None                   # resumed: keep the caption open
                    continue
                for other in list(self.open.values()):  # someone else: close pending ones
                    if other.spk != ev.spk and other.ended_at is not None:
                        self._close(other, other.ended_at)
                self.open[ev.spk] = _Open(ev.spk, ev.t, self.next_id, ev.t)
                self.next_id += 1
            elif ev.event == "end" and ev.spk in self.open:
                self.open[ev.spk].ended_at = ev.t
        for o in list(self.open.values()):
            if o.ended_at is not None:
                if now - o.ended_at >= MERGE_GAP_S:      # gap too long: it really ended
                    self._close(o, o.ended_at)
                continue
            if now - o.t0 >= COMMIT_S:                  # long turn: commit a chunk
                self._schedule(o, now, final=True)
                o.t0, o.cap_id, o.last_partial = now, self.next_id, now
                self.next_id += 1
            elif now - o.last_partial >= PARTIAL_S:     # provisional text
                self._schedule(o, now, final=False)
                o.last_partial = now

    async def finish(self, now: float):
        """Flush open turns and wait for every queued caption to be emitted."""
        for o in list(self.open.values()):
            self._schedule(o, o.ended_at if o.ended_at is not None else now, final=True)
        self.open.clear()
        if self.queue is not None:
            await self.queue.put(None)
            if self.worker is not None:
                await self.worker

    def cancel(self):
        if self.worker is not None:
            self.worker.cancel()

    @property
    def rtf(self) -> float:
        return self.asr_wall_s / max(self.asr_audio_s, 1e-6)
