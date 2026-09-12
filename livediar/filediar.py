"""Run the live engine over a wav file: prints turn events and writes RTTM.

    python -m livediar.filediar meeting.wav --preset low --rttm out.rttm
    python -m livediar.filediar meeting.wav --asr whisper --transcript out.txt

Feeds the file in 0.5 s blocks through the same code path the mic uses, so it
doubles as an integration test and a DER-eval front end (pair the RTTM with
pyannote.metrics or NeMo's scoring tools).
"""

import argparse
import asyncio
import sys
import wave

import numpy as np

from .asr import ActivityRing, AudioRing, Captioner, build_asr, build_overlap
from .engine_mock import SR, MockEngine
from .presets import DEFAULT_PRESET, PRESETS
from .turns import TurnTracker


def read_wav_16k_mono(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width != 2:
        sys.exit("filediar: need 16-bit PCM wav")
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    if rate != SR:
        n_out = int(round(pcm.size * SR / rate))
        x_old = np.linspace(0.0, 1.0, num=pcm.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        pcm = np.interp(x_new, x_old, pcm).astype(np.float32)
    return pcm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--engine", choices=["auto", "sortformer", "mock"], default="auto")
    ap.add_argument("--preset", choices=list(PRESETS), default=DEFAULT_PRESET)
    ap.add_argument("--model", default="nvidia/diar_streaming_sortformer_4spk-v2.1")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--rttm", default=None, help="write RTTM here")
    ap.add_argument("--asr", choices=["whisper", "none"], default="none")
    ap.add_argument("--asr-model", default=None)
    ap.add_argument("--lang", default=None, help="pin caption language; default auto")
    ap.add_argument("--langs", default="en,zh", help="auto-detect candidates, comma separated")
    ap.add_argument("--transcript", default=None, help="write speaker-attributed transcript here")
    ap.add_argument("--max-speakers", type=int, default=0)
    ap.add_argument("--overlap", choices=["words", "extract", "none"], default="words")
    args = ap.parse_args()

    preset = PRESETS[args.preset]
    engine = None
    if args.engine in ("auto", "sortformer"):
        try:
            from .engine_sortformer import SortformerEngine
            engine = SortformerEngine(preset, model_name=args.model, device=args.device)
        except Exception as e:  # noqa: BLE001
            if args.engine == "sortformer":
                raise
            print(f"# NeMo unavailable ({e}); using mock engine", file=sys.stderr)
    engine = engine or MockEngine(preset)

    pcm = read_wav_16k_mono(args.wav)
    tracker = TurnTracker(n_spk=engine.n_spk, max_spk=args.max_speakers)
    open_t = {}
    segments = []
    captions = {}

    asr = (build_asr(args.asr, args.asr_model, args.lang, args.langs.split(","))
           if args.asr != "none" else None)
    ring = AudioRing(keep_s=pcm.size / SR + 1)
    ring.push(pcm)                      # file mode: the whole file is the ring

    async def on_caption(m):
        captions[m["id"]] = m           # partials are replaced by finals in place
        if m["final"] and m["text"]:
            tag = (" ⧉" if m.get("recovered") else (" ~" if m.get("overlap") else ""))
            print(f"{m['t0']:8.2f}s  speaker_{m['spk']}{tag}  {m['text']}")

    captioner = None
    if asr:
        mode, bank, sep = build_overlap(args.overlap)
        captioner = Captioner(asr, ring, on_caption, activity=ActivityRing(engine.n_spk),
                              mode=mode, bank=bank, separator=sep)

    def consume(steps):
        for st in steps:
            _, events, smoothed = tracker.update(st.probs, st.t0)
            if captioner:
                captioner.on_frames(smoothed, tracker.last_active_frames)
            for ev in events:
                if not captioner:
                    print(f"{ev.t:8.2f}s  speaker_{ev.spk}  {ev.event}")
                if ev.event == "start":
                    open_t[ev.spk] = ev.t
                elif ev.spk in open_t:
                    segments.append((open_t.pop(ev.spk), ev.t, ev.spk))
            if captioner:
                captioner.on_events(events, st.t0 + st.probs.shape[0] * 0.08)

    async def run():
        if captioner:
            captioner.start()
        block = SR // 2
        for i in range(0, pcm.size, block):
            consume(await asyncio.to_thread(engine.feed, pcm[i:i + block]))
        consume(await asyncio.to_thread(engine.finalize))
        if captioner:
            await captioner.finish(pcm.size / SR)

    asyncio.run(run())
    end = pcm.size / SR
    for s, t in open_t.items():
        segments.append((t, end, s))

    print(f"\n# talk time: " + "  ".join(
        f"spk{s}={tracker.talk_s[s]:.1f}s/{tracker.turns[s]}turns" for s in range(engine.n_spk)))
    if args.rttm:
        name = args.wav.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        with open(args.rttm, "w") as f:
            for s0, s1, spk in sorted(segments):
                f.write(f"SPEAKER {name} 1 {s0:.3f} {s1 - s0:.3f} "
                        f"<NA> <NA> speaker_{spk} <NA> <NA>\n")
        print(f"# wrote {args.rttm}")
    if args.transcript and captioner:
        with open(args.transcript, "w") as f:
            for m in sorted((m for m in captions.values() if m["final"] and m["text"]),
                            key=lambda m: m["t0"]):
                f.write(f"[{m['t0']:7.2f}] speaker_{m['spk']}: {m['text']}\n")
        print(f"# wrote {args.transcript}  (asr rtf {captioner.rtf:.3f}, overlap {captioner.overlap_s:.1f}s, "
              f"extracted {captioner.extracted_s:.1f}s)")


if __name__ == "__main__":
    main()
