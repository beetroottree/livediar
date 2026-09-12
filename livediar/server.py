"""livediar server.

Browser mic  --PCM16 @16k over WebSocket-->  engine (Sortformer | mock)
                                             --> frame probs + turn events
Browser UI  <--JSON frames/turns/stats-------┘

Run:
    python -m livediar.server                    # auto: sortformer if NeMo present, else mock
    python -m livediar.server --engine mock      # UI demo, no model needed
    python -m livediar.server --preset ultra     # 0.32 s buffer
"""

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .asr import ActivityRing, AudioRing, Captioner, build_asr, build_overlap
from .engine_mock import MockEngine, SR
from .presets import DEFAULT_PRESET, PRESETS
from .turns import TurnTracker

STATIC = Path(__file__).resolve().parent.parent / "static"
PROFILES = Path(__file__).resolve().parent.parent / "profiles.json"


def load_profiles(bank):
    if bank is None or not PROFILES.exists():
        return
    try:
        for p in json.loads(PROFILES.read_text()).get("people", []):
            e = np.asarray(p["emb"], dtype=np.float32)
            bank.people[p["id"]] = {"name": p["name"], "emb": e / (np.linalg.norm(e) + 1e-9),
                                    "secs": float(p.get("secs", 0))}
        print(f"[livediar] {len(bank.people)} enrolled voice(s): "
              + ", ".join(v["name"] for v in bank.people.values()))
    except Exception as e:  # noqa: BLE001
        print(f"[livediar] could not read {PROFILES}: {e}")


def save_profiles(bank):
    PROFILES.write_text(json.dumps({"people": [
        {"id": pid, "name": p["name"], "secs": p["secs"], "emb": [round(float(x), 5) for x in p["emb"]]}
        for pid, p in bank.people.items()]}, ensure_ascii=False))


def people_msg(bank):
    return {"type": "profiles",
            "people": [bank.person_public(pid) for pid in bank.people] if bank else [],
            "enabled": bank is not None}

app = FastAPI(title="livediar")
CFG = {"engine": "auto", "preset": DEFAULT_PRESET,
       "model": "nvidia/diar_streaming_sortformer_4spk-v2.1", "device": "auto"}
# Turn-tracker defaults; the UI can override per connection with a "config" message.
TRACK = {"on_th": 0.55, "off_th": 0.40, "min_on": 2, "min_off": 4, "max_spk": 0}
# Captions: whisper (mlx, Apple GPU) | none. language None = auto-detect per turn.
ASR_CFG = {"asr": "whisper", "asr_model": None, "language": None, "langs": "en,zh"}
ASR = {"obj": None, "tried": False, "overlap": ("words", None, None)}
OVERLAP_MODE = {"mode": "words"}


def get_asr():
    """Build the caption backend once; a failure disables captions, not the app."""
    if not ASR["tried"]:
        ASR["tried"] = True
        if ASR_CFG["asr"] != "none":
            try:
                print(f"[livediar] loading asr {ASR_CFG['asr']} "
                      f"({ASR_CFG['asr_model'] or 'default model'}, "
                      f"lang={ASR_CFG['language'] or 'auto'}) ...")
                ASR["obj"] = build_asr(ASR_CFG["asr"], ASR_CFG["asr_model"], ASR_CFG["language"],
                                       ASR_CFG["langs"].split(","))
                print("[livediar] asr ready.")
                if OVERLAP_MODE["mode"] == "extract":
                    print("[livediar] loading overlap models (TitaNet profiles + SepFormer) ...")
                ASR["overlap"] = build_overlap(OVERLAP_MODE["mode"])
                print(f"[livediar] overlap mode: {ASR['overlap'][0]}")
                load_profiles(ASR["overlap"][1])
            except Exception as e:  # noqa: BLE001
                print(f"[livediar] asr unavailable ({type(e).__name__}: {e}); captions off.")
    return ASR["obj"]
MODEL_CACHE = {}


def build_engine():
    preset = PRESETS[CFG["preset"]]
    if CFG["engine"] in ("auto", "sortformer"):
        try:
            from .engine_sortformer import SortformerEngine
            key = (CFG["model"], CFG["preset"], CFG["device"])
            if key not in MODEL_CACHE:
                print(f"[livediar] loading {CFG['model']} (preset={preset.name}, "
                      f"buffer={preset.buffer_s:.2f}s) ...")
                MODEL_CACHE[key] = SortformerEngine(
                    preset, model_name=CFG["model"], device=CFG["device"])
                print("[livediar] model ready.")
            eng = MODEL_CACHE[key]
            eng.reset()
            return eng
        except Exception as e:  # noqa: BLE001 - fall back, but say why
            if CFG["engine"] == "sortformer":
                raise
            print(f"[livediar] NeMo engine unavailable ({type(e).__name__}: {e}); "
                  f"falling back to mock engine. Install extras per README for the real model.")
    return MockEngine(preset)


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    engine = await asyncio.to_thread(build_engine)
    tracker = TurnTracker(n_spk=engine.n_spk, on_th=TRACK["on_th"], off_th=TRACK["off_th"],
                          min_on_frames=TRACK["min_on"], min_off_frames=TRACK["min_off"],
                          max_spk=TRACK["max_spk"])
    asr = await asyncio.to_thread(get_asr)
    ring = AudioRing()
    captioner = None
    bank = None
    if asr is not None:
        mode, bank, sep = ASR["overlap"]
        if bank is not None:
            bank.reset_session()        # slot<->person mapping is per session; people persist
        captioner = Captioner(asr, ring, lambda m: sock.send_text(json.dumps(m)),
                              activity=ActivityRing(engine.n_spk), mode=mode,
                              bank=bank, separator=sep)
        captioner.start()
    enroll = None                       # {"name": str, "chunks": [np arrays]} while enrolling
    await sock.send_text(json.dumps({"type": "ready", **engine.info,
                                     "tracker": tracker.params,
                                     "overlap": ASR["overlap"][0] if asr else "none",
                                     **(asr.info if asr else {"asr": "none"})}))
    await sock.send_text(json.dumps(people_msg(bank)))

    audio_s = 0.0
    compute_s = 0.0
    last_stats = time.monotonic()

    async def handle_steps(steps):
        nonlocal compute_s, last_stats
        for st in steps:
            compute_s += st.step_wall_s
            active, events, smoothed = tracker.update(st.probs, st.t0)
            await sock.send_text(json.dumps({
                "type": "frames", "t0": round(st.t0, 3), "dt": 0.08,
                "probs": np.round(smoothed, 3).tolist(), "active": active}))
            for ev in events:
                await sock.send_text(json.dumps(
                    {"type": "turn", "spk": ev.spk, "event": ev.event,
                     "t": round(ev.t, 2)}))
            if captioner is not None:
                captioner.on_frames(smoothed, tracker.last_active_frames)
                captioner.on_events(events, st.t0 + st.probs.shape[0] * 0.08)
        if time.monotonic() - last_stats > 1.0:
            last_stats = time.monotonic()
            rtf = compute_s / max(audio_s, 1e-6)
            await sock.send_text(json.dumps({
                "type": "stats", "talk_s": [round(x, 1) for x in tracker.talk_s],
                "turns": tracker.turns, "rtf": round(rtf, 3),
                "suppressed_s": [round(x, 1) for x in tracker.suppressed_s],
                "asr_rtf": round(captioner.rtf, 3) if captioner else None,
                "overlap_s": round(captioner.overlap_s, 1) if captioner else 0,
                "extracted_s": round(captioner.extracted_s, 1) if captioner else 0,
                "clock": round(audio_s, 1)}))

    # Drain the socket into a queue independently of model speed. If the app only
    # read between inference steps, a client sending faster than realtime (the
    # "open recording" path) would stall the transport, keepalive pongs would
    # go unread, and uvicorn would drop the connection after ~20 s.
    inbox: asyncio.Queue = asyncio.Queue()

    async def pump():
        try:
            while True:
                m = await sock.receive()
                await inbox.put(m)
                if m.get("type") == "websocket.disconnect":
                    return
        except WebSocketDisconnect:
            await inbox.put({"type": "websocket.disconnect"})

    pump_task = asyncio.create_task(pump())
    try:
        while True:
            msg = await inbox.get()
            if msg.get("type") == "websocket.disconnect":
                break
            if (data := msg.get("bytes")) is not None:
                pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                if enroll is not None:          # enrollment audio, not meeting audio
                    enroll["chunks"].append(pcm)
                    continue
                audio_s += pcm.size / SR
                ring.push(pcm)
                steps = await asyncio.to_thread(engine.feed, pcm)
                await handle_steps(steps)
            elif (text := msg.get("text")) is not None:
                m = json.loads(text)
                if m.get("type") == "stop":
                    await handle_steps(await asyncio.to_thread(engine.finalize))
                    if captioner is not None:
                        await captioner.finish(audio_s)   # drain captions before "stopped"
                        captioner = None
                    await sock.send_text(json.dumps({"type": "stopped"}))
                    break
                if m.get("type") == "config":
                    await sock.send_text(json.dumps(
                        {"type": "config", "tracker": tracker.configure(**m)}))
                elif m.get("type") == "enroll_begin":
                    enroll = {"name": (m.get("name") or "Voice").strip()[:40], "chunks": []}
                elif m.get("type") == "enroll_end" and enroll is not None:
                    audio = np.concatenate(enroll["chunks"]) if enroll["chunks"] else np.zeros(0, np.float32)
                    name, enroll = enroll["name"], None
                    if bank is None:
                        await sock.send_text(json.dumps({"type": "error", "text": "voice profiles are off"}))
                    elif audio.size < 3 * SR:
                        await sock.send_text(json.dumps({"type": "error", "text": "need at least 3 s of speech to enroll"}))
                    else:
                        pid = m.get("id") or f"p{uuid.uuid4().hex[:10]}"
                        await asyncio.to_thread(bank.add_person, pid, name, audio)
                        save_profiles(bank)
                        await sock.send_text(json.dumps(people_msg(bank)))
                elif m.get("type") == "profile_delete" and bank is not None:
                    bank.people.pop(m.get("id"), None)
                    save_profiles(bank)
                    await sock.send_text(json.dumps(people_msg(bank)))
                elif m.get("type") == "profile_rename" and bank is not None and m.get("id") in bank.people:
                    bank.people[m["id"]]["name"] = (m.get("name") or "Voice").strip()[:40]
                    save_profiles(bank)
                    await sock.send_text(json.dumps(people_msg(bank)))
    except WebSocketDisconnect:
        pass
    finally:
        pump_task.cancel()
        if captioner is not None:
            captioner.cancel()


def main():
    ap = argparse.ArgumentParser(description="livediar - live speaker diarization server")
    ap.add_argument("--engine", choices=["auto", "sortformer", "mock"], default="auto")
    ap.add_argument("--preset", choices=list(PRESETS), default=DEFAULT_PRESET,
                    help="latency preset: ultra=0.32s low=1.04s high=10s vhigh=30.4s")
    ap.add_argument("--model", default=CFG["model"],
                    help="HF id or local .nemo (use ...4spk-v2 for CC-BY-4.0)")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8321)
    tg = ap.add_argument_group("turn tracker (UI can override per session)")
    tg.add_argument("--max-speakers", type=int, default=0,
                    help="cap on distinct speakers (0 = model max of 4); ghosts get suppressed")
    tg.add_argument("--on-th", type=float, default=TRACK["on_th"], help="turn starts above this prob")
    tg.add_argument("--off-th", type=float, default=TRACK["off_th"], help="turn ends below this prob")
    tg.add_argument("--min-on", type=int, default=TRACK["min_on"], help="frames (80 ms) above on-th to start")
    tg.add_argument("--min-off", type=int, default=TRACK["min_off"], help="frames below off-th to end")
    cg = ap.add_argument_group("captions")
    cg.add_argument("--asr", choices=["whisper", "none"], default="whisper",
                    help="caption backend (whisper = mlx-whisper on the Apple GPU)")
    cg.add_argument("--asr-model", default=None,
                    help="HF repo for the caption model (default mlx-community/whisper-large-v3-turbo)")
    cg.add_argument("--lang", default=None,
                    help="pin the caption language (en, zh, ...); default auto-detect per turn")
    cg.add_argument("--langs", default="en,zh",
                    help="candidate languages for auto-detect, comma separated (default en,zh)")
    cg.add_argument("--overlap", choices=["words", "extract", "none"], default="words",
                    help="overlapped speech: words = keep each word only for the speaker the diarizer "
                         "and voice profiles say dominates (default); extract = also un-mix the quieter "
                         "speaker with TitaNet+SepFormer (experimental: measured junk on real speech); "
                         "none = raw mix")
    args = ap.parse_args()
    CFG.update(engine=args.engine, preset=args.preset, model=args.model, device=args.device)
    ASR_CFG.update(asr=args.asr, asr_model=args.asr_model, language=args.lang, langs=args.langs)
    OVERLAP_MODE["mode"] = args.overlap
    TRACK.update(on_th=args.on_th, off_th=args.off_th, min_on=args.min_on,
                 min_off=args.min_off, max_spk=args.max_speakers)
    if args.engine != "mock":
        try:
            build_engine()  # warm the model before serving
        except Exception as e:  # noqa: BLE001
            print(f"[livediar] model preload failed: {e}")
    get_asr()  # warm the caption model too
    print(f"[livediar] open http://{args.host}:{args.port}  (mic requires localhost or https)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
