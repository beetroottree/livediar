# livediar

Live "who is speaking now" plus "what they said" in the browser, on the current
open-source SOTA for streaming diarization: **NVIDIA Streaming Sortformer v2.1**
(4 speaker slots, arrival-order labels, 1.04 s default decision buffer), with
speaker-attributed captions from **Whisper large-v3-turbo on the Apple GPU**
(mlx-whisper).

```
browser mic ──PCM16 @16 kHz──▶ WebSocket ──▶ Sortformer streaming step (NeMo, CPU)
                                              │  speaker cache + FIFO state
                                              ▼  hysteresis turn tracker + speaker cap
                                              │  turn audio slice ──▶ Whisper (MLX, GPU)
UI  ◀── frames / turns / captions / stats ◀───┘
```

## Project map

| where | what |
|---|---|
| `livediar/`, `static/` | the live app: Sortformer diarization, Whisper captions, voice profiles, replay (this README) |
| `bench/` | AMI test-set benchmark with ground truth (DER, cpWER, idWER); results in `bench/ami/results/` |
| `PLAN.md` | the programme for a room-aware full-duplex spoken model, and the $500 execution plan (§8) |
| `research/` | the code behind the plan: conditioning module, simulator, ablations, Dixtral gate, Moshi-LoRA pilot, Modal jobs |
| `docs/RUNS.md` | run log: every job, its cost, its metric, what it changed |
| `docs/DECISIONS.md` | every non-obvious decision with the evidence behind it |
| `docs/research/` | deep-research reports: full-duplex models, diarization conditioning, data & licences, infrastructure & Modal |
| `samples/` | test clips' transcripts and RTTMs (audio fetched per `samples/README.md`) |

## Quickstart (UI demo, no model, ~30 s)

```bash
pip install -r requirements.txt
python -m livediar.server --engine mock --asr none
# open http://127.0.0.1:8321  → "Start listening" (or "Watch demo")
```

The mock engine mimics the real one's timing and output shapes, so you can
develop the UI and the audio path anywhere. `index.html` also runs standalone:
its "Watch demo" button needs no server at all.

## Real model

```bash
# in a fresh env; NeMo is heavy. GPU strongly recommended (CPU works, slower).
pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main"
python -m livediar.server                 # downloads nvidia/diar_streaming_sortformer_4spk-v2.1
python -m livediar.server --preset ultra  # 0.32 s buffer, snappier, less accurate
python -m livediar.server --model nvidia/diar_streaming_sortformer_4spk-v2   # CC-BY-4.0 variant
```

Notes:
- v2.1 is the accuracy pick (much better on meeting-style speech); it ships
  under the **NVIDIA Open Model License**. v2 is **CC-BY-4.0** — choose per
  your redistribution needs.
- Mic capture requires `localhost` or HTTPS (browser rule).
- First run downloads ~470 MB from Hugging Face.

## Captions (who said what)

Sortformer never sees words, so a second model runs beside it. The design is
*diarizer-driven*: every turn the tracker closes is single-speaker by
construction, so the server slices that audio out of a rolling buffer and
transcribes it — attribution is exact, no word-to-speaker matching. Long turns
get a provisional caption every 3 s and are committed in ≤14 s chunks; a turn
that resumes within 1.2 s (a breath, a comma) is merged so Whisper sees
sentences instead of fragments. The backend is `mlx-whisper` on the Apple GPU
(`pip install mlx-whisper`, ~1.6 GB download on first run), which leaves the CPU
to Sortformer and runs ~20× realtime, so captions land within about a second
of a turn ending.

Language: auto-detect per turn, restricted to `--langs en,zh` (free auto-detect
guesses Dutch for a mumbled 1.5 s English fragment; restricting candidates and
using the session's majority language for short slices fixes that). English /
Mandarin code-switching works turn by turn. `--lang en` pins it; `--asr none`
turns captions off.

### Overlapping speech

When two people talk at once both turns contain the mix. `--overlap words`
(default) uses the diarizer's own per-frame activity to fix attribution:
Whisper runs with word timestamps, and a word inside an overlapped stretch is
kept only for the speaker who dominates it. Dominance comes from voice
profiles — TitaNet-large embeddings (`livediar/overlap.py`) learned from each
slot's clean, non-overlapped audio; the mixed audio embeds much closer to the
louder voice (0.69 vs 0.16 cosine on the Obama/Trump test). Sortformer's
probabilities saturate for both talkers, so they are only the fallback before
profiles exist. Result: nothing shows under two names, nothing is invented;
the quieter voice's overlapped words are dropped.

`--overlap extract` additionally un-mixes the quieter speaker's stretch with
SepFormer (speechbrain, WHAMR 16 kHz, runs on the Apple GPU in ~0.07× realtime)
and picks their stream by profile. It is **off by default because it was
measured to produce junk**: on a synthetic Obama/Trump overlap the separated
streams leak the other voice and Whisper turns them into fragments like
"months other. had", even with a per-word confidence floor. The mechanics
(profiles, stream picking, leakage filter) are in place for a better separator
— NVIDIA's multi-talker Parakeet (English only) is the likely upgrade. Note the
diarizer itself can miss the second voice for a second at overlap onset; no
attribution logic can recover words the diarizer never flagged as shared.

## Enrolled voices (who, not which slot)

Sortformer's slots are anonymous and arrival-ordered, and two voices with
similar pitch can get swapped or merged at an overlap. The fix is to stop
trusting the slot for identity: each person presses **+ Add my voice** once
and talks for 8 s; the server stores a TitaNet-large embedding in
`profiles.json`. From then on every turn's clean audio (≥ 1 s, overlapped
frames excluded) is verified against the enrolled voices; a caption carries
the verified person, tiles auto-name themselves once a slot has been
identified, and if the diarizer hands a turn to the wrong slot the caption is
still attributed to the right person. Overlap dominance uses the enrolled
profile too. Timbre, not pitch, is what the embedding encodes: Obama and Trump
differ by 11 Hz in median F0 and TitaNet still identifies 2 s snippets 13/13.
Identification needs cosine ≥ 0.40 to the best profile and a 0.08 margin over
the runner-up; otherwise the slot's existing mapping is used, never a guess.

## Speaker cap (you know how many people are in the room)

Sortformer will open a fourth slot for a cough or a bad echo. Set "people in
room" in the UI (remembered in the browser) or `--max-speakers 3`: once that
many slots have each held one continuous turn ≥ 1.5 s, an unestablished slot
can't start a turn; its speech is counted as `suppressed_s` in the stats.
Threshold-only tuning was measured to trade real talk time for almost no ghost
reduction, which is why the cap exists. Thresholds are still exposed
(`--on-th --off-th --min-on --min-off`, and a `config` WebSocket message).

## Replay

The browser keeps the session's PCM, every frame, turn event and caption.
After Stop, "Replay session" plays the audio back with the board, timeline and
transcript in sync (space = play/pause, ←/→ = 5 s). "Open recording…" decodes
any audio file locally, streams it through the server (faster than realtime;
the socket is drained into a queue so uvicorn's keepalive doesn't drop it) and
lands in replay. "Save .txt" exports the transcript with the names you typed.

## File mode (no mic, good for eval)

```bash
python -m livediar.filediar meeting.wav --rttm out.rttm
python -m livediar.filediar meeting.wav --asr whisper --max-speakers 3 --transcript out.txt
```

Feeds a wav through the identical streaming path and writes RTTM — pair with
`pyannote.metrics` for DER, e.g. against IMDA NSC-derived references. Sample
clips live in `samples/` (C-SPAN Obama–Trump Oval Office pool feed, a Mandarin
interview skit, an English/Mandarin sitcom scene) with their transcripts.

## Latency presets

| preset | decision buffer | config (chunk / rc / fifo) | use |
|---|---|---|---|
| `ultra` | 0.32 s | 3 / 1 / 188 | interruption-grade reactions |
| `low` (default) | 1.04 s | 6 / 7 / 188 | live "who's talking" boards |
| `high` | 10 s | 124 / 1 / 124 | live captions w/ better DER |
| `vhigh` | 30.4 s | 340 / 40 / 40 | near-offline quality |

Latency = (chunk_len + right_context) × 80 ms. All presets use the same
weights — switch freely.

## How the live adapter works

`livediar/engine_sortformer.py` mirrors NeMo's `forward_streaming` loop
incrementally: it keeps a raw-sample ring buffer, computes the mel window each
step (chunk ± left/right context, exactly the geometry of
`SortformerModules.streaming_feat_loader`), and calls
`model.forward_streaming_step(...)` with persistent
`StreamingSortformerState` (speaker cache + FIFO). New 80 ms frame
probabilities go through a median-3 + hysteresis turn tracker
(`livediar/turns.py`) before the UI sees them, so the board doesn't flicker.

## Honest limits

- 4 speaker slots (model architecture); 5+ people degrades. Slots are
  arrival-ordered — stable but anonymous. Rename tiles in the UI, or add an
  enrollment layer for automatic naming.
- Primarily English/Mandarin training data; expect degradation on heavy
  code-switching or far-field noise until you fine-tune (see roadmap).
- Per-chunk mel can differ negligibly from whole-file mel at chunk edges.
- Browser echo cancellation is on so the app doesn't diarize its own tab.

## Roadmap hooks

This is the serving skeleton for the bigger plan: ONNX backend (drop the NeMo
dependency), per-speaker enrollment (slot → name), accent packs and per-tenant
fine-tuning (NSC-style pseudo-label + simulation pipeline), and a
`>4 speakers` offline fallback via pyannote community-1.
