# Run log

One entry per job of PLAN.md §8. Nothing is "done" without its metric. Costs
come from `spend.jsonl` on the Modal volume (`modal run research/modal_app.py::spend`);
local jobs cost $0 and list wall-clock instead.

| # | job | where | started | status | cost | metric | changed the plan? |
|---|---|---|---|---|---|---|---|
| 1 | AMI benchmark (16 meetings × none/words/enrolled) | local | 2026-09-12 00:08 | running — 11/16 diarized, 17/48 scored; paused 6 h by a laptop sleep, resumed under `caffeinate` | $0 | partial: cpWER 22–44 %, attribution beats raw mix on every meeting so far; enrolled identity 61 → 26 % idWER on IS1009c | yes: identity must be a model input (PLAN §2) |
| 2 | toy conditioning ablation, 320 rooms | local (MPS) | 2026-09-12 11:47 | v1/v2 invalid (from-scratch CTC never learned); **v3 (frozen wav2vec2 + trainable tag rows + conditioning): baseline cond 0 → speaker-agnostic WER 78 %, cpWER 100 % (no tags, as expected); additive cond 1 → WER 99 %, cpWER 100 %: the gate drifted to −0.20 (conditioning RMS 0.25 at the encoder input) and the frozen encoder collapsed to all-blank; at RMS 0.06 the same checkpoint decodes normally. Per-layer arm (cond 2) → WER 96 %, cpWER 100 %: same collapse.** The scaled Modal copy (job 4) is cancelled: the Dixtral gate (job 3) already answers the conditioning question at scale on real meetings (20.3 % vs 41.5 %), and DiCoW's per-layer FDDT *is* the plan's conditioning | $0 local; job 4 ≈ $0.60 spent on a failed simulate step | negative for input-level additive injection into a frozen encoder | yes: `DiarizationConditioning` now has a bounded gate (max 0.1 × unit RMS); the pilot keeps the injection but the LoRA can adapt around it, which the frozen toy could not |
| 3 | Dixtral on Sortformer masks, 16 meetings | Modal 16×H100 | 2026-09-12 16:5x | **14/16 done** (TS3003c/d wait on the local pool, auto re-run queued); 10-container concurrency cap on the Starter plan; two client-side hiccups (a map that stopped the app on the missing-cache meetings → entrypoint now tolerant; warm containers not seeing late uploads → volume reload at job start) | ~$1.5 (per-container log lost entries to concurrent appends — fixed to per-call files) | **word-weighted cpWER 20.3 % vs the Whisper pipeline's 41.5 % (attribution) / 42.5 % (raw) on the same meetings** — see table below | yes: the stage-1 target is set at 20 %; mask conditioning halves the error everywhere except where the diarizer itself confuses speakers (IS1009c 53.5 % vs enrolled identity 25.5 %) |
| 3b | Dixtral on Sortformer masks, IS1009a only | local (MPS) | 2026-09-12 13:0x | running under `caffeinate` at `nice 5`: window 5/7 at RTF 3.9 (first attempt hit RTF 9.2 under heavier contention, then the Metal crash) | $0 | pending | shows why job 3 is a Modal job |
| 4 | scaled ablation v3 on Modal | Modal A100 | 2026-09-12 17:0x | **cancelled** after its simulate step failed in-container (being debugged for the pilot's room data, which uses the same step) — superseded by job 3 as the conditioning evidence | ~$0.60 | — | — |
| 5 | per-scenario turn statistics; regenerate rooms | local (CPU) | 2026-09-12 13:1x | **done** for AMI: `research/fit_turns.py` on train+dev references (154 meetings, 75 k turns): overlap 13.8 % of speech time, 39 % of handovers overlap, turn p50 1.53 s, 36 % of turns < 0.8 s, continuation 0.22 → `simulate.py` defaults replaced; 2 000-room set regenerated with them | $0 | overlap_frac 1–8 % of frames on 2–4-speaker rooms (13.8 % of *speech* on AMI) | yes: the old 15 % single prior was replaced by fitted per-corpus values (PLAN §4.2) |
| 6 | Moshi-LoRA stage-1 pilot | Modal 1×H100, segmented | — | data building: 30 AMI train + 6 dev meetings downloading (CPU); room data built in-container from job 4's rooms; first 1 000-step segment (~1 h, ~$4) runs after job 3's number is in | est. $120 | — | — |
| 7 | pilot second arm | Modal 1×H100 | — | reserve | est. $120 | — | — |
| 8 | assistant-in-the-room scripts + 100 h TTS | local | — | not started | $0 | — | — |

## Job 3 — Dixtral (DiCoW + Voxtral 3B) driven by Sortformer masks, AMI test (cpWER)

Masks: cached Sortformer v2.1 probabilities → livediar TurnTracker (cap 4) → 50 Hz STNO, 120 s windows, one H100 per meeting, RTF 0.03–0.04. Whisper columns are this repo's caption pipeline on the same cached masks (raw mix / word-level attribution / enrolled identity); idWER penalises wrong identity with no permutation.

| meeting | DER | Dixtral | Whisper raw | Whisper attr | Whisper enrolled | idWER |
|---|---|---|---|---|---|---|
| EN2002a | 26.5 % | 26.9 % | 43.9 % | 42.6 % | 42.9 % | 46.8 % |
| EN2002b | 25.7 % | 18.1 % | 41.5 % | 40.9 % | 40.7 % | 45.9 % |
| EN2002c | – | 19.8 % | 41.7 % | – | – | – |
| EN2002d | 30.0 % | 21.0 % | 42.5 % | 42.3 % | – | – |
| ES2004a | 21.5 % | 14.5 % | 29.2 % | 27.3 % | – | – |
| ES2004b | – | 10.1 % | – | – | – | – |
| ES2004c | – | 9.6 % | – | – | – | – |
| ES2004d | – | 17.1 % | – | – | – | – |
| IS1009a | 25.4 % | 22.6 % | 42.2 % | 37.2 % | – | – |
| IS1009b | – | 11.2 % | – | – | – | – |
| IS1009c | 35.1 % | **53.5 %** | 63.5 % | 61.4 % | 61.3 % | **25.5 %** |
| IS1009d | – | 43.0 % | – | – | – | – |
| TS3003a | 27.3 % | 13.8 % | 23.6 % | 22.3 % | 22.2 % | 23.3 % |
| TS3003b | – | 12.1 % | – | – | – | – |
| **14 meetings, word-weighted** | | **20.3 %** | 42.5 % | 41.5 % | 43.5 % | |

Reading: diarization-conditioned listening roughly halves speaker-attributed WER with *our* masks, which is the plan's central assumption (PLAN §5 gate) confirmed. The two IS1009 meetings where the diarizer's own DER is worst show the limit: mask conditioning inherits slot confusion (53.5 %), and only identity outside the mask recovers it (enrolled 25.5 %) — PLAN §2's identity-binding principle, now with a number. Whisper columns fill in as the local scoring loop finishes.

## Incidents

- **2026-09-12 05:23–11:42** — laptop idle-slept and hibernated mid-benchmark. CPU workers paused; the Metal compiler service died, killing every MLX/MPS process (Whisper scoring, toy ablation at its final eval, Dixtral after one window). Fix: all long jobs now run under `caffeinate -i`; the scoring loop had to be restarted in a fresh process context because children of the surviving shell could not reach the compiler. Lesson recorded in memory.
- **2026-09-12 ~05:00** — four research subagents died on the account's API session limit; relaunched after reset, all four completed.
- **2026-09-12 00:xx** — 5 parallel Sortformer workers ran at 1.9× realtime each (6 P-cores, 12 E-cores); reduced to a 3-worker pool, 1.2–1.4× each.
