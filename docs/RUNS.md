# Run log

One entry per job of PLAN.md §8. Nothing is "done" without its metric. Costs
come from `spend.jsonl` on the Modal volume (`modal run research/modal_app.py::spend`);
local jobs cost $0 and list wall-clock instead.

| # | job | where | started | status | cost | metric | changed the plan? |
|---|---|---|---|---|---|---|---|
| 1 | AMI benchmark (16 meetings × none/words/enrolled) | local | 2026-09-12 00:08 | running — 11/16 diarized, 17/48 scored; paused 6 h by a laptop sleep, resumed under `caffeinate` | $0 | partial: cpWER 22–44 %, attribution beats raw mix on every meeting so far; enrolled identity 61 → 26 % idWER on IS1009c | yes: identity must be a model input (PLAN §2) |
| 2 | toy conditioning ablation, 320 rooms / 2.5 k steps | local (MPS) | 2026-09-12 11:47 | running — conditioned arm at step 2 200; first attempt lost to the Metal crash at its eval step | $0 | pending: dev cpWER conditioned vs. unconditioned | — |
| 3 | Dixtral on Sortformer masks, 16 meetings | Modal 16×H100 | — | blocked: Modal token secret (`as-…`) not yet provided | est. $12 | — | — |
| 3b | Dixtral on Sortformer masks, IS1009a only | local (MPS) | queued after job 2 | first attempt: 1 window in 1 105 s (RTF 9.2) then Metal crash | $0 | — | shows why job 3 is a Modal job |
| 4 | scaled ablation, 2 000 rooms / 20 k steps, additive vs. per-layer | local (MPS) | rooms generating (CPU) | queued after 3b | $0 | — | — |
| 5 | per-scenario turn statistics; regenerate rooms | local | — | not started | $0 | — | — |
| 6 | Moshi-LoRA stage-1 pilot | Modal 1×H100 | — | not started; needs job 3 + Modal token | est. $120 | — | — |
| 7 | pilot second arm | Modal 1×H100 | — | reserve | est. $120 | — | — |
| 8 | assistant-in-the-room scripts + 100 h TTS | local | — | not started | $0 | — | — |

## Incidents

- **2026-09-12 05:23–11:42** — laptop idle-slept and hibernated mid-benchmark. CPU workers paused; the Metal compiler service died, killing every MLX/MPS process (Whisper scoring, toy ablation at its final eval, Dixtral after one window). Fix: all long jobs now run under `caffeinate -i`; the scoring loop had to be restarted in a fresh process context because children of the surviving shell could not reach the compiler. Lesson recorded in memory.
- **2026-09-12 ~05:00** — four research subagents died on the account's API session limit; relaunched after reset, all four completed.
- **2026-09-12 00:xx** — 5 parallel Sortformer workers ran at 1.9× realtime each (6 P-cores, 12 E-cores); reduced to a 3-worker pool, 1.2–1.4× each.
