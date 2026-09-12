# research/ — code for the room-aware model programme

The plan is `../PLAN.md`; run history is `../docs/RUNS.md`; the reasoning
behind each choice is `../docs/DECISIONS.md`; the deep-research reports are
`../docs/research/`. This directory is the part of the plan that is code.

| file | what it is | status |
|---|---|---|
| `conditioning.py` | additive per-frame diarization + identity conditioning (zero-gated at init), the DiCoW-style per-layer state affine, and mask corruption | runs; shape/permutation tests in `__main__` |
| `duplex_lm.py` | K-stream full-duplex LM skeleton: room audio + own audio + monologue + floor token | runs; forward/backward at toy size |
| `simulate.py` | room simulator; turn-taking parameters fitted to AMI (`fit_turns.py`); exact activity + serialized transcript | runs on any 16 kHz single-speaker pool (LibriSpeech dev-clean used) |
| `fit_turns.py` | fits the simulator's schedule to reference RTTMs | run on AMI train+dev |
| `toy_ablation_w2v.py` | **the valid conditioning ablation (v3)**: frozen wav2vec2-base-960h, four trainable `<spk k>` outputs + conditioning; arms cond 0/1/2 | running locally |
| `toy_ablation.py` | v1/v2 of the ablation (from-scratch CTC; did not learn — kept for the record, see `docs/RUNS.md` job 2) | superseded |
| `dixtral_ami.py` | stage-1 gate: Dixtral (DiCoW + Voxtral 3B) driven by Sortformer masks on AMI, scored with `bench/ami_bench.py` cpWER | runs (needs `.venv-dixtral`, pinned torch 2.10 / transformers 4.55) |
| `pilot_data.py` | builds the Moshi-LoRA pilot dataset in moshi-finetune's format (stereo 24 kHz + alignments + activity) | smoke-tested on 30 rooms |
| `pilot/pilot_hooks.py` | dataset/collate/fuser patches and the conditioning injection for moshi-finetune | unit-tested without Moshi |
| `pilot/build_trainer.py` | generates `train_pilot.py` from upstream moshi-finetune @2acc879 by anchored replacements | generates cleanly |
| `pilot/pilot.yaml` | pilot config (LoRA r64, 100 s sequences, batch 4, merged checkpoints) | spec |
| `modal_app.py` | Modal jobs: `dixtral_ami` (16×H100), `toy_ablation_job` (A100), `pilot_segment`/`pilot_stage1` (segmented H100 training), `sync_data`, `spend` | verified against modal 1.5.5; blocked on the workspace payment method |
| `stages.yaml` | the six-stage curriculum with gates and data-derived compute | spec |

Scratch that is deliberately not committed (`.gitignore`): `data/` (LibriSpeech
pool, simulated rooms, checkpoints, result JSONs — copied to `results/` when
final), `dixtral_repo/`, `moshi_finetune_repo/`, `moshi_repo/`, `.venv-dixtral/`.

## Running things

```bash
# local (M5 Pro): simulation + the conditioning ablation
python research/simulate.py research/data/pool.jsonl research/data/rooms --n 320 --dur 90 --spk 2,4
for c in 0 1 2; do python research/toy_ablation_w2v.py --cond $c --steps 1500 --bs 4 --rooms rooms; done

# local: Dixtral on one AMI meeting (slow here, ~4x realtime on MPS)
PYTHONPATH=.:bench:research/dixtral_repo:research/dixtral_repo/src WANDB_MODE=disabled \
  research/.venv-dixtral/bin/python research/dixtral_ami.py IS1009a --win 120

# Modal (after `modal token set ...` and a payment method on the workspace)
modal volume put livediar-data bench/ami/wav ami/wav        # etc. — or modal run research/modal_app.py::sync_data
modal run --detach research/modal_app.py::dixtral_ami       # 16 meetings, ~$12
modal run research/modal_app.py::spend                      # running total
python research/pilot_data.py rooms research/data/rooms2k research/pilot_data/rooms
modal run research/modal_app.py::sync_data --what pilot
modal run --detach research/modal_app.py::pilot_stage1 --segments 6 --steps 1000
```
