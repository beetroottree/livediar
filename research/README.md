# research/ — scaffold for the room-aware duplex model

The plan is `../PLAN.md`. This directory is the part of it that is code today.

| file | what it is | status |
|---|---|---|
| `conditioning.py` | additive per-frame diarization + identity conditioning, plus mask corruption | runs, shape-tested |
| `duplex_lm.py` | K-stream full-duplex LM skeleton: room audio + own audio + monologue + floor token | runs, forward/backward tested at toy size |
| `simulate.py` | room simulator with AMI-like turn-taking; exact activity + SOT transcript | runs on any 16 kHz single-speaker pool |
| `stages.yaml` | the six-stage curriculum with gates and compute | spec |

Nothing here is trained. The point is that the *interfaces* are fixed: the
model consumes exactly what `livediar/` already produces per 80 ms frame
(mixture audio, activity matrix, slot identities), and it is scored by
exactly what `bench/` already computes (cpWER, idWER on AMI).

## Step 1 this week: Dixtral on our masks

```bash
pip install "transformers>=4.50" torch
python - <<'PY'
from transformers import AutoModel, AutoProcessor
m = AutoModel.from_pretrained("BUT-FIT/Dixtral", trust_remote_code=True)
p = AutoProcessor.from_pretrained("BUT-FIT/Dixtral")
PY
```

Then build FDDT masks from `bench/ami/cache/<m>.probs.npz` through
`livediar.turns.TurnTracker` (the `last_active_frames` matrix is the STNO
source: alone / overlapped / non-target / silence per target slot), run
Dixtral per target slot, and score with `bench/ami_bench.py`'s `cp_wer`.
That number, next to the Whisper pipeline's number on the same 16 meetings,
is the stage-1 target and the first real test of the plan's central
assumption: that Sortformer masks are good enough to condition on.

## Step 2: the smallest proof of conditioning

```bash
python research/conditioning.py      # shape + permutation-equivariance + corruption
python research/duplex_lm.py         # forward/backward at toy size
python research/simulate.py pool.jsonl rooms --n 1000 --seed 0
```

Train the toy `DuplexLM` (text head only) on simulated rooms with and
without `DiarizationConditioning`; the conditioned model must win on
held-out rooms' cpWER. If it does not, fix the conditioning before spending
a GPU-hour on the real thing.
