# Training and evaluation infrastructure on Modal

Research note for PLAN.md §5 (curriculum) and §7 (compute), written
2026-09-12 against Modal's live docs and client 1.5.5. Every number is
tagged **[V]** (verified this session against the cited primary source) or
**[E]** (my estimate, with the derivation shown). Prices are Modal's list
prices with no region pinning; they change, re-check `modal.com/pricing`
before committing money.

The one finding that changes the plan: §5's H100-hour budgets are
Moshi-scale pretraining budgets, not budgets derived from §4's data. A 7B
temporal transformer at 12.5 Hz processes ~440 hours of audio per H100-hour
at torchtitan efficiency, so stage 1's 40k H100-h is ~17.6M audio-hours of
exposure — Moshi's own pretraining saw 16M — against a §4 corpus of ~130k
hours, i.e. ~150 epochs. Sized from the data instead, stage 1 is 2.7k–23k
H100-h ($11k–$89k on Modal) depending on epochs and loop efficiency, not
$158k. §3 does the arithmetic; §6 says what to run first.

## 1. Modal in September 2026

### GPUs and prices [V]

From https://modal.com/pricing (per-second billing, no minimum; hourly is
the per-second rate × 3600):

| GPU | $/s | $/h | notes |
|---|---|---|---|
| B300 | 0.001972 | 7.10 | needs CUDA 13.1+ |
| B200 | 0.001736 | 6.25 | `gpu="B200+"` may land on B300 at B200 price |
| H200 SXM | 0.001261 | 4.54 | |
| H100 SXM5 | 0.001097 | 3.95 | `gpu="H100"` may be silently upgraded to H200 at H100 price; `"H100!"` pins it |
| RTX PRO 6000 | 0.000842 | 3.03 | |
| A100 80 GB | 0.000694 | 2.50 | `gpu="A100"` may upgrade 40→80 GB free |
| A100 40 GB | 0.000583 | 2.10 | |
| L40S | 0.000542 | 1.95 | |
| A10 | 0.000306 | 1.10 | |
| L4 | 0.000222 | 0.80 | |
| T4 | 0.000164 | 0.59 | |

CPU $0.0000131/core-s (min 0.125 core), memory $0.00000222/GiB-s, Volume
storage $0.09/GiB-month with the first 1 TiB free. Plans: Starter $0 with
$30/month free compute and **10 concurrent GPUs**; Team $250/month with
$100/month credit and **50 concurrent GPUs**; Enterprise custom. The
concurrency caps matter more than the prices for this plan: 64 GPUs needs a
limit raise or Enterprise, and 256 is not a thing Modal sells as one job
(see multi-node).

Region pinning multiplies GPU+CPU+memory cost: 1.15× for a broad region
(`region=["us"]`), 1.75× for a narrow one (`region=["us-west"]`), the
smaller multiplier wins if you list both
(https://modal.com/docs/guide/region-selection) [V]. Third-party blogs
quote 1.5× for broad; Modal's page says 1.15×. Leave `region` unset.

For scale: the cheapest marketplace H100s are $1.38–1.49/h and hyperscaler
on-demand is $11.68–12.29/h
(https://intuitionlabs.ai/articles/data-center-gpu-pricing-2026,
https://shattered.io/h100-h200-b200-cloud-gpu-pricing-2026/) [V as quoted].
Modal at $3.95 is 2.7× the floor; that is the price of not running a
cluster. Fine for evals, ablations and a first 7B run; worth revisiting for
anything past ~$50k.

### GPUs per container, multi-node [V]

`gpu="H100:8"` gives up to 8 per container for B300/B200/H200/H100/A100/
L4/T4/L40S (A10 caps at 4), 2,304 GB RAM max; "requesting more than 2 GPUs
per container will usually result in larger wait times"
(https://modal.com/docs/guide/gpu). Fallback lists work:
`gpu=["H100", "A100-80GB:2"]`.

Multi-node is `@modal.experimental.clustered(size, broadcast=True,
rdma=False, fabric_size=None)` stacked under `@app.function`
(signature from the 1.5.5 client). Docs
(https://modal.com/docs/guide/multi-node-training): beta; **up to 64
devices**; since 2026-05-31 every node must take the full GPU complement
(`H100:8` valid, `H100:4` not); GPU-only; gang-scheduled, so the job runs
on all nodes or not at all; 50 Gbps private IPv6 for orchestration and a
3,200 Gbps RoCE fabric with `rdma=True`; `modal.experimental.get_cluster_info()`
returns `rank`, `container_ips` for `torchrun --nnodes --node-rank
--master-addr`. Only the rank-0 return value comes back. Special cases:
support@modal.com. The companion repo
https://github.com/modal-labs/multinode-training-guide has runnable
torchrun/nanoGPT/megatron/nemo-rl/verl/ray/ms-swift/slime examples and
still says the product is "early preview and not generally accessible" —
assume you need to ask before a 64-GPU job is schedulable.

So on Modal: 8 GPUs is one container, no cluster API; 64 is the ceiling of
one cluster; 256 means four independent jobs, which is not data-parallel
training. PLAN §7's "with 256 H100s" is not a Modal plan.

### Volumes, buckets [V]

https://modal.com/docs/guide/volumes and the `modal.Volume` reference:
`Volume.from_name(name, *, environment_name=None, create_if_missing=False,
version=None, create_options=None)`; `batch_upload(force=False)` yields a
context with `put_file` / `put_directory` (force overwrites files, never
directories); `commit()` and `reload()`; background commits every few
seconds plus a final one on shutdown, so an explicit `commit()` is belt and
braces, not required. v1: 500k inode hard cap, degrades past 50k files;
v2 (`version=2`, open beta, "we cannot yet guarantee that no data will be
lost"): no file cap, 262,144 files per directory, 1 TiB per file, built for
high-concurrency writes; keep >5 concurrent commits of small changes off
v1. Throughput "up to 2.5 GB/s". Last write wins. `modal volume get` for
files over 16 MB (the web UI caps there).

`modal.CloudBucketMount(bucket_name, bucket_endpoint_url=None,
key_prefix=None, secret=None, oidc_auth_role_arn=None, read_only=False,
requester_pays=False, force_path_style=False)` mounts S3/GCS/R2 via
mountpoint-s3: sequential reads of large files are fast, but no append, no
seek+write, no rename (https://modal.com/docs/guide/cloud-bucket-mounts).
Right for a 100k-hour corpus in R2/S3 that you stream; wrong for checkpoints.

### Timeouts, preemption, long jobs [V]

Default `timeout=300`, max **86,400 s (24 h)**, separate `startup_timeout`
since 1.1.4 (https://modal.com/docs/guide/timeouts). Every Function is
preemptible by default; the container gets an interrupt with a grace
period, and Modal re-runs the same input; `nonpreemptible=True` exists
(3× CPU/memory price) **but not for GPU functions**
(https://modal.com/docs/guide/preemption). The canonical pattern is
https://modal.com/docs/examples/long-training: `timeout=86400`,
`retries=modal.Retries(initial_delay=0.0, max_retries=10)`,
`single_use_containers=True`, checkpoint to a Volume, resume from
`last.ckpt` on entry, launch with `fn.spawn(...)` under `modal run
--detach` so the laptop can sleep. That gives 10 × 24 h = 10 days per
job; a stage-1 run longer than that is several chained jobs.

`modal run` makes an ephemeral App that dies with the client unless
`--detach`; `modal deploy` persists functions you then `.spawn()` from
anywhere (https://modal.com/docs/guide/apps).

### Cost controls [V]

Workspace **budget** (caps gross usage per billing cycle) and **spend
limit** (caps net out-of-pocket; workloads stop when hit), both on Usage &
Billing; environment budgets on Team+; email/webhook alerts at thresholds
(https://modal.com/docs/guide/budgets, https://modal.com/docs/guide/billing).
Set a spend limit before the first `.map` over 16 H100s.

### Training examples in modal-examples [V]

https://modal.com/docs/examples lists: `long-training` (resumable),
`llm-finetuning` (axolotl), `unsloth_finetune`, `grpo_verl`, `grpo_trl`,
`fine_tune_asr` (Whisper), `hp_sweep_gpt`, `diffusers_lora_finetune`. No
speech-LM or multi-node example there; multi-node lives in
`multinode-training-guide`.

### `research/modal_app.py` against client 1.5.5

I installed `modal==1.5.5` and introspected every call the draft uses.
All of it exists with these signatures [V]:

- `Image.debian_slim(python_version=None, force_build=False)`
- `Image.add_local_dir(local_path, remote_path, *, copy=False, ignore=[])`,
  `ignore: Sequence[str] | Callable[[Path], bool]` — a list is a
  dockerignore-style matcher relative to `local_path`, so `"ami/wav"` and
  `"dixtral_repo/.git"` are right. Replaces the deprecated
  `copy_local_dir`; `modal.Mount` no longer exists in the package.
- `Image.pip_install(...)` and `Image.uv_pip_install(...)`; the docs now
  recommend `uv_pip_install` with tight pins.
- `Volume.from_name(..., create_if_missing=True)`, `batch_upload(force=True)`,
  `put_directory`, `put_file`, `commit()`.
- `Secret.from_name(name, *, environment_name=None, required_keys=[])`.
- `App.function(gpu: str | list[str], timeout=300, startup_timeout, retries,
  volumes, secrets, single_use_containers, region, nonpreemptible, ...)`;
  `"H100"` / `"A100"` strings are the current syntax.
- `Function.map(*input_iterators, kwargs={}, order_outputs=True,
  return_exceptions=False)` — `kwargs={"win": win}` is correct.
- `App.local_entrypoint()`, `Function.remote`, `Function.spawn`.

Nothing is deprecated. The pins (`torch==2.10.0`, `transformers==4.55.0`,
`mistral_common==1.8.5`, `peft==0.17.1`, `accelerate==1.8.1`,
`numpy==1.26.4`) all exist on PyPI [V] and are exactly
`research/dixtral_repo/requirements.txt`, which is the right choice. Fixes,
in priority order:

1. **Model weights on the Volume.** Add `.env({"HF_HOME": "/data/hf"})` to
   the image. Without it, sixteen parallel `dixtral_one` containers each
   download Voxtral-Mini-3B (~9.5 GB bf16) plus the DiCoW-large encoder
   before doing any work — that is most of the wall-clock and a chunk of
   the dollars for a ten-minute job. First container populates, the rest
   `reload()` and hit cache.
2. **Always attach the HF secret.** The `if os.environ.get("MODAL_HF_SECRET")`
   guard means a missed env var silently runs without a token. Create it
   once (`modal secret create huggingface HF_TOKEN=...`) and pass
   `secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])]`
   unconditionally; `required_keys` fails loudly at hydration if missing.
3. **Preemption.** Add `retries=modal.Retries(max_retries=2, initial_delay=0)`
   and make `dixtral_one` idempotent: return early if
   `/data/results/{meeting}.dixtral.json` exists (a re-run after preemption
   should not redo a finished meeting). Run with
   `modal run --detach research/modal_app.py::dixtral_ami`.
4. **Write results to the Volume directly.** `_link_data()` symlinks into
   `/repo/bench/ami/...`, which is an `add_local_dir(copy=False)` mount;
   writable in practice but it is the one path I have not verified on
   1.5.5. Safer: pass an output directory (`bench.ami_bench` already
   resolves `ROOT/bench/ami/results`; add a `LIVEDIAR_RESULTS` env or
   `--out` flag) and point it at `/data/results`, dropping the copy step.
5. **`startup_timeout`.** The default execution timeout also governs
   import; with 6 GB of weights loading from the Volume set
   `startup_timeout=900`.
6. **`uv_pip_install`** instead of `pip_install` for the ~2 GB torch layer;
   same pins.
7. **Ignore more.** `ignore=["__pycache__", "*.pt", "*.ckpt", ".venv*",
   "results"]` on the bench mount so a local results dir does not shadow
   the container's.
8. **Concurrency reality.** `.map` over 16 meetings on a Starter workspace
   runs 10 at a time (plan cap); fine, just expect two waves.
9. **toy_ablation_job**: `gpu="A100"` is fine (may be upgraded to 80 GB
   free); `--bs 16` matches the script's `--bs` flag. Add
   `retries` and have the script skip `features` if the cache exists so a
   preempted run does not recompute Whisper-tiny features.
10. **Data volume**: `bench/ami/wav` is 996 MB (16 files), `research/data`
    2.0 GB; v1 Volume is right for this. Use a second, `version=2` Volume
    for training checkpoints once stage-1 runs write from 8+ containers.

Stage-1 skeleton for later, unchanged in spirit from the long-training
example:

```python
@app.function(gpu="H100:8", timeout=86400, volumes={"/ckpt": CKPT, "/data": VOL},
              retries=modal.Retries(max_retries=10, initial_delay=0),
              single_use_containers=True, secrets=[hf])
@modal.experimental.clustered(size=8, rdma=True)     # 64 H100, needs Modal's OK
def train_stage1(cfg: str):
    info = modal.experimental.get_cluster_info()
    # torchrun --nnodes 8 --node-rank info.rank --master-addr info.container_ips[0] --nproc-per-node 8
    # resume from /ckpt/last.pt if present; save every N steps; Volume commits in background
```

## 2. Reusable training code

| codebase | what is public | fit for us |
|---|---|---|
| **kyutai moshi** https://github.com/kyutai-labs/moshi [V] | inference only (PyTorch bf16/int8, MLX int4/int8/bf16, Rust/candle). Model: Helium 7B temporal (32 layers, d=4096) + depth transformer (6 layers, d=1024, 16 heads), Mimi 12.5 Hz × 8 codebooks × 2048, acoustic delay 1–2 frames. Paper https://arxiv.org/abs/2410.00037: Helium 500k steps on 2.1T text tokens (4.2M-token batches); Moshi pretraining 1M steps on 7M audio-hours with **16-hour batches**; post-training 100k steps, 8-hour batches; Fisher 10k steps + instruct 30k steps. H100 + FSDP, GPU count not stated. | The architecture we are copying. The step/batch numbers are the only public calibration for a 12.5 Hz 7B audio LM. |
| **moshi-finetune** https://github.com/kyutai-labs/moshi-finetune [V] | Apache-2.0, derived from mistral-finetune. LoRA (rank ≤128 recommended) **and** full fine-tune. Data: stereo wav (left = Moshi, right = user) + per-file JSON with timestamped transcript + jsonl manifest; `annotate.py` makes the JSON. Reported: 1×H100 ≈ 12k tok/s at 39.6 GB peak; 8×H100 ≈ 10.7k tok/s (per GPU, 23.7 GB each). Their token count is `steps × GPUs × bs × duration × 9 × 12.5` (1 text + 8 audio per frame), so 12k tok/s ≈ **1,330 frames/s/GPU** for bs 16 × 100 s with LoRA. Defaults bs 16, 100 s, lr 2e-6, 2k steps. | Closest thing to a stage-3 recipe (two-stream dialogue). Extend the data loader to K streams + activity matrix + floor token; the FSDP/LoRA plumbing is done. |
| **CSM (Sesame)** https://github.com/SesameAILabs/csm [V] | 1B Llama-3.2 backbone + small Mimi-code decoder, Apache-2.0, **inference only**, in transformers ≥ 4.52.1. | Not a training codebase; useful as a reference for the 1–2B student's decoder shape. |
| **Step-Audio 2** https://github.com/stepfun-ai/Step-Audio2 [V] | Step-Audio-2-mini (Qwen2.5-7B backbone, s3tokenizer + CosyVoice codec), Apache-2.0, inference/vLLM only; no SFT/RL scripts. DuplexSLA (https://github.com/hyzhang24/DuplexSLA, arXiv 2605.20755) builds a full-duplex model on it via continued pretraining. | Candidate backbone init alongside Helium/Qwen3-Omni; not a training stack. |
| **SLAM-LLM** https://github.com/X-LANCE/SLAM-LLM, paper https://arxiv.org/abs/2601.09385 [V] | MIT. Modular encoder/projector/LLM/PEFT; DeepSpeed and FSDP; recipes for ASR, contextual ASR, ST, TTS (VALL-E-X), captioning, spatial audio (BAT), and SLAM-Omni (single-stage speech-to-speech). Claims industrial scale to 100k audio-hours. | Encoder-side path (DiCoW/Whisper encoder → 7B) if the stage-1 ablation picks the cross-attention listener. No multi-stream Mimi-token path. |
| **NeMo speechlm2 / SALM** https://github.com/NVIDIA-NeMo/Speech, https://huggingface.co/nvidia/canary-qwen-2.5b [V] | Canary-Qwen-2.5B: 90k steps on **32×A100-80GB**, 1.3B tokens, LLM frozen, encoder+projector+LoRA trained; needs NeMo trunk, PyTorch ≥ 2.6 (FSDP2). Users report the shipped example config does not match the released model (issues #14438, #14880). | Same as SLAM-LLM: encoder-side only. Sortformer already lives here, so one NeMo image covers diarizer + a SALM baseline. |
| **Dixtral** https://github.com/BUTSpeechFIT/Dixtral, paper https://arxiv.org/abs/2606.18134 [V] | Apache-2.0; Hydra configs + SLURM scripts; DiCoW v3.3 large encoder with FDDT/STNO + Voxtral-Mini-3B-2507 (LoRA on decoder). Trained on 8×A5000-24GB (QA stage on H100), 20k steps, global batch 32, grad checkpointing, bf16. AMI cpWER 19.8 % (DiariZen), 17.1 % (oracle). Long-form: 5-min chunks at diarization pauses, S decoding passes per chunk. | The stage-1 target and the encoder-path baseline, already vendored in `research/dixtral_repo`. |
| **torchtitan** https://arxiv.org/abs/2410.06511 [V] | PyTorch-native pretraining: FSDP2/TP/PP/CP, float8, compile; Llama-3.1-8B numbers below. | Right backbone trainer for stage 1 at 7B: swap the embedding/head for K-stream + depth transformer, keep parallelism. Megatron-Bridge is the alternative if you want NeMo's FP8 numbers and its performance page (https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html) no longer lists 8B models. |

None of these trains a K-stream, diarization-conditioned Moshi. The
minimal path is torchtitan for the backbone loop, moshi-finetune's data
format extended to K channels + activity, and `research/duplex_lm.py` as
the model.

## 3. Throughput and cost

### Frames per second per GPU [V numbers, E mapping]

Public 7–8B decoder training throughput on H100, per GPU:

- torchtitan Llama-3.1-8B, 8×H100, FSDP, seq 8192, bf16: **6,258 tok/s**;
  +compile 6,674; +compile+float8 9,409. 128×H100: 5,645 / 6,482
  (https://arxiv.org/html/2410.06511v3).
- NeMo 24.09 Llama3-8B, 8×H100, seq 8192, FP8: **12,273 tok/s/GPU**, 711
  TFLOP/s (https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/performance/performance_summary.html).
- moshi-finetune LoRA, 1×H100: ~1,330 frames/s (from their 12k tok/s ÷ 9)
  — small batch, adapters, no compile; a lower bound for a fine-tune loop.

A Moshi frame is one temporal-transformer position regardless of how many
streams are summed into it, so 7B tokens/s ≈ frames/s. The depth
transformer adds 8 sequential steps of a ~0.1B model per frame:
6 × 0.1e9 × 8 ≈ 5 GFLOP against 6 × 7e9 = 42 GFLOP for the backbone, +12 %
[E]. K-stream conditioning (`conditioning.py`) is an embedding sum, free.
Planning figure: **5,500 frames/s/GPU bf16 FSDP without compile, ~8,000
with compile+float8, ~11,000 FP8 Megatron-class** [E]. At 12.5 Hz:

    5,500 frames/s ÷ 12.5 = 440 audio-seconds per second = 440 audio-hours per GPU-hour
    1,330 frames/s (LoRA-class loop)                     = 106 audio-hours per GPU-hour

Cross-check against Moshi: pretraining 1M steps × 16 h = 16M audio-hours →
16M / 440 ≈ **36k H100-h**; post-training 100k × 8 h = 0.8M → 1.8k H100-h
[E]. Whether Kyutai hit torchtitan efficiency is unknown, but §5's 40k for
stage 1 is, to within noise, "pretrain Moshi from scratch". That is the
right number for the branch of §2 that initialises from a text LLM and has
to learn Mimi tokens; it is the wrong number for the branch that
initialises from Moshi and continues pretraining on room audio.

### What §5's budgets buy vs what §4's data needs [E]

§4 targets: 4.1 room audio 20k h, 4.2 speaker-attributed monologue 100k h
regenerated per epoch, 4.3 duplex scripts 10k h. Stage 1 sees ~120k h per
epoch.

| stage | §5 budget (H100-h) | audio-hours the budget processes at 440/GPU-h | epochs over its §4 data | data-derived need, 10 epochs, 440 audio-h/GPU-h |
|---|---|---|---|---|
| 1 room listening | 40,000 | 17.6M | ~150 over 120k h | 1.2M audio-h → **2,730 H100-h** |
| 2 identity | 10,000 | 4.4M | ~37 | 1.2M → 2,730 (fewer epochs likely suffice) |
| 3 duplex | 30,000 | 13.2M | ~1,300 over 10k h | 100k → 230 (audio-out streams add no frames) |
| 4 RL timing | 10,000 | — | rollout-bound, not frame-bound | ~1–3k [E, weak: generation at ~20 ms/frame batched] |
| 5 distil | 15,000 | teacher forward ≈ ⅓ of a 7B step + 1–2B student ≈ ⅐ | | 1.3M audio-h × 0.48 / 440 ≈ 1,400 |
| **total** | **105,000** | | | **≈ 9k H100-h** |

Pessimistic version: 20 epochs and a LoRA-class loop at 106 audio-h per
GPU-h is 8.3× worse → stage 1 ≈ 22,600 H100-h, programme ≈ 75k H100-h,
which is within reach of §5's total. So the honest range for the
Moshi-initialised programme is **9k–75k H100-h**, and §5's 105k is the
ceiling that only a from-scratch pretraining needs. Recommend re-deriving
each row as `frames × epochs / (frames/s/GPU)` from the data programme's
actual hours, and keeping the §5 figures only for the text-LLM-init branch
of §2. Job 6 in the execution plan measures the real frames/s/GPU so the
divisor stops being a guess.

The reason the Moshi-init branch is cheap: Moshi's temporal transformer
already speaks Mimi tokens, so stage 1 is continued pretraining and the
7M-hour, ~36k-H100-h pretraining is inherited, not repeated. That is also
the argument for preferring Moshi init over Qwen3-Omni init in the stage-1
ablation unless the latter wins by a lot.

### Modal dollars for the §5 budgets as written [V price × E need]

H100 at $3.95/h, no region multiplier, no idle time:

| stage | H100-h | Modal $ | 8 GPUs (days) | 64 GPUs (days) | 256 GPUs (days) |
|---|---|---|---|---|---|
| 0 codec + init | 2,000 | 7,900 | 10.4 | 1.3 | 0.3 |
| 1 room listening | 40,000 | 158,000 | 208 | 26 | 6.5 |
| 2 identity | 10,000 | 39,500 | 52 | 6.5 | 1.6 |
| 3 duplex | 30,000 | 118,500 | 156 | 19.5 | 4.9 |
| 4 RL timing | 10,000 | 39,500 | 52 | 6.5 | 1.6 |
| 5 distil | 15,000 | 59,250 | 78 | 9.8 | 2.4 |
| **total (§5 rows 1–5)** | **105,000** | **414,750** | 547 | 68 | 17 |
| total incl. stage 0 | 107,000 | 422,650 | 557 | 70 | 17.4 |

Wall-clock assumes perfect scaling and 24 h/day; torchtitan's 8→128 GPU
FSDP efficiency is 90 %, so add ~10 % at 64. The 256-GPU column is
arithmetic only: Modal's cluster ceiling is 64 devices. On B200 ($6.25/h)
a bf16 training step runs roughly 2–2.2× an H100's (public torchtitan
numbers on B200 are still sparse) [E], so a B200-hour costs 1.58× and does
~2.1× the work → **~25 % cheaper per unit of training** and half the
wall-clock; worth a benchmark before stage 1 proper. Region pinning would
add 15 %; ten-day chained jobs at 10 retries each are free.

Data-derived version of the same table (10 epochs, 5,500 frames/s, all
stages, Moshi init) comes to roughly **9k H100-h ≈ $36k** on Modal;
**75k H100-h ≈ $300k** at the LoRA-loop, 20-epoch worst case. Plan the
Modal budget around $40k with a hard spend limit, and treat anything past
~$50k as the point where a reserved cluster at $1.5–2.5/h beats Modal.

## 4. Serving

**7B on one H100, 80 ms frames.** Moshi reports 160 ms theoretical
(80 ms frame + 80 ms acoustic delay) and 200 ms practical **on an L4**
(https://github.com/kyutai-labs/moshi, https://arxiv.org/abs/2410.00037)
[V], so the per-step model time on an L4 (300 GB/s) is already under
80 ms. A 7B bf16 decode step is bandwidth-bound: 14 GB / 3.35 TB/s ≈ 4 ms
of weight traffic on H100, plus 8 sequential depth-transformer steps (~1 ms
each with CUDA graphs, ~3–5 ms each eagerly) and Mimi encode/decode
(<10 ms per PLAN §3) → **15–25 ms per frame with CUDA graphs, 30–50 ms
eager** [E]. PLAN §3's <40 ms budget holds on H100 provided the step is
graph-captured (Moshi's PyTorch backend does this); it does not hold in
eager mode with K-stream conditioning glued on in Python. Batch-1 serving
uses ~5 % of an H100; the same GPU can serve ~10–20 rooms if the temporal
step is batched across sessions [E].

**Apple silicon, 1–2B student.** `moshi_mlx` runs the *7B* Moshi at
int4/int8 on a "MacBook Pro M3" (the only tested machine in the README;
no numbers published) [V]. A 1–2B student in bf16 is 2–4 GB of weights;
on an M3/M4 Max (400–546 GB/s) that is 5–10 ms of weight traffic per
temporal step, plus 8 depth steps of a much smaller model, so **~15–25 ms
per 80 ms frame, RTF ≈ 0.2–0.3** [E] — consistent with §5's RTF < 0.5
gate and with community MLX/llama.cpp figures of ~87 tok/s for
Llama-3.2-1B Q8 (https://www.kunalganglani.com/llm-benchmarks, chip
unspecified) [V as quoted]. Sortformer at 50 ms/step on CPU (PLAN §3) is the
tighter on-device budget, not the LM. Plain M-series (100–150 GB/s)
laptops need the int4 student.

## 5. Evaluation infrastructure

**Dixtral over AMI (9 h, 16 meetings) on one H100.** No published RTF
[V: absent from paper and repo]. From `research/dixtral_ami.py`: 120-s
windows (270 for 9 h), per window S ≤ 4 DiCoW encoder passes (Whisper-large
encoder on 30-s chunks, <1 s total on H100) then one batched
`generate(max_new_tokens=1024)` over the present speakers. HF `generate`
on a 3B decoder in bf16 runs ~30–60 steps/s single-stream on H100
(launch-bound, not bandwidth-bound); the longest speaker in a 2-min window
is ~300–600 tokens → 10–20 s per window, worst case 1024 steps ≈ 20–35 s.
**RTF ≈ 0.1–0.2 → 55–110 min for 9 h, ≈ 1–2 H100-h ≈ $4–8 sequential**
[E]. With `.map` over 16 meetings: each container pays model load
(~2–3 min with weights on the Volume, ~10 min if downloading) plus 4–7
min of work → ≈ 2–3 GPU-h ≈ **$8–12, ~10–15 min wall** [E]. Voxtral-Mini
needs ~9.5 GB VRAM (https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)
[V], so an L40S ($1.95) or A100 ($2.50) would do; H100 is worth it only
for wall-clock.

**Whisper large-v3-turbo.** On one H100 SXM with HF transformers fp16 +
compile: **597× real time** on 120-s clips (RTF 0.002), 490× on 30-s,
222× on 10-s; batching 10-s clips scales 219× → 404× from bs 1 → 32 with
only 14 % marginal efficiency past bs 8; MLPerf-style offline peak 1,292×
(https://inferencebench.io/blog/whisper-large-v3-turbo-597x-realtime-asr-benchmark/)
[V]. 9 h of AMI is **~1 GPU-minute of decoding**; the bench pipeline's
segmentation, diarization masking and scoring will dominate — budget
10–20 min on an L4/A10 at well under $1 [E].

**Sortformer.** `nvidia/diar_streaming_sortformer_4spk-v2` model card,
RTX 6000 Ada: RTF **0.002** at the 30.4-s-latency preset, 0.005 at 10 s,
**0.093 at 1.04 s** (chunk 6, right context 7, FIFO 188), 0.180 at 0.32 s
(https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) [V].
9 h at the `low` preset (1.04 s, the one PLAN §3 serves with) ≈ 50 min on
one GPU; offline preset ≈ 1 min. Our masks for AMI are already cached
(`bench/ami/cache`, 3.6 MB), so the Dixtral job needs no diarizer GPU time.

## 6. Modal execution plan

Order, with GPU, count, hours and list-price dollars. Set a workspace
spend limit of $500 before job 1 and raise it deliberately.

| # | job | GPU | count | GPU-h | $ | gate |
|---|---|---|---|---|---|---|
| 1 | `sync_data` (3 GB to Volume) | — | — | — | ~0 | Volume `ls` shows ami/wav ×16 |
| 2 | Dixtral AMI eval, `dixtral_ami --meetings all`, after the fixes in §1 | H100 | 16 parallel (10 on Starter) | 2–3 | **8–12** [E] | cpWER per meeting next to Whisper's; the stage-1 target |
| 3 | Whisper baseline on GPU (optional, `whisper_ami`) | L4 | 4 | <1 | <1 [E] | parity with laptop numbers |
| 4 | Toy ablation, 20k steps × 2 arms | A100 | 1 | 2–3 | **5–8** [E] | conditioned < unconditioned cpWER on held-out rooms |
| 5 | 1B ablations (§5: STNO vs QK-bias, identity emb vs prompt, ± DiCoW path, corruption): 8 arms × 2 epochs over 4.1 (40k audio-h = 1.8B frames) at ~38k frames/s/GPU for 1B [E] | H100:8 | 1 node | ~13 each → 105 | **~420** [E] | picks the stage-1 architecture |
| 6 | Stage-1 7B pilot: Moshi-init, 2 epochs over 4.1 (40k audio-h at 440/GPU-h), one 24-h job with checkpoint/resume | H100:8 | 1 node | ~90 (≈ 11 h wall) | **~360** [E] | loss sane, AMI cpWER moving; **measures real frames/s/GPU** |
| 7 | Stage 1 proper, data-derived (10–20 epochs over 120k h, 440–106 audio-h/GPU-h) | H100:8 ×8, `clustered(size=8, rdma=True)` | 64 | 2,700–22,600 | **11k–89k** [E] | cpWER ≤ Dixtral (19.8 %) with our masks; 2–15 days wall |
| 7' | Stage 1 as budgeted in §5 (from-scratch branch) | same | 64 | 40,000 | 158,000 | 26 days; needs Enterprise concurrency |

Jobs 1–6 total about $800 and fit a Team plan (50 GPUs); job 5 is the
first one that wants `H100:8`, and the doc warns >2 GPUs per container
queues longer, so submit it in the evening. Job 7 needs (a) Modal's
sign-off on clustered functions, (b) a `version=2` checkpoint Volume,
(c) the torchtitan-based loop with K-stream inputs, and (d) job 6's
measured frames/s to size it. Do not book job 7' until §5's budgets have
been re-derived from data hours; the honest reading of the public numbers
is that the Moshi-initialised 7B programme on Modal is a $36k–$300k
affair, and only the from-scratch branch needs §7's $400k+.

## Sources

Modal: https://modal.com/pricing · https://modal.com/docs/guide/gpu ·
https://modal.com/docs/guide/multi-node-training ·
https://github.com/modal-labs/multinode-training-guide ·
https://modal.com/docs/guide/volumes · https://modal.com/docs/reference/modal.Volume ·
https://modal.com/docs/guide/cloud-bucket-mounts · https://modal.com/docs/guide/timeouts ·
https://modal.com/docs/guide/preemption · https://modal.com/docs/examples/long-training ·
https://modal.com/docs/guide/apps · https://modal.com/docs/guide/region-selection ·
https://modal.com/docs/guide/budgets · https://modal.com/docs/guide/billing ·
https://modal.com/docs/examples · https://modal.com/docs/reference/modal.Function ·
https://modal.com/docs/reference/modal.Secret · https://modal.com/docs/guide/images ·
client introspection: `modal==1.5.5` from PyPI, 2026-09-12.
Market prices: https://intuitionlabs.ai/articles/data-center-gpu-pricing-2026 ·
https://shattered.io/h100-h200-b200-cloud-gpu-pricing-2026/.
Training code: https://github.com/kyutai-labs/moshi · https://github.com/kyutai-labs/moshi-finetune ·
https://arxiv.org/abs/2410.00037 · https://github.com/SesameAILabs/csm ·
https://github.com/stepfun-ai/Step-Audio2 · https://github.com/hyzhang24/DuplexSLA ·
https://github.com/X-LANCE/SLAM-LLM · https://arxiv.org/abs/2601.09385 ·
https://github.com/NVIDIA-NeMo/Speech · https://huggingface.co/nvidia/canary-qwen-2.5b ·
https://github.com/BUTSpeechFIT/Dixtral · https://arxiv.org/abs/2606.18134 ·
https://arxiv.org/abs/2410.06511 (torchtitan) ·
https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/performance/performance_summary.html ·
https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html.
Serving/eval: https://huggingface.co/mistralai/Voxtral-Mini-3B-2507 ·
https://inferencebench.io/blog/whisper-large-v3-turbo-597x-realtime-asr-benchmark/ ·
https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2 ·
https://www.kunalganglani.com/llm-benchmarks.
