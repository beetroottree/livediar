"""Modal jobs for the parts of PLAN.md that need real GPUs.

    modal setup                                                       # once: attach the account
    modal secret create huggingface HF_TOKEN=hf_...                   # once: HF token (Voxtral base is gated)
    modal run research/modal_app.py::sync_data                        # once: upload AMI + rooms (~2.5 GB)
    modal run --detach research/modal_app.py::dixtral_ami             # stage-1 gate, 16 meetings on 16 H100s (~$10)
    modal run --detach research/modal_app.py::toy_ablation            # conditioning proof at scale (~$6)

Modal in 2026 (docs/research/infra.md): H100 $3.95/h per-second billing, Starter plans cap at
10 concurrent GPUs (Team 50), GPU functions are preemptible, single-job clustering tops out
at 64 GPUs and needs approval. Set a workspace spend limit before running anything.

Data lives in a Modal Volume `livediar-data` laid out like bench/ami and
research/data locally; `sync_data` uploads what this repo has. Results are
written back to the volume under results/ and pulled with `modal volume get`.

Everything here is a thin wrapper: the science is in research/*.py and
bench/ami_bench.py, which run unchanged inside the container.
"""

import os
import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
VOL = modal.Volume.from_name("livediar-data", create_if_missing=True)
DATA = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch==2.10.0", "torchaudio==2.10.0", "transformers==4.55.0", "mistral_common==1.8.5",
        "peft==0.17.1", "accelerate==1.8.1", "librosa", "soundfile", "safetensors", "numpy==1.26.4",
        "intervaltree", "jiwer", "pyannote.core", "pyannote.metrics", "whisper-normalizer",
        "huggingface_hub", "wandb",
    )
    .add_local_dir(str(REPO / "livediar"), "/repo/livediar")
    .add_local_dir(str(REPO / "bench"), "/repo/bench", ignore=["ami/wav", "ami/cache", "ami/logs", "ami/manual"])
    .add_local_dir(str(REPO / "research"), "/repo/research",
                   ignore=["data", ".venv-dixtral", "dixtral_repo/.git"])
)

app = modal.App("livediar-research", image=image)
MEETINGS = ["EN2002a", "EN2002b", "EN2002c", "EN2002d", "ES2004a", "ES2004b", "ES2004c", "ES2004d",
            "IS1009a", "IS1009b", "IS1009c", "IS1009d", "TS3003a", "TS3003b", "TS3003c", "TS3003d"]


# Modal 2026 list prices per GPU-hour (docs/research/infra.md); used for the running spend estimate.
GPU_USD_H = {"H100": 3.95, "H200": 4.54, "B200": 6.25, "A100": 2.50, "A100-80GB": 2.50, "L40S": 1.95, "L4": 0.80, "T4": 0.59}


BUDGET_USD = float(os.environ.get("LIVEDIAR_BUDGET_USD", "450"))   # hard ceiling on estimated spend


def _spent() -> float:
    import json
    VOL.reload()                       # other containers append to spend.jsonl; see the latest commit
    p = Path(f"{DATA}/results/spend.jsonl")
    if not p.exists():
        return 0.0
    return sum(json.loads(l)["est_usd"] for l in p.read_text().splitlines() if l.strip())


def _guard(job: str, gpu: str, est_hours: float):
    """Refuse to start a job whose estimated cost would push total spend past BUDGET_USD."""
    n = int(gpu.split(":")[1]) if ":" in gpu else 1
    est = GPU_USD_H.get(gpu.split(":")[0], 4.0) * n * est_hours
    spent = _spent()
    if spent + est > BUDGET_USD:
        raise RuntimeError(f"budget guard: {job} would cost ~${est:.2f}, spent ${spent:.2f}, ceiling ${BUDGET_USD:.0f}")
    print(f"[budget] {job}: est ${est:.2f}, spent so far ${spent:.2f}, ceiling ${BUDGET_USD:.0f}", flush=True)


def _record_spend(job: str, gpu: str, seconds: float, note: str = ""):
    """Append one line to spend.jsonl on the volume; `modal run ...::spend` sums it."""
    import json, time
    n = int(gpu.split(":")[1]) if ":" in gpu else 1
    kind = gpu.split(":")[0]
    usd = GPU_USD_H.get(kind, 4.0) * n * seconds / 3600
    Path(f"{DATA}/results").mkdir(parents=True, exist_ok=True)
    with open(f"{DATA}/results/spend.jsonl", "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "job": job, "gpu": gpu,
                            "seconds": round(seconds), "est_usd": round(usd, 3), "note": note}) + "\n")
    VOL.commit()
    return usd


@app.function(volumes={DATA: VOL})
def spend_total() -> str:
    import json
    p = Path(f"{DATA}/results/spend.jsonl")
    if not p.exists():
        return "no spend recorded"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    tot = sum(r["est_usd"] for r in rows)
    by = {}
    for r in rows:
        by[r["job"]] = by.get(r["job"], 0) + r["est_usd"]
    return "\n".join(f"{k:24s} ${v:8.2f}" for k, v in sorted(by.items())) + f"\n{'TOTAL':24s} ${tot:8.2f}  ({len(rows)} calls)"


@app.local_entrypoint()
def spend():
    print(spend_total.remote())


def _link_data():
    """Make /repo/bench/ami/{wav,cache,manual,setup} point at the volume."""
    for d in ("wav", "cache", "manual", "setup"):
        dst = Path(f"/repo/bench/ami/{d}")
        if not dst.exists():
            dst.symlink_to(f"{DATA}/ami/{d}")
    Path("/repo/bench/ami/results").mkdir(exist_ok=True)


# HF weights are cached on the volume so 16 parallel containers download Dixtral once,
# not 16 times; GPU functions are preemptible on Modal, so jobs are idempotent (skip if the
# result already exists on the volume) and retried.
# Dixtral, Voxtral and Moshi weights are not gated; attach an HF secret only if you have one:
#   modal secret create huggingface HF_TOKEN=hf_...   and run with USE_HF_SECRET=1
HF_SECRET = [modal.Secret.from_name("huggingface")] if os.environ.get("USE_HF_SECRET") else []


@app.function(gpu="H100", timeout=6 * 3600, volumes={DATA: VOL}, secrets=HF_SECRET,
              retries=modal.Retries(max_retries=3, initial_delay=10.0))
def dixtral_one(meeting: str, win: int = 120) -> str:
    import time
    VOL.reload()                       # see files uploaded after this container started
    done = Path(f"{DATA}/results/{meeting}.dixtral.json")
    if done.exists():
        return done.read_text()
    t0 = time.time()
    _guard("dixtral_ami", "H100", 0.5)
    _link_data()
    Path(f"{DATA}/hf").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, WANDB_MODE="disabled", HF_HOME=f"{DATA}/hf",
               PYTHONPATH="/repo:/repo/bench:/repo/research/dixtral_repo:/repo/research/dixtral_repo/src")
    subprocess.run(["python", "/repo/research/dixtral_ami.py", meeting, "--win", str(win), "--device", "cuda"],
                   check=True, cwd="/repo", env=env)
    out = Path(f"/repo/bench/ami/results/{meeting}.dixtral.json").read_text()
    done.parent.mkdir(exist_ok=True)
    done.write_text(out)
    hyp = Path(f"/repo/bench/ami/results/{meeting}.dixtral.hyp.txt")
    if hyp.exists():
        Path(f"{DATA}/results/{meeting}.dixtral.hyp.txt").write_text(hyp.read_text())
    _record_spend("dixtral_ami", "H100", time.time() - t0, meeting)
    return out


@app.local_entrypoint()
def dixtral_ami(meetings: str = "all", win: int = 120):
    ms = MEETINGS if meetings == "all" else meetings.split(",")
    for m, r in zip(ms, dixtral_one.map(ms, kwargs={"win": win}, return_exceptions=True)):
        print(m, "FAILED:" if isinstance(r, Exception) else "", str(r)[:200])   # one failure must not stop the app


@app.function(gpu="A100", timeout=4 * 3600, volumes={DATA: VOL}, secrets=HF_SECRET,
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def toy_ablation_job(steps: int = 3000, rooms: int = 2000, bs: int = 8) -> dict:
    """Conditioning ablation v3 (frozen wav2vec2 listener) at scale: simulate `rooms` rooms from the
    uploaded LibriSpeech pool in-container (CPU), then train arms cond 0/1/2 on the A100."""
    import json, time
    t0 = time.time()
    _guard("toy_ablation", "A100", 3.0)
    Path(f"{DATA}/hf").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH="/repo:/repo/research", HF_HOME=f"{DATA}/hf")
    root = Path("/repo/research/data"); root.mkdir(exist_ok=True)
    rooms_dir = Path(f"{DATA}/rooms_modal")
    if not (rooms_dir / f"room{rooms - 1:05d}" / "audio.wav").exists():
        subprocess.run(["python", "/repo/research/simulate.py", f"{DATA}/pool_modal.jsonl", str(rooms_dir),
                        "--n", str(rooms), "--dur", "90", "--spk", "2,4", "--seed", "1"], check=True, cwd="/repo", env=env,
                       stdout=subprocess.DEVNULL)
        VOL.commit()
    if not (root / "rooms_modal").exists():
        (root / "rooms_modal").symlink_to(rooms_dir)
    res = {}
    for cond in (0, 1, 2):
        out = root / f"toy_result_w2v_rooms_modal_cond{cond}_s{steps}.json"
        if not out.exists():
            subprocess.run(["python", "/repo/research/toy_ablation_w2v.py", "--cond", str(cond), "--steps", str(steps),
                            "--bs", str(bs), "--rooms", "rooms_modal", "--max-rooms", str(rooms)],
                           check=True, cwd="/repo", env=env)
        res[cond] = json.loads(out.read_text())
        Path(f"{DATA}/results").mkdir(exist_ok=True)
        Path(f"{DATA}/results/toy_w2v_cond{cond}.json").write_text(json.dumps(res[cond], indent=1))
        VOL.commit()
    _record_spend("toy_ablation", "A100", time.time() - t0, f"rooms={rooms} steps={steps}")
    return res


@app.local_entrypoint()
def toy_ablation(steps: int = 3000, rooms: int = 2000):
    for c, r in toy_ablation_job.remote(steps=steps, rooms=rooms).items():
        print(c, r)


# ---------------------------------------------------------------- job 6: Moshi-LoRA stage-1 pilot
# Segmented training: each call trains `steps` steps from the latest merged checkpoint on the
# volume and saves a new one, so a preempted or timed-out container loses at most one segment.
# Loop it: `modal run --detach research/modal_app.py::pilot_stage1 --segments 6`.
pilot_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install("torch==2.6.0", "torchaudio==2.6.0", "triton>=3.2", "fire", "simple-parsing", "pyyaml",
                 "safetensors", "tensorboard", "tqdm", "sphn", "sentencepiece", "numpy", "scipy",
                 "huggingface_hub", "wandb", "pyannote.core", "pyannote.metrics", "jiwer", "whisper-normalizer")
    .add_local_dir(str(REPO / "livediar"), "/repo/livediar")
    .add_local_dir(str(REPO / "bench"), "/repo/bench", ignore=["ami/wav", "ami/cache", "ami/logs", "ami/manual", "ami/setup"])
    .run_commands(
        "git clone --depth 1 https://github.com/kyutai-labs/moshi.git /opt/moshi && pip install /opt/moshi/moshi",
        "git clone https://github.com/kyutai-labs/moshi-finetune.git /opt/moshi-finetune && cd /opt/moshi-finetune && git checkout 2acc879",
    )
    .add_local_dir(str(REPO / "research"), "/repo/research",
                   ignore=["data", ".venv-dixtral", "dixtral_repo", "moshi_finetune_repo", "moshi_repo"])
)


@app.function(image=pilot_image, gpu="H100", timeout=7 * 3600, volumes={DATA: VOL}, secrets=HF_SECRET,
              retries=modal.Retries(max_retries=2, initial_delay=30.0))
def pilot_segment(steps: int = 1000, run: str = "pilot1") -> str:
    """One training segment. Reads /data/pilot/<run>/latest.json for where to start."""
    import json, time
    t0 = time.time()
    _guard("pilot_stage1", "H100", 3.0)
    base = Path(f"{DATA}/pilot/{run}"); base.mkdir(parents=True, exist_ok=True)
    state_p = base / "latest.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {"segment": 0, "step": 0, "init_moshi": None, "init_cond": None}
    seg = state["segment"] + 1
    run_dir = base / f"seg{seg:02d}"
    env = dict(os.environ, HF_HOME=f"{DATA}/hf", PILOT_DATA_DIR=f"{DATA}/pilot", WANDB_MODE="disabled",
               PYTHONPATH="/opt/moshi-finetune:/repo/research/pilot:/repo/research")
    if state["init_moshi"]:
        env["PILOT_INIT_MOSHI"] = state["init_moshi"]
    if state["init_cond"]:
        env["PILOT_INIT_COND"] = state["init_cond"]
    # pilot data for simulated rooms is built in-container from the rooms the ablation job simulated
    # (research/pilot_data.py: stereo 24 kHz + alignments + activity); AMI pilot data is uploaded.
    if not Path(f"{DATA}/pilot/rooms/data.jsonl").exists():
        assert Path(f"{DATA}/rooms_modal").exists(), "run toy_ablation first (it simulates /data/rooms_modal)"
        subprocess.run(["python", "/repo/research/pilot_data.py", "rooms", f"{DATA}/rooms_modal", f"{DATA}/pilot/rooms"],
                       check=True, cwd="/repo", env=env)
        VOL.commit()
    # AMI pilot data (train subset + dev) is built in-container from the 16 kHz wavs + references on the volume
    _link_data()
    for split, sub in (("train", "ami"), ("dev", "ami_dev")):
        if not Path(f"{DATA}/pilot/{sub}/data.jsonl").exists():
            subprocess.run(["python", "/repo/research/pilot_data.py", "ami", f"{DATA}/pilot/{sub}", split],
                           check=True, cwd="/repo", env=dict(env, PYTHONPATH="/repo:/repo/bench:/repo/research"))
            VOL.commit()
    subprocess.run(["python", "/repo/research/pilot/build_trainer.py", "/opt/moshi-finetune"], check=True)
    cfg = Path("/opt/moshi-finetune/pilot_seg.yaml")
    y = (Path("/repo/research/pilot/pilot.yaml").read_text()
         .replace('run_dir: ""', f'run_dir: "{run_dir}"').replace("max_steps: 1000", f"max_steps: {steps}"))
    cfg.write_text(y)
    subprocess.run(["torchrun", "--nproc-per-node", "1", "train_pilot.py", str(cfg)],
                   check=True, cwd="/opt/moshi-finetune", env=env)
    ck = sorted((run_dir / "checkpoints").glob("checkpoint_*"))[-1] / "consolidated"
    state = {"segment": seg, "step": state["step"] + steps,
             "init_moshi": str(ck / "consolidated.safetensors"), "init_cond": str(ck / "diar_cond.pt")}
    state_p.write_text(json.dumps(state))
    _record_spend("pilot_stage1", "H100", time.time() - t0, f"{run} seg{seg} steps={steps}")
    return json.dumps(state)


@app.local_entrypoint()
def pilot_stage1(segments: int = 1, steps: int = 1000, run: str = "pilot1"):
    for _ in range(segments):
        print(pilot_segment.remote(steps=steps, run=run))


@app.local_entrypoint()
def sync_data(what: str = "ami"):
    """Upload data to the volume. what = ami (bench/ami wav+cache+manual+setup, ~1.2 GB) |
    rooms (research/data/rooms2k + pool, ~6 GB, only for the Modal toy job) | pilot (research/pilot_data)."""
    with VOL.batch_upload(force=True) as b:
        if what == "ami":
            for d in ("wav", "cache", "manual", "setup"):
                p = REPO / "bench" / "ami" / d
                if p.exists():
                    b.put_directory(str(p), f"ami/{d}")
        elif what == "rooms":
            b.put_directory(str(REPO / "research" / "data" / "rooms2k"), "rooms")
            b.put_file(str(REPO / "research" / "data" / "pool.jsonl"), "pool.jsonl")
        elif what == "pilot":
            b.put_directory(str(REPO / "research" / "pilot_data"), "pilot")
    print("uploaded", what)
