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
    done = Path(f"{DATA}/results/{meeting}.dixtral.json")
    if done.exists():
        return done.read_text()
    t0 = time.time()
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
    for r in dixtral_one.map(ms, kwargs={"win": win}):
        print(r)


@app.function(gpu="A100", timeout=4 * 3600, volumes={DATA: VOL}, secrets=HF_SECRET,
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def toy_ablation_job(steps: int = 20000, rooms: int = 2000):
    """Conditioning proof with more rooms and steps than the laptop run."""
    import time
    t0 = time.time()
    Path(f"{DATA}/hf").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH="/repo:/repo/research", HF_HOME=f"{DATA}/hf")
    root = Path("/repo/research/data"); root.mkdir(exist_ok=True)
    if not (root / "rooms").exists():
        (root / "rooms").symlink_to(f"{DATA}/rooms")
    if not (root / "pool.jsonl").exists():
        (root / "pool.jsonl").symlink_to(f"{DATA}/pool.jsonl")
    subprocess.run(["python", "/repo/research/toy_ablation.py", "features"], check=True, cwd="/repo", env=env)
    for cond in (1, 0):
        subprocess.run(["python", "/repo/research/toy_ablation.py", "train", "--cond", str(cond),
                        "--steps", str(steps), "--bs", "16"], check=True, cwd="/repo", env=env)
    res = {c: (root / f"toy_result_cond{c}.json").read_text() for c in (1, 0)}
    Path(f"{DATA}/results").mkdir(exist_ok=True)
    for c, t in res.items():
        Path(f"{DATA}/results/toy_cond{c}.json").write_text(t)
    _record_spend("toy_ablation", "A100", time.time() - t0)
    return res


@app.local_entrypoint()
def toy_ablation(steps: int = 20000):
    print(toy_ablation_job.remote(steps=steps))


# ---------------------------------------------------------------- job 6: Moshi-LoRA stage-1 pilot
# Segmented training: each call trains `steps` steps from the latest merged checkpoint on the
# volume and saves a new one, so a preempted or timed-out container loses at most one segment.
# Loop it: `modal run --detach research/modal_app.py::pilot_stage1 --segments 6`.
pilot_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install("torch==2.6.0", "torchaudio==2.6.0", "triton>=3.2", "fire", "simple-parsing", "pyyaml",
                 "safetensors", "tensorboard", "tqdm", "sphn", "sentencepiece", "numpy", "scipy",
                 "huggingface_hub", "wandb")
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
