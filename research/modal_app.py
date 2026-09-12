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
HF_SECRET = [modal.Secret.from_name("huggingface")]   # create with: modal secret create huggingface HF_TOKEN=...


@app.function(gpu="H100", timeout=6 * 3600, volumes={DATA: VOL}, secrets=HF_SECRET,
              retries=modal.Retries(max_retries=3, initial_delay=10.0))
def dixtral_one(meeting: str, win: int = 120) -> str:
    done = Path(f"{DATA}/results/{meeting}.dixtral.json")
    if done.exists():
        return done.read_text()
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
    VOL.commit()
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
    VOL.commit()
    return res


@app.local_entrypoint()
def toy_ablation(steps: int = 20000):
    print(toy_ablation_job.remote(steps=steps))


@app.local_entrypoint()
def sync_data():
    """Upload bench/ami/{wav,cache,manual,setup} and research/data/{pool.jsonl,pool_wav,rooms} to the volume."""
    with VOL.batch_upload(force=True) as b:
        for d in ("wav", "cache", "manual", "setup"):
            p = REPO / "bench" / "ami" / d
            if p.exists():
                b.put_directory(str(p), f"ami/{d}")
        for d in ("rooms", "pool_wav"):
            p = REPO / "research" / "data" / d
            if p.exists():
                b.put_directory(str(p), d)
        pool = REPO / "research" / "data" / "pool.jsonl"
        if pool.exists():
            b.put_file(str(pool), "pool.jsonl")
    print("uploaded")
