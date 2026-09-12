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


def _spend_rows():
    """One JSON file per call under results/spend/ — concurrent appends to a single file lost
    entries (14 containers wrote spend.jsonl at once and 2 survived). Legacy spend.jsonl is still read."""
    import json
    VOL.reload()
    rows = []
    d = Path(f"{DATA}/results/spend")
    if d.exists():
        rows += [json.loads(f.read_text()) for f in d.glob("*.json")]
    p = Path(f"{DATA}/results/spend.jsonl")
    if p.exists():
        rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows


def _spent() -> float:
    return sum(r["est_usd"] for r in _spend_rows())


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
    import uuid
    d = Path(f"{DATA}/results/spend"); d.mkdir(parents=True, exist_ok=True)
    (d / f"{int(time.time())}-{job}-{uuid.uuid4().hex[:6]}.json").write_text(json.dumps(
        {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "job": job, "gpu": gpu,
         "seconds": round(seconds), "est_usd": round(usd, 3), "note": note}))
    VOL.commit()
    return usd


@app.function(volumes={DATA: VOL})
def spend_total() -> str:
    rows = _spend_rows()
    if not rows:
        return "no spend recorded"
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


@app.function(volumes={DATA: VOL}, timeout=1800)
def debug_sim(n: int = 3) -> str:
    """Run the simulator for a few rooms in-container and return its stderr (CPU only, cents)."""
    r = subprocess.run(["python", "/repo/research/simulate.py", f"{DATA}/pool_modal.jsonl", f"{DATA}/rooms_debug",
                        "--n", str(n), "--dur", "30", "--spk", "2,3", "--seed", "1"], cwd="/repo",
                       capture_output=True, text=True, env=dict(os.environ, PYTHONPATH="/repo:/repo/research"))
    return f"exit {r.returncode}\nSTDOUT:\n{r.stdout[-1500:]}\nSTDERR:\n{r.stderr[-3000:]}"


@app.local_entrypoint()
def debug_simulate(n: int = 3):
    print(debug_sim.remote(n=n))


@app.local_entrypoint()
def toy_ablation(steps: int = 3000, rooms: int = 2000):
    for c, r in toy_ablation_job.remote(steps=steps, rooms=rooms).items():
        print(c, r)


# Job 6 (Moshi-LoRA stage-1 pilot) lives in research/modal_pilot.py: its image (torch 2.6 + moshi)
# is heavy, and every `modal run` of an app builds all of that app's images.

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
