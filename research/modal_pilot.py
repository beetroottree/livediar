"""Modal app for job 6 only: the Moshi-LoRA stage-1 pilot (PLAN.md §8).

Kept separate from research/modal_app.py so the cheap gate/debug jobs never wait on this
image (torch 2.6 + moshi + moshi-finetune). Shares the `livediar-data` volume, the spend log
and the budget guard.

    modal run --detach research/modal_pilot.py::pilot_stage1 --segments 1 --steps 1000
"""

import os
import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
VOL = modal.Volume.from_name("livediar-data", create_if_missing=True)
DATA = "/data"
GPU_USD_H = {"H100": 3.95, "H200": 4.54, "B200": 6.25, "A100": 2.50, "A100-80GB": 2.50, "L40S": 1.95, "L4": 0.80, "T4": 0.59}
HF_SECRET = [modal.Secret.from_name("huggingface")] if os.environ.get("USE_HF_SECRET") else []
BUDGET_USD = float(os.environ.get("LIVEDIAR_BUDGET_USD", "450"))

# --- helpers copied from modal_app.py (the container cannot import that module at load time) ---
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


def _link_data():
    """Make /repo/bench/ami/{wav,cache,manual,setup} point at the volume."""
    for d in ("wav", "cache", "manual", "setup"):
        dst = Path(f"/repo/bench/ami/{d}")
        if not dst.exists():
            dst.symlink_to(f"{DATA}/ami/{d}")
    Path("/repo/bench/ami/results").mkdir(exist_ok=True)


app = modal.App("livediar-pilot")


# Segmented training: each call trains `steps` steps from the latest merged checkpoint on the
# volume and saves a new one, so a preempted or timed-out container loses at most one segment.
# Loop it: `modal run --detach research/modal_app.py::pilot_stage1 --segments 6`.
pilot_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install("torch==2.6.0", "torchaudio==2.6.0", "triton>=3.2", "fire", "simple-parsing", "pyyaml",
                 "safetensors", "tensorboard", "tqdm", "sphn", "sentencepiece", "numpy", "scipy",
                 "huggingface_hub", "wandb", "pyannote.core", "pyannote.metrics", "jiwer", "whisper-normalizer")
    .run_commands(
        "git clone --depth 1 https://github.com/kyutai-labs/moshi.git /opt/moshi && pip install /opt/moshi/moshi",
        "git clone https://github.com/kyutai-labs/moshi-finetune.git /opt/moshi-finetune && cd /opt/moshi-finetune && git checkout 2acc879",
    )
    .add_local_dir(str(REPO / "livediar"), "/repo/livediar")
    .add_local_dir(str(REPO / "bench"), "/repo/bench", ignore=["ami/wav", "ami/cache", "ami/logs", "ami/manual", "ami/setup"])
    .add_local_dir(str(REPO / "research"), "/repo/research",
                   ignore=["data", "pilot_data", "results", ".venv-dixtral", "dixtral_repo/.git", "moshi_finetune_repo", "moshi_repo", "__pycache__"])
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


