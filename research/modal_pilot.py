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

import sys
for _p in ("/repo/research", str(Path(__file__).resolve().parent)):   # container mount / local checkout
    if _p not in sys.path:
        sys.path.insert(0, _p)
from modal_app import DATA, GPU_USD_H, HF_SECRET, REPO, VOL, _guard, _link_data, _record_spend  # noqa: E402,F401

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


