"""Hooks that turn moshi-finetune into the stage-1 "room listening" pilot (PLAN.md §8 job 6).

What changes relative to upstream moshi-finetune (kyutai-labs/moshi-finetune @ 2acc879):

  1. Dataset: every Sample also carries `activity`, the (T, 4) bool matrix of
     who is talking per 80 ms frame, read from `<wav>.activity.npy` next to the
     audio and sliced to the sample's window (research/pilot_data.py writes
     both). Batch.collate stacks it to (B, T, 4).
  2. Model: a DiarizationConditioning module (research/conditioning.py, ~5 M
     params at dim 4096) maps activity to one vector per frame; it enters the
     temporal transformer through Moshi's own `condition_tensors` / fuser path
     ("sum" fusing). Upstream's fuser asserts a single vector per sequence;
     `patch_fuser()` lifts that to per-timestep tensors.
  3. Alignment: forward_text sees delayed_codes[:, :, :-1]; column 0 is the
     initial token and column s holds stream k's code from frame s-1-delay_k.
     The user (room) stream's first codebook has delay 0 in moshiko, so the
     conditioning for column s is activity[s-1]; we shift by 1 + that delay.
  4. Objective: unchanged - text loss on the (speaker-tagged room transcript)
     text stream + audio loss on Moshi's own (silent) stream. Set
     first_codebook_weight_multiplier low in the config so the silent audio
     does not dominate.
  5. Checkpoints: the conditioning module is saved next to each upstream
     checkpoint; segments restart from a merged consolidated checkpoint (see
     modal_app.py::pilot_stage1) - no in-place FSDP resume needed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # research/ -> conditioning.py
from conditioning import DiarizationConditioning  # noqa: E402

FRAME_S, K = 0.08, 4
DATA_DIR = Path(os.environ.get("PILOT_DATA_DIR", "."))


# ------------------------------------------------------------------ dataset
def load_activity(path: str, start_sec: float, T: int) -> torch.Tensor:
    p = Path(path)
    if not p.is_absolute():
        p = DATA_DIR / p
    act_path = Path(str(p)[: -len(p.suffix)] + ".activity.npy")
    out = torch.zeros(T, K, dtype=torch.bool)
    if not act_path.exists():
        return out
    a = np.load(act_path)
    f0 = int(round(start_sec / FRAME_S))
    seg = a[f0:f0 + T]
    out[: seg.shape[0], : min(seg.shape[1], K)] = torch.from_numpy(seg[:, :K].astype(bool))
    return out


def patch_dataset():
    from finetune.data import interleaver as IL

    orig_call = IL.InterleavedTokenizer.__call__

    def call(self, wav, start_sec, path):
        sample = orig_call(self, wav, start_sec, path)
        T = sample.codes.shape[-1]
        sample.activity = load_activity(path, start_sec, T)
        return sample

    IL.InterleavedTokenizer.__call__ = call

    orig_collate = IL.Batch.collate.__func__

    def collate(cls, batch):
        b = orig_collate(cls, batch)
        b.activity = torch.stack([s.activity for s in batch])         # (B, T, K)
        return b

    IL.Batch.collate = classmethod(collate)


# -------------------------------------------------------------------- model
def patch_fuser():
    """Allow per-timestep 'sum' conditions (upstream asserts shape[1] == 1)."""
    from moshi.conditioners import base as CB

    def get_sum(self, conditions):
        out = None
        for name in self.fuse2cond["sum"]:
            cond = conditions[name].tensor                                  # (B, T, C)
            out = cond if out is None else out + cond
        return out

    CB.ConditionFuser.get_sum = get_sum


def build_conditioning(model, dim: int, device="cuda"):
    """Create the conditioning module and attach a sum-fuser to the (FSDP-wrapped) LM."""
    from moshi.conditioners import ConditionFuser
    cond = DiarizationConditioning(dim, K).to(device)
    fuser = ConditionFuser(fuse2cond={"sum": ["diar"], "cross": [], "prepend": []})
    inner = getattr(model, "module", model)          # FSDP wrapper -> LMModel
    inner.fuser = fuser
    return cond


def make_condition(cond_module, activity: torch.Tensor, model, T: int):
    """activity (B, T, K) bool -> {"diar": ConditionType((B, T, C), mask)} aligned with forward_text's columns."""
    from moshi.conditioners import ConditionType
    inner = getattr(model, "module", model)
    user_first_cb = inner.audio_offset + inner.dep_q
    shift = 1 + int(inner.delays[user_first_cb])
    act = activity.to(device="cuda")
    act = torch.cat([torch.zeros(act.shape[0], shift, K, dtype=act.dtype, device=act.device), act], 1)[:, :T]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        c = cond_module(act)                                               # (B, T, C)
    mask = torch.ones(c.shape[0], c.shape[1], dtype=torch.bool, device=c.device)
    return {"diar": ConditionType(c, mask)}


# -------------------------------------------------------------- checkpoints
def save_cond(run_dir, step: int, cond_module):
    p = Path(run_dir) / "checkpoints" / f"checkpoint_{step:06d}" / "consolidated"
    p.mkdir(parents=True, exist_ok=True)
    torch.save(cond_module.state_dict(), p / "diar_cond.pt")


def load_cond(cond_module, path: str | None):
    if path and Path(path).exists():
        cond_module.load_state_dict(torch.load(path, map_location="cuda"))
        return True
    return False


def apply_all():
    patch_dataset()
    patch_fuser()


if __name__ == "__main__":
    # unit check without Moshi: activity slicing + alignment shift
    import tempfile
    d = Path(tempfile.mkdtemp()); a = np.zeros((1125, 2), bool); a[100:200, 1] = True
    np.save(d / "x.activity.npy", a)
    t = load_activity(str(d / "x.wav"), start_sec=8.0, T=50)                # frames 100..150
    assert t.shape == (50, 4) and t[:, 1].all() and not t[:, 0].any(), t.sum(0)
    print("activity slicing ok", tuple(t.shape))
