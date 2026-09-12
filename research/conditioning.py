"""Diarization conditioning for a frame-synchronous audio LM (PLAN.md §2).

Per 80 ms frame and per speaker slot k we know, from Sortformer + the turn
tracker, an activity state and, from voice profiles, an identity vector.
This module turns that into one additive embedding per frame that is summed
onto the audio-token embedding before the temporal transformer — the same
"frame-level diarization-dependent transformation" idea DiCoW applies inside
Whisper's encoder, applied here at the LM input.

States per slot (STNO-style, but from the listener's point of view):
    0 silent     slot is not active
    1 alone      slot is the only active speaker
    2 overlapped slot is active together with at least one other

Identity: a unit-norm speaker embedding (TitaNet-large, 192-d) per slot, or
a learned "unknown" vector when the slot has no profile yet. Identity is a
*prompt*, so enrolling someone at runtime needs no weights to change.

Corruption (PLAN.md §4.4) lives here too so the training loop can call one
function: boundary jitter, frame flips, slot swaps and a lag, all applied to
the activity matrix before the states are derived.
"""

from __future__ import annotations

import torch
from torch import nn

N_STATES = 3


def activity_to_states(active: torch.Tensor) -> torch.Tensor:
    """(B, T, K) bool activity -> (B, T, K) long states in {0, 1, 2}."""
    n_active = active.sum(-1, keepdim=True)
    states = torch.zeros_like(active, dtype=torch.long)
    states[active & (n_active == 1)] = 1
    states[active & (n_active > 1)] = 2
    return states


def corrupt_activity(active: torch.Tensor, *, jitter_frames: int = 3, flip_p: float = 0.05,
                     swap_p: float = 0.02, max_lag: int = 12,
                     generator: torch.Generator | None = None) -> torch.Tensor:
    """Make the training-time masks as unreliable as a live diarizer (§4.4).

    jitter_frames : each turn boundary moves by up to ±this many frames
    flip_p        : fraction of frames whose activity bit is flipped
    swap_p        : per-example probability of swapping two slots for a stretch
    max_lag       : masks are delayed by a random 0..max_lag frames (the live
                    decision buffer is 13 frames at the `low` preset)
    """
    B, T, K = active.shape
    out = active.clone()
    g = generator
    # boundary jitter: shift each slot's track by a per-turn random offset,
    # approximated cheaply by a per-slot circular roll of a few frames
    for b in range(B):
        for k in range(K):
            r = int(torch.randint(-jitter_frames, jitter_frames + 1, (1,), generator=g))
            if r:
                out[b, :, k] = torch.roll(out[b, :, k], r, dims=0)
    # frame flips
    flips = torch.rand(B, T, K, generator=g) < flip_p
    out = out ^ flips
    # slot swaps for a random stretch
    for b in range(B):
        if K > 1 and float(torch.rand(1, generator=g)) < swap_p:
            i, j = torch.randperm(K, generator=g)[:2].tolist()
            s = int(torch.randint(0, T, (1,), generator=g)); e = min(T, s + int(torch.randint(12, 60, (1,), generator=g)))
            out[b, s:e, i], out[b, s:e, j] = out[b, s:e, j].clone(), out[b, s:e, i].clone()
    # lag: the model sees the past, not the present
    lag = int(torch.randint(0, max_lag + 1, (1,), generator=g))
    if lag:
        out = torch.cat([torch.zeros(B, lag, K, dtype=out.dtype), out[:, :-lag]], dim=1)
    return out


class DiarizationConditioning(nn.Module):
    """Additive per-frame conditioning from slot states + slot identities.

    forward(active, identity) -> (B, T, d_model)
        active   : (B, T, K) bool
        identity : (B, K, d_id) unit-norm speaker embeddings, or None per slot
                   signalled by an all-zero row (replaced by the learned unknown)
    Each slot contributes state_emb[k][state] + proj(identity[k]) gated by
    whether the slot is active; contributions are summed over slots, so the
    embedding is permutation-equivariant in the slots — arrival order carries
    no meaning, which is what we want from Sortformer's anonymous slots.
    """

    def __init__(self, d_model: int, n_slots: int = 4, d_id: int = 192, max_gate: float = 0.1):
        super().__init__()
        self.n_slots = n_slots
        self.max_gate = max_gate
        self.state_emb = nn.Embedding(N_STATES, d_model)
        self.slot_pos = nn.Parameter(torch.zeros(n_slots, d_model))     # weak slot prior
        self.id_proj = nn.Linear(d_id, d_model, bias=False)
        self.unknown_id = nn.Parameter(torch.randn(d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)
        # zero-initialised, BOUNDED output gate: a pretrained host is untouched at step 0 and the
        # conditioning can never exceed max_gate x unit RMS. Measured 2026-09-12 on a frozen
        # wav2vec2-base: an unbounded gate drifted to -0.20 (conditioning RMS 0.25 at the encoder
        # input) and the encoder collapsed to all-blank output; at RMS 0.06 it was unaffected.
        self.gate = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.state_emb.weight, std=0.02)
        nn.init.normal_(self.slot_pos, std=0.02)

    def forward(self, active: torch.Tensor, identity: torch.Tensor | None = None) -> torch.Tensor:
        B, T, K = active.shape
        assert K == self.n_slots, f"expected {self.n_slots} slots, got {K}"
        states = activity_to_states(active)                                   # (B,T,K)
        e = self.state_emb(states) + self.slot_pos[None, None]                # (B,T,K,d)
        if identity is None:
            ident = self.unknown_id.expand(B, K, -1)
        else:
            known = identity.norm(dim=-1, keepdim=True) > 0
            ident = torch.where(known, self.id_proj(identity), self.unknown_id.expand(B, K, -1))
        e = e + ident[:, None] * active[..., None].to(e.dtype)                # identity only when talking
        return self.max_gate * torch.tanh(self.gate) * self.norm(e.sum(2))    # (B,T,d), |scale| < max_gate


class PerLayerStateConditioning(nn.Module):
    """DiCoW/FDDT-style per-layer conditioning for a K-slot listener.

    DiCoW applies, in every encoder layer, a diagonal affine transform chosen
    by the frame's STNO class for one *target* speaker, with a suppressive
    initialisation (target/overlap = identity, silence/non-target scaled
    down) that makes it trainable. A K-stream listener has no single target,
    so the per-layer class here is the frame's *global* state
    {silence, one speaker, overlap}, and the additive slot/identity
    embedding (DiarizationConditioning) still tells the model who.

        h' = h * (1 + scale[state]) + shift[state]      per layer

    scale/shift start at zero (identity), so a pretrained host is untouched
    at step 0; DiCoW's suppressive prior for silence is learned, not imposed. Used by research/toy_ablation.py
    --cond 2 as the second ablation arm.
    """

    def __init__(self, d_model: int, n_layers: int):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(n_layers, N_STATES, d_model))
        self.shift = nn.Parameter(torch.zeros(n_layers, N_STATES, d_model))
        # identity at init for every state; the suppressive prior (silence scale < 0) is
        # learned rather than imposed so a pretrained host is not perturbed at step 0

    @staticmethod
    def frame_state(active: torch.Tensor) -> torch.Tensor:
        n = active.sum(-1)                                # (B,T)
        return torch.clamp(n, max=2).long()               # 0 silence, 1 single, 2 overlap

    def forward(self, h: torch.Tensor, layer: int, state: torch.Tensor) -> torch.Tensor:
        sc = self.scale[layer][state]                     # (B,T,d)
        sh = self.shift[layer][state]
        return h * (1 + sc) + sh


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, K = 2, 50, 4
    active = torch.zeros(B, T, K, dtype=torch.bool)
    active[:, 5:30, 0] = True; active[:, 20:45, 1] = True                   # overlap 20-30
    st = activity_to_states(active)
    assert st[0, 10, 0] == 1 and st[0, 25, 0] == 2 and st[0, 25, 1] == 2 and st[0, 40, 1] == 1
    ident = torch.nn.functional.normalize(torch.randn(B, K, 192), dim=-1); ident[:, 3] = 0
    cond = DiarizationConditioning(d_model=256)
    y = cond(active, ident)
    assert y.shape == (B, T, 256)
    perm = [1, 0, 2, 3]
    y2 = cond(active[..., perm], ident[:, perm])
    print("shape ok", tuple(y.shape), "| slot-permutation drift", float((y - y2).abs().max()))
    c = corrupt_activity(active, generator=torch.Generator().manual_seed(1))
    print("corruption changed", int((c ^ active).sum()), "of", active.numel(), "frame-bits")
