"""K-stream full-duplex LM skeleton (PLAN.md §2).

One transformer step per 80 ms frame. Inputs per frame are summed into one
embedding: room-audio codec tokens (mixture), the model's own previous audio
tokens, the inner-monologue text token, and the diarization conditioning.
Outputs per frame: next inner-monologue token, the floor decision, and the
model's own audio codebooks (predicted by a small depth transformer, as in
Moshi; here a stack of heads for clarity).

This is the *structure* the training stages in research/stages.yaml assume.
Dimensions are small defaults so the shape test runs on a laptop; the 7B
configuration only changes the numbers.

Not implemented on purpose (they are standard and would obscure the shape):
KV-cache streaming inference, the acoustic delay schedule, Mimi encode/decode.
"""

from __future__ import annotations

import torch
from torch import nn

from conditioning import DiarizationConditioning

FLOOR = ["listen", "backchannel", "speak", "yield"]


class FrameEmbedder(nn.Module):
    def __init__(self, d_model, n_codebooks, codebook_size, vocab, n_slots):
        super().__init__()
        self.room = nn.ModuleList(nn.Embedding(codebook_size + 1, d_model) for _ in range(n_codebooks))
        self.own = nn.ModuleList(nn.Embedding(codebook_size + 1, d_model) for _ in range(n_codebooks))
        self.text = nn.Embedding(vocab, d_model)
        self.diar = DiarizationConditioning(d_model, n_slots)

    def forward(self, room_codes, own_codes, text, active, identity):
        # room_codes/own_codes: (B,T,Q) long; text: (B,T) long; active: (B,T,K) bool
        e = self.text(text)
        for q, emb in enumerate(self.room):
            e = e + emb(room_codes[..., q])
        for q, emb in enumerate(self.own):
            e = e + emb(own_codes[..., q])
        return e + self.diar(active, identity)


class DuplexLM(nn.Module):
    def __init__(self, d_model=256, n_layers=4, n_heads=4, n_codebooks=8, codebook_size=2048,
                 vocab=32000, n_slots=4, max_frames=4096):
        super().__init__()
        self.embed = FrameEmbedder(d_model, n_codebooks, codebook_size, vocab, n_slots)
        self.pos = nn.Embedding(max_frames, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.text_head = nn.Linear(d_model, vocab)
        self.floor_head = nn.Linear(d_model, len(FLOOR))
        self.audio_heads = nn.ModuleList(nn.Linear(d_model, codebook_size + 1) for _ in range(n_codebooks))

    def forward(self, room_codes, own_codes, text, active, identity=None):
        B, T = text.shape
        h = self.embed(room_codes, own_codes, text, active, identity) + self.pos(torch.arange(T, device=text.device))[None]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=text.device)
        h = self.norm(self.temporal(h, mask=mask, is_causal=True))
        return {"text": self.text_head(h),                                   # (B,T,V)
                "floor": self.floor_head(h),                                 # (B,T,4)
                "audio": torch.stack([hd(h) for hd in self.audio_heads], 2)} # (B,T,Q,C+1)


def loss(out, targets, weights=(1.0, 0.5, 1.0)):
    """Stage-3 objective: monologue + floor + own audio, next-frame targets."""
    ce = nn.functional.cross_entropy
    l_text = ce(out["text"][:, :-1].flatten(0, 1), targets["text"][:, 1:].flatten())
    l_floor = ce(out["floor"][:, :-1].flatten(0, 1), targets["floor"][:, 1:].flatten())
    a = out["audio"][:, :-1]; ta = targets["own_codes"][:, 1:]
    l_audio = ce(a.flatten(0, 2), ta.flatten())
    return weights[0] * l_text + weights[1] * l_floor + weights[2] * l_audio, \
        {"text": float(l_text), "floor": float(l_floor), "audio": float(l_audio)}


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, Q, K = 2, 64, 8, 4
    m = DuplexLM()
    room = torch.randint(0, 2048, (B, T, Q)); own = torch.randint(0, 2048, (B, T, Q))
    text = torch.randint(0, 32000, (B, T)); floor = torch.randint(0, 4, (B, T))
    active = torch.rand(B, T, K) < 0.3
    out = m(room, own, text, active)
    l, parts = loss(out, {"text": text, "floor": floor, "own_codes": own})
    l.backward()
    n = sum(p.numel() for p in m.parameters())
    print(f"forward/backward ok | params {n/1e6:.1f}M | text {tuple(out['text'].shape)} floor {tuple(out['floor'].shape)} audio {tuple(out['audio'].shape)} | loss {float(l):.2f} {parts}")
