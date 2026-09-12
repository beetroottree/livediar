"""Streaming latency presets for Sortformer v2 / v2.1.

Values match NVIDIA's published configurations (HF model cards; also embedded
in the CoreML/ONNX ports). All units are 80 ms encoder frames.

Input-buffer latency = (chunk_len + chunk_right_context) * 0.08 s.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    name: str
    chunk_len: int
    chunk_left_context: int
    chunk_right_context: int
    fifo_len: int
    spkcache_len: int
    spkcache_update_period: int

    @property
    def buffer_s(self) -> float:
        return (self.chunk_len + self.chunk_right_context) * 0.08

    @property
    def hop_s(self) -> float:
        """New audio consumed per streaming step."""
        return self.chunk_len * 0.08


PRESETS = {
    # 1.04 s input buffer — the sweet spot for live "who is speaking now".
    "low": Preset("low", chunk_len=6, chunk_left_context=1, chunk_right_context=7,
                  fifo_len=188, spkcache_len=188, spkcache_update_period=144),
    # 0.32 s input buffer — snappiest, noticeably less accurate, highest RTF.
    "ultra": Preset("ultra", chunk_len=3, chunk_left_context=1, chunk_right_context=1,
                    fifo_len=188, spkcache_len=188, spkcache_update_period=144),
    # 10.0 s input buffer — near-offline accuracy, still streaming.
    "high": Preset("high", chunk_len=124, chunk_left_context=1, chunk_right_context=1,
                   fifo_len=124, spkcache_len=188, spkcache_update_period=144),
    # 30.4 s input buffer — NVIDIA's best-quality streaming config.
    "vhigh": Preset("vhigh", chunk_len=340, chunk_left_context=1, chunk_right_context=40,
                    fifo_len=40, spkcache_len=188, spkcache_update_period=300),
}

DEFAULT_PRESET = "low"
