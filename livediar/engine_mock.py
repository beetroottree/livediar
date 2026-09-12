"""Engine interface + a dependency-free mock engine.

The mock engine mimics the real one's cadence and output shapes: it consumes
16 kHz float32 audio and, every `chunk_len` frames (0.48 s at the low-latency
preset), emits a (chunk_len, 4) block of speaker probabilities driven by a
small conversation simulator. It reacts to input level (silence in -> silence
out) so mic capture can be sanity-checked end to end without the model.
"""

import time
from dataclasses import dataclass
from typing import List

import numpy as np

from .presets import Preset

SR = 16000
FRAME_S = 0.08


@dataclass
class StepOut:
    t0: float                # start time (s, stream clock) of first frame in this block
    probs: np.ndarray        # (T, n_spk) float32 in [0, 1]
    step_wall_s: float       # wall time spent producing this block


class BaseEngine:
    n_spk = 4

    def __init__(self, preset: Preset):
        self.preset = preset

    @property
    def info(self) -> dict:
        return {"engine": self.kind, "n_spk": self.n_spk,
                "preset": {"name": self.preset.name,
                           "buffer_s": round(self.preset.buffer_s, 2),
                           "hop_s": round(self.preset.hop_s, 2)}}

    def feed(self, pcm: np.ndarray) -> List[StepOut]:  # pragma: no cover - interface
        raise NotImplementedError

    def finalize(self) -> List[StepOut]:
        return []


class MockEngine(BaseEngine):
    kind = "mock"

    def __init__(self, preset: Preset, seed: int = 7):
        super().__init__(preset)
        self.rng = np.random.default_rng(seed)
        self.buf = np.zeros(0, dtype=np.float32)
        self.emitted_frames = 0
        self.cur = 0                      # current speaker slot
        self.hold = self.rng.integers(20, 60)  # frames until next turn change
        self.levels = np.zeros(self.n_spk, dtype=np.float32)
        self.overlap_with = -1

    def _advance_script(self):
        self.hold -= 1
        if self.hold <= 0:
            prev = self.cur
            self.cur = int(self.rng.choice([s for s in range(self.n_spk) if s != prev]))
            self.hold = int(self.rng.integers(25, 75))       # 2-6 s turns
            self.overlap_with = prev if self.rng.random() < 0.25 else -1
            self.overlap_left = int(self.rng.integers(4, 10))
        if self.overlap_with >= 0:
            self.overlap_left -= 1
            if self.overlap_left <= 0:
                self.overlap_with = -1

    def feed(self, pcm: np.ndarray) -> List[StepOut]:
        t_start = time.perf_counter()
        self.buf = np.concatenate([self.buf, pcm.astype(np.float32)])
        hop = int(self.preset.chunk_len * FRAME_S * SR)
        out: List[StepOut] = []
        while self.buf.size >= hop:
            block_audio, self.buf = self.buf[:hop], self.buf[hop:]
            room_rms = float(np.sqrt(np.mean(block_audio ** 2) + 1e-9))
            gate = min(1.0, room_rms / 0.01)  # quiet mic -> quiet room
            T = self.preset.chunk_len
            probs = np.zeros((T, self.n_spk), dtype=np.float32)
            for ti in range(T):
                self._advance_script()
                target = np.full(self.n_spk, 0.06, dtype=np.float32)
                target[self.cur] = 0.94
                if self.overlap_with >= 0:
                    target[self.overlap_with] = 0.80
                self.levels += 0.45 * (target - self.levels)
                probs[ti] = np.clip(
                    self.levels * gate + self.rng.normal(0, 0.03, self.n_spk), 0, 1)
            t0 = self.emitted_frames * FRAME_S
            self.emitted_frames += T
            out.append(StepOut(t0, probs, time.perf_counter() - t_start))
            t_start = time.perf_counter()
        return out
