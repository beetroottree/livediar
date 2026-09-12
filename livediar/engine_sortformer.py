"""Live streaming adapter for NVIDIA Streaming Sortformer (v2 / v2.1).

This mirrors `SortformerEncLabelModel.forward_streaming` (NeMo main) but
incrementally, for audio that arrives in real time:

  offline reference                       this adapter
  -----------------                       ------------
  mel(all audio)                          mel over a sliding window per step
  streaming_feat_loader slices chunks     same slice geometry, computed from a
    of chunk_len*8 mel frames plus          growing sample buffer
    left/right context
  forward_streaming_step(...) per chunk   identical call, persistent state

Slice geometry (mel frames, hop 10 ms; encoder frames are 8 mel frames):
  step k covers mel [k*C*8, (k+1)*C*8) with left context lc*8 (clamped at
  stream start) and right context rc*8. left_offset / right_offset passed to
  forward_streaming_step are in mel frames, exactly as in
  `SortformerModules.streaming_feat_loader`.

Per-step mel is computed over just the needed audio window (this is how the
CoreML/ONNX ports of this model run too); boundary frames can differ from
whole-file mel by a hair, which is inaudible in the predictions.

Heavy imports (torch / NeMo) are deferred to construction so the rest of the
package works without them.
"""

import time
from typing import List

import numpy as np

from .engine_mock import SR, FRAME_S, BaseEngine, StepOut
from .presets import Preset

MEL_HOP = 160          # 10 ms at 16 kHz
MEL_WIN_TAIL = 240     # extra samples so the last mel frame has a full window
SUB = 8                # encoder subsampling: 8 mel frames per output frame


class SortformerEngine(BaseEngine):
    kind = "sortformer"

    def __init__(self, preset: Preset,
                 model_name: str = "nvidia/diar_streaming_sortformer_4spk-v2.1",
                 device: str = "auto"):
        super().__init__(preset)
        import torch  # deferred
        from nemo.collections.asr.models import SortformerEncLabelModel  # deferred

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = SortformerEncLabelModel.from_pretrained(
            model_name, map_location=self.device)
        self.model.eval().to(self.device)
        self.model.async_streaming = False

        m = self.model.sortformer_modules
        m.chunk_len = preset.chunk_len
        m.chunk_left_context = preset.chunk_left_context
        m.chunk_right_context = preset.chunk_right_context
        m.fifo_len = preset.fifo_len
        m.spkcache_len = preset.spkcache_len
        m.spkcache_update_period = preset.spkcache_update_period
        if hasattr(self.model, "_check_streaming_parameters"):
            self.model._check_streaming_parameters()
        self.n_spk = m.n_spk
        self.model_name = model_name

        self._reset_state()

    # ------------------------------------------------------------------ state
    def _reset_state(self):
        torch = self.torch
        m = self.model.sortformer_modules
        self.state = m.init_streaming_state(
            batch_size=1, async_streaming=False, device=self.device)
        self.total_preds = torch.zeros((1, 0, self.n_spk), device=self.device)
        self.emitted_frames = 0        # output (80 ms) frames already returned
        self.step_idx = 0              # streaming steps taken
        self.buf = np.zeros(0, dtype=np.float32)
        self.buf_start = 0             # absolute sample index of buf[0]

    # ------------------------------------------------------------- mel window
    def _mel_for_step(self, k: int):
        """Compute the mel slice for streaming step k from the sample buffer."""
        torch = self.torch
        m = self.model.sortformer_modules
        C = m.chunk_len * SUB                            # mel frames per hop
        stt_feat, end_feat = k * C, (k + 1) * C
        lo = min(m.chunk_left_context * SUB, stt_feat)   # left offset, mel frames
        ro = m.chunk_right_context * SUB                 # right offset, mel frames
        s0 = (stt_feat - lo) * MEL_HOP
        s1 = end_feat * MEL_HOP + ro * MEL_HOP + MEL_WIN_TAIL
        a, b = s0 - self.buf_start, s1 - self.buf_start
        audio = self.buf[a:b]
        need = lo + C + ro
        with torch.inference_mode():
            sig = torch.as_tensor(audio, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
            length = torch.tensor([sig.shape[1]], device=self.device)
            feats, _ = self.model.preprocessor(input_signal=sig, length=length)
        if feats.shape[2] < need:                        # only at final flush
            pad = feats.new_zeros((1, feats.shape[1], need - feats.shape[2]))
            feats = torch.cat([feats, pad], dim=2)
        feats = feats[:, :, :need]
        chunk_t = feats.transpose(1, 2)                  # (1, T_mel, 128)
        lengths = torch.tensor([need], dtype=torch.int64, device=self.device)
        return chunk_t, lengths, lo, ro

    def _step_ready(self, k: int) -> bool:
        m = self.model.sortformer_modules
        C = m.chunk_len * SUB
        s1 = (k + 1) * C * MEL_HOP + m.chunk_right_context * SUB * MEL_HOP + MEL_WIN_TAIL
        return self.buf_start + self.buf.size >= s1

    def _run_step(self, k: int) -> StepOut:
        torch = self.torch
        t_wall = time.perf_counter()
        chunk_t, lengths, lo, ro = self._mel_for_step(k)
        with torch.inference_mode():
            self.state, self.total_preds = self.model.forward_streaming_step(
                processed_signal=chunk_t,
                processed_signal_length=lengths,
                streaming_state=self.state,
                total_preds=self.total_preds,
                left_offset=lo,
                right_offset=ro,
            )
        new = self.total_preds[0, self.emitted_frames:].float().cpu().numpy()
        t0 = self.emitted_frames * FRAME_S
        self.emitted_frames = self.total_preds.shape[1]
        # drop consumed audio, keep what the next step's left context needs
        m = self.model.sortformer_modules
        keep_from = max(0, ((k + 1) * m.chunk_len * SUB
                            - m.chunk_left_context * SUB) * MEL_HOP)
        if keep_from > self.buf_start:
            self.buf = self.buf[keep_from - self.buf_start:]
            self.buf_start = keep_from
        return StepOut(t0, new, time.perf_counter() - t_wall)

    # -------------------------------------------------------------------- api
    def feed(self, pcm: np.ndarray) -> List[StepOut]:
        self.buf = np.concatenate([self.buf, pcm.astype(np.float32)])
        out: List[StepOut] = []
        while self._step_ready(self.step_idx):
            out.append(self._run_step(self.step_idx))
            self.step_idx += 1
        return out

    def finalize(self) -> List[StepOut]:
        """Flush the tail: pad with silence so the final partial chunk is emitted."""
        m = self.model.sortformer_modules
        C = m.chunk_len * SUB
        total = self.buf_start + self.buf.size
        if total <= self.step_idx * C * MEL_HOP:
            return []
        s1 = (self.step_idx + 1) * C * MEL_HOP + m.chunk_right_context * SUB * MEL_HOP + MEL_WIN_TAIL
        pad = s1 - total
        if pad > 0:
            self.buf = np.concatenate([self.buf, np.zeros(pad, dtype=np.float32)])
        out = [self._run_step(self.step_idx)]
        self.step_idx += 1
        return out

    def reset(self):
        self._reset_state()

    @property
    def info(self) -> dict:
        d = super().info
        d.update({"model": self.model_name, "device": str(self.device)})
        return d
