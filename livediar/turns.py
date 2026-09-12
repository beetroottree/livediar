"""Turn tracking: frame probabilities -> stable "who is speaking now".

Sortformer emits per-frame sigmoid probabilities for 4 speaker slots at
12.5 fps (80 ms frames). Raw thresholding flickers, so we apply, per slot:

  median smoothing (width 3)
  -> hysteresis (on above ON_TH, off below OFF_TH)
  -> minimum on/off durations
  -> optional speaker-count cap (see below)

which yields clean start/end turn events plus running talk-time stats.

Speaker cap
-----------
Sortformer's four slots are arrival-ordered and it will happily open a fourth
slot for a cough, a door, or a bad echo. When the user knows how many people
are in the room (`max_spk`), a slot that has never *established* itself (one
continuous turn of at least `establish_s` seconds) may not start a turn once
`max_spk` other slots are established. Established slots are never blocked,
so a real speaker who arrives late still gets in as long as a ghost has not
taken the seat first — ghosts rarely hold a 1.5 s continuous turn. Blocked
speech is counted in `suppressed_s` for diagnostics.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

FRAME_S = 0.08


@dataclass
class TurnEvent:
    spk: int
    event: str  # "start" | "end"
    t: float


@dataclass
class _SlotState:
    active: bool = False
    run: int = 0            # consecutive frames on the other side of the threshold
    cur_len: int = 0        # frames in the current turn
    longest: int = 0        # longest single turn so far, frames
    hist: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


class TurnTracker:
    def __init__(self, n_spk: int = 4, on_th: float = 0.55, off_th: float = 0.40,
                 min_on_frames: int = 2, min_off_frames: int = 4,
                 max_spk: Optional[int] = None, establish_s: float = 1.5):
        self.n_spk = n_spk
        self.on_th, self.off_th = on_th, off_th
        self.min_on, self.min_off = min_on_frames, min_off_frames
        self.max_spk = n_spk if not max_spk else max(1, min(int(max_spk), n_spk))
        self.establish_s = establish_s
        self.slots = [_SlotState() for _ in range(n_spk)]
        self.talk_s = [0.0] * n_spk
        self.turns = [0] * n_spk
        self.suppressed_s = [0.0] * n_spk
        self.smoothed_last = [0.0] * n_spk
        self.last_active_frames = np.zeros((0, n_spk), dtype=bool)  # per-frame activity of the last update

    # ---------------------------------------------------------------- config
    @property
    def params(self) -> dict:
        return {"on_th": self.on_th, "off_th": self.off_th, "min_on": self.min_on,
                "min_off": self.min_off, "max_spk": self.max_spk,
                "establish_s": self.establish_s}

    def configure(self, **kw) -> dict:
        """Update thresholds at runtime (from the UI). Unknown keys are ignored."""
        if "on_th" in kw: self.on_th = float(kw["on_th"])
        if "off_th" in kw: self.off_th = float(kw["off_th"])
        if "min_on" in kw: self.min_on = max(1, int(kw["min_on"]))
        if "min_off" in kw: self.min_off = max(1, int(kw["min_off"]))
        if "max_spk" in kw:
            v = kw["max_spk"]
            self.max_spk = self.n_spk if not v else max(1, min(int(v), self.n_spk))
        if "establish_s" in kw: self.establish_s = float(kw["establish_s"])
        return self.params

    # --------------------------------------------------------------- helpers
    def _established(self, s: int) -> bool:
        return self.slots[s].longest * FRAME_S >= self.establish_s

    def _may_start(self, s: int) -> bool:
        if self.max_spk >= self.n_spk or self._established(s):
            return True
        others = sum(1 for k in range(self.n_spk) if k != s and self._established(k))
        return others < self.max_spk

    def update(self, probs: np.ndarray, t0: float) -> Tuple[List[bool], List[TurnEvent], np.ndarray]:
        """Consume a (T, n_spk) block of frame probabilities starting at t0 seconds.

        Returns (active_now, events, smoothed_probs).
        """
        events: List[TurnEvent] = []
        smoothed = np.zeros_like(probs)
        active_frames = np.zeros(probs.shape, dtype=bool)
        for ti in range(probs.shape[0]):
            t = t0 + ti * FRAME_S
            for s in range(self.n_spk):
                st = self.slots[s]
                st.hist.pop(0)
                st.hist.append(float(probs[ti, s]))
                p = sorted(st.hist)[1]  # median of 3
                smoothed[ti, s] = p
                self.smoothed_last[s] = p
                if st.active:
                    self.talk_s[s] += FRAME_S
                    st.cur_len += 1
                    st.longest = max(st.longest, st.cur_len)
                    if p < self.off_th:
                        st.run += 1
                        if st.run >= self.min_off:
                            st.active, st.run, st.cur_len = False, 0, 0
                            events.append(TurnEvent(s, "end", t))
                    else:
                        st.run = 0
                else:
                    if p > self.on_th:
                        st.run += 1
                        if st.run >= self.min_on:
                            if self._may_start(s):
                                st.active, st.run = True, 0
                                st.cur_len = self.min_on
                                st.longest = max(st.longest, st.cur_len)
                                self.turns[s] += 1
                                events.append(TurnEvent(s, "start", t - (self.min_on - 1) * FRAME_S))
                            else:
                                self.suppressed_s[s] += FRAME_S
                                smoothed[ti, s] = 0.0   # hide it from the board too
                    else:
                        st.run = 0
                active_frames[ti, s] = st.active
        self.last_active_frames = active_frames
        return [st.active for st in self.slots], events, smoothed

    @property
    def active(self) -> List[bool]:
        return [st.active for st in self.slots]
