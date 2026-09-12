"""Overlap handling: voice profiles + targeted extraction of the quieter speaker.

The diarizer already tells us *where* two people talk at once (per-frame
activity) and *who dominates* (per-frame probability). That alone fixes most
caption confusion at the word level (see asr.Captioner). This module adds the
second step for the person who is active but not dominant:

  SpeakerBank : TitaNet-large embeddings (NeMo) learned from each slot's clean,
                non-overlapped turns - the "what does this voice sound like".
  Separator   : SepFormer (speechbrain, WHAMR 16 kHz) un-mixes an overlapped
                stretch into two streams; the bank picks the stream that
                matches the target speaker.

Measured on real speech (Obama/Trump mix): the dominant voice transcribes
better from the raw mix than from any separated stream, and separated streams
leak words from the other voice. So extraction is used *only* for the
non-dominant speaker's stretch, and words that duplicate the dominant
speaker's transcript in that stretch are dropped as leakage.
"""

import numpy as np

SR = 16000


class SpeakerBank:
    def __init__(self, device: str = "cpu", model: str = "nvidia/speakerverification_en_titanet_large"):
        import torch
        from nemo.collections.asr.models import EncDecSpeakerLabelModel
        self.torch = torch
        self.model = EncDecSpeakerLabelModel.from_pretrained(model, map_location=device).eval()
        self.device = device
        self.emb = {}       # slot -> unit vector (running mean), learned online this session
        self.secs = {}      # slot -> seconds enrolled
        self.people = {}    # person id -> {"name", "emb" (unit np array), "secs"}; enrolled by the user
        self.slot_person = {}   # slot -> person id (current best mapping)
        self.slot_votes = {}    # slot -> {person id: seconds of verified audio}

    def embed(self, audio: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        l = torch.tensor([x.shape[1]])
        with torch.inference_mode():
            _, e = self.model.forward(input_signal=x, input_signal_length=l)
        e = e[0].cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)

    def enroll(self, spk: int, audio: np.ndarray, max_s: float = 60.0):
        if self.secs.get(spk, 0.0) >= max_s:
            return
        e = self.embed(audio)
        w = audio.size / SR
        if spk in self.emb:
            self.emb[spk] = self.emb[spk] * self.secs[spk] + e * w
            self.secs[spk] += w
            self.emb[spk] /= np.linalg.norm(self.emb[spk]) + 1e-9
        else:
            self.emb[spk], self.secs[spk] = e, w

    def known(self, spk: int) -> bool:
        return self.profile(spk) is not None

    def profile(self, spk: int):
        """Best available voice vector for a slot: the enrolled person it maps to,
        else the profile learned online from its clean turns."""
        pid = self.slot_person.get(spk)
        if pid is not None and pid in self.people:
            return self.people[pid]["emb"]
        if spk in self.emb and self.secs[spk] >= 1.5:
            return self.emb[spk]
        return None

    # ---- enrolled people ---------------------------------------------------
    def add_person(self, pid: str, name: str, audio: np.ndarray) -> dict:
        e = self.embed(audio)
        self.people[pid] = {"name": name, "emb": e, "secs": audio.size / SR}
        return self.person_public(pid)

    def person_public(self, pid: str) -> dict:
        p = self.people[pid]
        return {"id": pid, "name": p["name"], "secs": round(p["secs"], 1)}

    def identify(self, audio: np.ndarray, min_sim: float = 0.40, min_margin: float = 0.08):
        """-> (person id or None, best sim, margin) for a clean single-speaker clip."""
        if not self.people:
            return None, 0.0, 0.0
        e = self.embed(audio)
        sims = sorted(((float(e @ p["emb"]), pid) for pid, p in self.people.items()), reverse=True)
        best, pid = sims[0]
        margin = best - (sims[1][0] if len(sims) > 1 else -1.0)
        if best >= min_sim and margin >= min_margin:
            return pid, best, margin
        return None, best, margin

    def vote(self, spk: int, pid: str, secs: float):
        """Accumulate verified seconds; returns the slot's mapping if it changed."""
        v = self.slot_votes.setdefault(spk, {})
        v[pid] = v.get(pid, 0.0) + secs
        lead = max(v, key=v.get)
        if self.slot_person.get(spk) != lead:
            self.slot_person[spk] = lead
            return lead
        return None

    def reset_session(self):
        self.emb, self.secs, self.slot_person, self.slot_votes = {}, {}, {}, {}

    def pick(self, spk: int, streams: np.ndarray, others=()) -> int:
        """Index of the stream (n, k) that best matches `spk`.

        Uses the target's profile when known; otherwise picks the stream least
        like any known 'other' speaker; otherwise -1 (no basis to choose).
        """
        embs = [self.embed(streams[:, k]) for k in range(streams.shape[1])]
        mine = self.profile(spk)
        if mine is not None:
            sims = [float(e @ mine) for e in embs]
            return int(np.argmax(sims))
        known_others = [self.profile(o) for o in others if self.known(o)]
        if known_others:
            sims = [max(float(e @ po) for po in known_others) for e in embs]
            return int(np.argmin(sims))
        return -1

    @property
    def info(self) -> dict:
        return {s: round(v, 1) for s, v in self.secs.items()}


class Separator:
    def __init__(self, device: str = "auto"):
        import torch
        from speechbrain.inference.separation import SepformerSeparation
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.torch, self.device = torch, device
        self.model = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-whamr16k",
            savedir=None, run_opts={"device": device})
        self._cache = {}

    def separate(self, audio: np.ndarray, key=None) -> np.ndarray:
        """(n,) mix -> (n, 2) estimated sources, RMS-normalised to the mix."""
        if key is not None and key in self._cache:
            return self._cache[key]
        torch = self.torch
        x = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            est = self.model.separate_batch(x)[0].cpu().numpy()
        est = est[:audio.size]
        mix_rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
        for k in range(est.shape[1]):
            est[:, k] *= mix_rms / (np.sqrt(np.mean(est[:, k] ** 2)) + 1e-9)
        if key is not None:
            if len(self._cache) > 32:
                self._cache.clear()
            self._cache[key] = est
        return est
