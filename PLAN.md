# livediar → a room-aware spoken model

Goal, in one line: an always-listening model that hears a room of people,
knows who is who, keeps the conversation in context, decides on its own when
to speak, and answers in speech. Raw audio in, speech out, with the
diarizer's "who is talking" stream as a first-class input rather than a
transcript in the middle.

This document is the plan for building that model, written the way a lab
would write it: what the SOTA gives us, exactly where it stops, the
architecture we train, the data programme, the training curriculum, the
evaluation that decides go/no-go at each stage, and the compute. The code
under `research/` is the executable scaffold for it: the conditioning module,
the multi-stream model skeleton, the data simulator, the stage configs. It is
a scaffold, not a trained model; every number below that comes from our own
runs is labelled as such, everything else is cited.

## 1. Where the SOTA stops

Three families exist and none does the whole job.

**Full-duplex dialogue models** (Moshi, `moshika-rl-seamless`, Step-Audio 2.5,
DuplexSLA). Moshi models two parallel audio streams — the user's and its own
— as Mimi codec tokens at 12.5 Hz, plus an "inner monologue" text stream that
it predicts ahead of its own audio, which is what makes its speech coherent.
Because both streams are always modelled, the model can talk while the user
talks: interruption, backchannels and barge-in are just token patterns, and
the 2026 `moshika-rl-seamless` post-trains those four behaviours with RL on
Seamless Interaction, a 4 000 h two-party corpus with one channel per
speaker. This is the right *shape* for "jump in whenever". Its limit is hard
and structural: exactly one user stream, no notion of a third person, no
identity, and the training data is dyadic.

**Diarization-conditioned transcribers** (DiCoW, Dixtral). DiCoW feeds
Whisper's encoder a per-frame STNO mask — silence / target only / non-target
only / target overlapped — and learns to transcribe the target speaker out of
the mix. Dixtral puts that encoder in front of the Voxtral 3B spoken LLM with
the decoder frozen, and beats Gemini 3.0 Flash by 29 points absolute cpWER on
speaker-attributed transcription while also answering questions about
multi-speaker audio (Interspeech 2026; weights `BUT-FIT/Dixtral`, Apache 2.0).
This is the right *conditioning mechanism* for "aware of many people". Its
limits: offline (whole recording in, text out), one target speaker per pass,
no speech output, no policy for when to speak.

**Omni models** (Qwen3-Omni, Kimi-Audio, Gemma 3n) and the live assistants
(OpenAI Realtime, Gemini Live). Native audio in, sometimes speech out,
streaming, but no diarization input at all; multi-speaker handling in the
commercial stacks is a separate transcription model bolted on the side.
They are useful as initialisations and as baselines, not as the answer.

What we already have on our side: a streaming diarizer (Sortformer v2.1,
80 ms frames, arrival-ordered slots) with a speaker cap, TitaNet voice
profiles that turn slots into named people, and an AMI harness with DER,
cpWER and identity-WER — i.e. the *input side* of the model and the
*evaluation side* exist and are measured (§6).

## 2. The model: a K-stream full-duplex LM with diarization conditioning

Frame rate is the key coincidence: Sortformer emits a decision every 80 ms
and Mimi tokenises audio at 12.5 Hz, also 80 ms. One transformer step per
frame carries everything.

Per frame *t* the model consumes:

- **Room audio tokens** `a_t` — Mimi tokens of the *mixture* (8 codebooks).
  We do not separate; blind separation was measured to hurt (README
  "Overlapping speech"). The model learns to listen selectively, the way
  DiCoW does, because we tell it who is active.
- **Diarization conditioning** `d_t` — for each of K slots (K = 4, later 8)
  the STNO-style state {silent, active-alone, active-overlapped} and the
  slot's *identity embedding* (TitaNet vector of the enrolled or online
  profile, projected). These are summed into the frame embedding (FDDT-style
  additive conditioning; `research/conditioning.py`). Masks are corrupted
  during training (§4) so the model is robust to diarizer errors.
- **Its own previous audio tokens** `m_{t-1}` and **inner-monologue text**
  `w_t`, exactly as in Moshi, with the acoustic delay so text leads audio.

Per frame it predicts: its own audio tokens (speech out), its inner monologue
(which we make *speaker-attributed*: the monologue carries `<spk:NAME>` tags
in serialized-output-training order, so the model's private transcript of
the room is who-said-what), and a **floor token** ∈ {listen, backchannel,
speak, yield}. The floor token is the interjection policy made explicit and
trainable; Moshi learns it implicitly from silence tokens, we supervise it
(§4.3) and then RL it (§5 stage 4).

Backbone: a 7B-class temporal transformer (init from Moshi's Helium or from
Qwen3-Omni's thinker, decided by stage-1 ablation) with Moshi's depth
transformer for the codebooks. Encoder-side alternative kept in the plan:
Dixtral's DiCoW-conditioned Whisper encoder as a second listening path,
fused by cross-attention — it is the proven multi-speaker listener and
costs nothing to try since the weights are public.

Identity: enrollment is a prompt, not a weight. At session start the K
identity embeddings come from `profiles.json`; unknown slots get a learned
"unknown" vector and the model refers to them as "speaker 3" until named.
Names bind to slots via the identity embedding, so a diarizer slot swap
mid-meeting does not rename anyone.

## 3. Serving: continuous, no transcript in the loop

The livediar server already produces per frame: mixture audio, Sortformer
probabilities, tracker activity, slot→person mapping. The serving path is
that stream, verbatim, into the model:

```
mic ─► Mimi encode ─┐
Sortformer ─► tracker ─► STNO + identity ─┼─► K-stream LM ─► floor / monologue / Mimi decode ─► speaker
profiles.json ──────────────────────────┘
```

Latency budget per frame at 80 ms: Sortformer step 50 ms on CPU today
(RTF 0.64), Mimi encode/decode < 10 ms, LM step must be < 40 ms → 7B in
bf16 on one H100 is fine, on-device needs the 1–2B variant (§7). The decision
buffer of the `low` preset (1.04 s) is the floor on how far behind the room
the model's *knowledge of who is speaking* runs; the audio itself arrives
immediately, so a fast reaction can start on audio alone and the attribution
catches up — this is why the conditioning must be robust to a short lag
(train with lagged masks, §4).

## 4. Data programme

The model needs three things no public corpus gives together: many
speakers, overlap with ground-truth per-speaker text, and an *assistant
participant* whose interjections are the target behaviour. So the data is
built in layers.

**4.1 Real multi-party audio, pseudo-labelled at scale.** Meetings and
multi-host conversations: AMI, ICSI, CHiME-6/7/8, NOTSOFAR-1, DiPCo,
AliMeeting (Mandarin), plus podcasts / panel recordings gathered under
licence. Labels come from the pipeline we already run — Sortformer +
profiles + Whisper — with two upgrades before scaling: DiCoW for the
overlapped stretches (it is the measured better transcriber there) and an
LLM pass that repairs the serialized transcript. Every pseudo-label carries
its confidence; low-confidence frames are masked out of the loss, not
dropped from the audio. Target: 20 k hours.

**4.2 Simulated rooms with exact labels.** `research/simulate.py`: sample
2–6 speakers from single-speaker corpora (LibriTTS-R, Common Voice, Emilia,
the IMDA NSC for Singapore English/Mandarin code-switching), place them in a
room with an RIR, generate a turn-taking schedule from a Markov model fitted
to AMI turn statistics (turn length, gap, overlap rate ~15%, backchannel
rate), mix, and emit the audio, exact per-frame activity, and the SOT
transcript. Cheap, unlimited, and the only place we can get *exact* overlap
labels. Target: 100 k hours, regenerated per epoch.

**4.3 Assistant-in-the-room dialogues.** The behaviour data. An LLM writes
multi-party scripts in which an assistant is present: sometimes addressed
directly, sometimes not, sometimes it should interject with a correction or a
fact, sometimes it should stay silent through a long exchange, sometimes
it is interrupted. Scripts carry floor annotations per line (`listen`,
`backchannel`, `speak`, `yield`). Multi-voice TTS renders the humans (with
the §4.2 room simulator for overlap and acoustics) and a fixed assistant
voice renders the assistant stream. This is the Moshi synthetic-dialogue
recipe generalised from two parties to a room. Target: 10 k hours, plus a
human-recorded slice (500 h) for RL preference data: raters mark
interjections as welcome / late / rude.

**4.4 Corruption.** Every training example's masks are perturbed: boundary
jitter ±240 ms, 5–10% frames flipped, whole-slot swaps for a few seconds,
and a random 0.3–1.5 s lag, so the model tolerates the real diarizer's
errors and latency instead of trusting it.

## 5. Training curriculum

| stage | objective | data | goal metric | compute (H100-h) |
|---|---|---|---|---|
| 0 codec + init | reuse Mimi; pick backbone init by ablation | — | — | 2 k |
| 1 room listening | next-token on room audio + speaker-attributed monologue, conditioning on | 4.1 + 4.2 | cpWER on AMI/NOTSOFAR ≤ Dixtral | 40 k |
| 2 identity | add identity embeddings + enrollment prompts, names in monologue | 4.2 with names, 4.1 with profiles | idWER ≈ cpWER (identity free) | 10 k |
| 3 duplex behaviour | add own-audio stream + floor token, supervised | 4.3 | floor F1, MOS of speech | 30 k |
| 4 RL for timing | preference RL on interjection timing (Moshika-RL recipe, room version) | 4.3 human slice | rater preference win-rate | 10 k |
| 5 distil | 1–2B on-device student | all | ≤ +2 cpWER, RTF < 0.5 on M-series | 15 k |

Stage 1 is the gate. If a diarization-conditioned LM does not match Dixtral
on cpWER with *our* diarizer's masks, the conditioning is not working and
nothing downstream matters. It is also the cheapest stage to iterate on
because the AMI harness scores it directly.

Ablations that decide architecture, run at 1B scale before stage 1 at 7B:
additive STNO vs. query-key biasing; identity as embedding vs. as text
prompt; mixture tokens only vs. mixture + DiCoW encoder path; corruption
strength.

## 6. Evaluation

Every stage has a number that already exists in `bench/`:

- **Who**: DER (pyannote refs) — property of the diarizer, tracked so
  regressions are attributed correctly.
- **Who said what**: cpWER, and idWER with enrolled profiles, on AMI (16 test
  meetings, 9 h) and NOTSOFAR-1. Our current Whisper pipeline is the
  baseline; the first AMI meetings scored so far sit at cpWER 22–41% with
  the overlap attribution consistently better than the raw mix by 0.6–5
  points, and on the one meeting where the diarizer badly mixed speakers
  (IS1009c, DER 35%) enrolled identity cut speaker-attributed WER from 61%
  to 26% — the strongest evidence yet that identity must be a model input,
  not a post-hoc slot label (own run, 2026-09-12, full table in
  `bench/ami/results/RESULTS.md` when complete).
- **Understanding**: Dixtral's multi-speaker QA set, plus our own
  "who agreed to what" questions generated from AMI's abstractive summaries.
- **When to speak**: floor-token F1 against script labels; latency from cue
  to onset; human preference win-rate vs. Moshika-RL on room scenarios.
- **Speech**: MOS, speaker-consistency of the assistant voice.

Go/no-go per stage is the goal-metric column in §5.

## 7. Compute and timeline

~110 k H100-hours for a 7B programme end to end, dominated by stage 1; a
1–2B track runs alongside at a fifth of that. With 256 H100s that is a
three-month programme with stage 1 done in the first month. Data generation
(4.2, 4.3) is CPU/TTS-bound and runs in parallel from day one.

## 8. What to do first (this week)

1. Reproduce Dixtral on the AMI harness with Sortformer masks instead of
   its own diarizer. This tells us whether our masks are good enough for
   conditioning (the whole plan assumes yes) and gives the stage-1 target
   number. `research/README.md` step 1.
2. Stand up the simulator (`research/simulate.py`) and validate it by
   training a 100M-parameter model on simulated rooms only and checking
   that cpWER on held-out simulated rooms beats the unconditioned model —
   the smallest possible proof that additive diarization conditioning
   works in our stack (`research/conditioning.py`, `research/duplex_lm.py`).
3. Generate the first 1 000 assistant-in-the-room scripts and listen to 20
   of them; the behaviour spec in §4.3 will change after that.
