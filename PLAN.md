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
runs is labelled as such, everything else is cited. The three deep-research
reports behind the revisions of 2026-09-12 are in `docs/research/`
(`duplex.md`, `conditioning.md`, `data.md`, `infra.md`), each claim tagged
verified / inferred with its URL; this file states the conclusions.

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
  profile, projected). Two injection points, ablated at 1B (§5): the
  additive input embedding in `research/conditioning.py`, and DiCoW's
  actual mechanism, a per-layer diagonal affine blend over the STNO
  probabilities with *suppressive initialisation* (target/overlap = identity,
  silence/non-target scaled towards zero), which is what made FDDT trainable
  and is applied in all 32 encoder layers of DiCoW v3. DiCoW's query-key
  biasing is dropped: it hallucinates at init and wrecks timestamps
  (tcpWER 55.8 vs 47.8 on AMI; `docs/research/conditioning.md` §1). Masks are
  corrupted during training (§4.4) so the model is robust to diarizer errors,
  and *identity is bound outside the mask*: a slot swap inside a DiCoW-style
  mask is unrecoverable, so names attach to the identity embedding, never to
  the slot index. The conditioning output is **zero-gated at
  initialisation** (a learnable scalar starting at 0): injected at unit scale
  into a pretrained wav2vec2 it silenced the model outright (all-blank
  output, own run 2026-09-12), and the same would happen to Moshi. DiCoW's
  suppressive prior is therefore learned, not imposed.
- **Its own previous audio tokens** `m_{t-1}` and **inner-monologue text**
  `w_t`, exactly as in Moshi, with the acoustic delay so text leads audio.

Per frame it predicts: its own audio tokens (speech out), its inner monologue
(which we make *speaker-attributed*: the monologue carries `<spk:NAME>` tags
in serialized-output-training order, so the model's private transcript of
the room is who-said-what), and a **floor token** ∈ {listen, backchannel,
speak, yield}. The floor token is the interjection policy made explicit and
trainable. Every 2026 system that measures it wins with an explicit control
channel over implicit silence modelling — DuplexSLA's rate-limited action
channel (`response/interrupt/backchannel`), SALMONN-omni's `<shift>`,
MiniCPM-o's listen/speak bit, DuplexOmni's `[CUT]/[WAIT]` — so the floor
token is a DuplexSLA-style action channel, supervised from scripts (§4.3)
and then RL-tuned (§5 stage 4). Moshi's serialisation is kept exactly
(17 streams per step, acoustic delay 2 in pretraining → 1 in post-training,
text delay 0): changing the token layout would put us on DuplexSLA's
500k-hour continued-pretraining cost curve (`docs/research/duplex.md` §1, §3).

Backbone: Moshi's 7B Helium temporal transformer (32 layers, d 4096) with its
6-layer depth transformer for the codebooks, initialised from the public
Moshi weights — the only public model that already speaks full-duplex with
this exact layout (Qwen3-Omni's report does not contain the word "duplex").
Encoder-side path kept: Dixtral's DiCoW-conditioned Whisper encoder as a
second listener fused by cross-attention. It is the measured best
multi-speaker listener (AMI cpWER 19.8 with its own diarizer, 14.4 on
Mixer6, vs Gemini 3.0 Flash 44.4) and the routing study in
`docs/research/duplex.md` supports keeping it for overlap robustness.
SE-DiCoW's enrollment cross-attention (Libri3Mix full overlap 16.0 → 9.7) is
the evidence that identity embeddings help exactly where masks fail.

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

**4.1 Real multi-party audio, pseudo-labelled at scale.** The licence
inventory (`docs/research/data.md`) changes the arithmetic: commercially
clean, human-labelled multi-party audio is only ~700 h in total — AMI 100 h,
ICSI 72 h, CHiME-6 50 h (CC BY-SA since 2024), DiPCo 5 h, NOTSOFAR-1 ~300 h
across devices (CC BY 4.0, plus 1 000 h simulated), AliMeeting 119 h and
AISHELL-4 120 h (Mandarin, CC BY-SA) — plus ~4 600 h of dyadic Singlish /
code-switch with per-speaker mics in the IMDA NSC under the Singapore Open
Data Licence. Fisher, CALLHOME, Switchboard and Mixer 6 are LDC; Seamless
Interaction (4 000 h, per-participant channels), CANDOR and SPoRC are
CC BY-NC; Ego4D and MSDWild are research-only. None of those can feed a
commercial model. The route to scale is podcasts: DuplexChat (2026) released
a CC BY 4.0 pipeline and URL list covering 282 k h English / 133 k h
Japanese, the only open pipeline that outputs *separated* per-speaker
streams (pyannote community-1 + DialogueSidon + Parakeet) with published
turn statistics (10 % simultaneous speech, 3.1 backchannels/min). So 4.1 is
two tiers: the ~700 h human-labelled set as the clean anchor and for
per-scenario turn statistics, and 20 k+ h of DuplexChat-pipeline podcast
audio pseudo-labelled by our own pipeline — Sortformer + profiles + DiCoW on
the overlapped stretches + an LLM pass that repairs the serialized
transcript. Every pseudo-label carries its confidence; low-confidence frames
are masked out of the loss, not dropped from the audio.

**4.2 Simulated rooms with exact labels.** `research/simulate.py`: sample
2–6 speakers from single-speaker corpora — LibriTTS-R, Common Voice 26
(28.9 k validated h, CC0), Emilia-YODAS (114 k h, CC BY 4.0; base Emilia is
CC BY-NC and is *not* used), GigaSpeech, MLS, WenetSpeech and AISHELL-1/3 for
Mandarin, the IMDA NSC read parts for Singapore English — place them in a
room with an RIR (BUT ReverbDB's 279 real RIRs, openSLR 28; pyroomacoustics
rather than the AGPL gpuRIR), add MUSAN/DNS noise (WHAM! is NC), and drive
turn-taking from a Markov model fitted *per scenario*, because published
overlap rates disagree by 4×: AMI 21–22 %, ICSI 9–14 %, AliMeeting 35–42 %,
AISHELL-4 9–19 %, telephone ~10 %. NeMo's simulator exposes the same knobs
(`turn_prob`, `overlap_prob`, `mean_overlap`, `mean_silence`,
`dominance_var`) and is the reference implementation to match. Emit audio,
exact per-frame activity and the SOT transcript. Cheap, unlimited, and the
only place we get *exact* overlap labels. Target: 100 k hours, regenerated
per epoch.

**4.3 Assistant-in-the-room dialogues.** The behaviour data. An LLM writes
multi-party scripts in which an assistant is present: sometimes addressed
directly, sometimes not, sometimes it should interject with a correction or a
fact, sometimes it should stay silent through a long exchange, sometimes
it is interrupted. Scripts carry floor annotations per line (`listen`,
`backchannel`, `speak`, `yield`) *and an addressee label* — AMI has
addressee annotations, and the one multi-party benchmark cell that exists
(Full-Duplex-Bench v1.5 "talking to others") is exactly the case where
Moshi fails, answering 59 % of speech aimed at someone else. Multi-voice
TTS renders the humans with the §4.2 simulator for overlap and acoustics;
the 2026 open options with laughter/backchannel tags and commercial
licences are CosyVoice 3 (Apache, `[laughter]`, `[breath]`), Chatterbox
Turbo (MIT), Orpheus (Apache) and Dia2 (Apache, two-speaker scripts);
Voxtral TTS and Fish are NC. A fixed assistant voice renders the assistant
stream. This is Moshi's recipe (Fisher fine-tune, then 20 k+ h of
LLM-scripted dialogues rendered by a multi-stream TTS with one actor voice)
generalised from two parties to a room. Target: 10 k hours, plus a
human-recorded slice (500 h) for RL preference data: raters mark
interjections as welcome / late / rude. No public set carries that label;
the 2026 RL papers synthesise rewards from VAD timing on Fisher and
Seamless, both non-commercial, so this slice is not optional.

**4.4 Corruption.** Every training example's masks are perturbed: boundary
jitter ±240 ms, 5–10 % frames flipped, whole-slot swaps for a few seconds,
and a random 0.3–1.5 s lag, so the model tolerates the real diarizer's
errors and latency instead of trusting it. DiCoW only ever trained with
symmetric jitter and lost 7–14 tcpWER points going from oracle to real
diarization (AMI 17.6 → 25.0, NOTSOFAR 19.7 → 33.5), which its authors
attribute to missed speech it "has not encountered during training". Our
own AMI runs show Sortformer's errors are miss-dominated (DER 21–26 with
15–19 points of miss) with a ~1 s lag at overlap onset, so the corruption
model is asymmetric on purpose: *late onsets* and dropped short turns are
over-represented, conditioning is delayed rather than the audio, and slot
swaps are included because identity must survive them (§2).

## 5. Training curriculum

| stage | objective | data | goal metric | compute (H100-h) |
|---|---|---|---|---|
| 0 codec + init | reuse Mimi and the public Moshi weights; verify frame alignment | — | — | 0.5 k |
| 1 room listening | next-token on room audio + speaker-attributed monologue, conditioned | 4.1 + 4.2 | cpWER on AMI/NOTSOFAR ≤ Dixtral-with-Sortformer-masks | 3–23 k |
| 2 identity | identity embeddings + enrollment prompts, names in monologue | 4.2 with names, 4.1 with profiles | idWER − cpWER ≤ 1 point | 1–3 k |
| 3 duplex behaviour | own-audio stream + action channel, supervised | 4.3 | NOTSOFAR-1 turn-taking F1 ≥ 0.76, MOS ≥ 4.0 | 3–10 k |
| 4 RL for timing | DuplexPO-style windowed factorised reward on floor decisions | 4.3 human slice | onset MAE, VIR, rater win-rate | 1–3 k |
| 5 distil | 1–2B on-device student | all | ≤ +2 cpWER, RTF < 0.5 on M-series | 1–4 k |

The compute column was re-derived from data, not asserted: at the measured
planning rate of ~5 500 frames/s per H100 (torchtitan Llama-8B bf16 FSDP,
6 258 tok/s/GPU, with the depth transformer's overhead) one H100-hour
consumes ~440 audio-hours, so stage 1 over the §4 corpus is 2.7 k H100-h per
epoch-equivalent and 22.6 k at the upper end of the data mix; the first
draft's 40 k was ~150 epochs over the corpus and is withdrawn
(`docs/research/infra.md` §3). Stage 4 follows DuplexPO (arXiv 2607.07148)
rather than Kyutai's GRPO-plus-judge recipe: a windowed, factorised reward
(onset Gaussian, backchannel proximity, yield penalty, participation
regulariser) took Moshi's onset MAE from 1.99 s to 0.69 s without a 235B
judge in the loop, and it extends naturally with an addressed-to-agent mask
so the model learns *not* to answer speech aimed at someone else.

Stage 1 is the gate. If a diarization-conditioned LM does not match Dixtral
on cpWER with *our* diarizer's masks, the conditioning is not working and
nothing downstream matters. It is also the cheapest stage to iterate on
because the AMI harness scores it directly. No published number exists for
Dixtral on Sortformer masks, which is why `research/dixtral_ami.py` runs it
(§8); the streaming floor is NVIDIA's multitalker Parakeet at AMI-SDM cpWER
37.4 with a 1.12 s latency, the only public streaming system that consumes
Sortformer activity.

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
- **When to speak**: the standard full-duplex definitions rather than a
  bespoke floor-F1 — onset MAE, take-over rate and latency (Full-Duplex-Bench
  v1), yield rate and VIR (v3), the v1.5 Respond/Resume/Unknown cells with
  "talking to others" as the multi-party probe, HumDial's third-party
  scenarios — plus the one multi-party turn-taking number that exists:
  NOTSOFAR-1 turn-taking precision/recall/F1 as reported for Amazon's
  ModeratorLM (F1 0.76, false-positive rate 0.01) against Moshi (0.11 /
  0.66), and AMI next-speaker / addressee probes (humans 60.1 F1). Human
  preference win-rate vs. the stage-3 model on room scenarios.
- **Speech**: MOS, speaker-consistency of the assistant voice.

Go/no-go per stage is the goal-metric column in §5.

## 7. Compute and timeline

9 k–75 k H100-hours for the 7B programme end to end (the first draft's
110 k is withdrawn, see §5), dominated by stage 1; a 1–2B track runs
alongside at a fifth of that. On Modal at 2026 prices (H100 $3.95/h,
per-second billing) that is $36 k–$300 k. Modal's constraints shape the
schedule more than the money: GPU functions are preemptible (so every job
checkpoints to a Volume and is idempotent), a single function runs at most
24 h, single-job clustering is gang-scheduled full nodes up to 64 GPUs and
needs support approval, and plan tiers cap concurrency (Starter 10 GPUs,
Team 50). Stage 1 on 64 H100s is 2–15 days; the first pilot (job 6 in §8)
replaces the throughput estimate with a measurement before the big spend.
Data generation (4.2, 4.3) is CPU/TTS-bound and runs in parallel from day
one. Serving: Moshi reports ~200 ms end-to-end on an L4 and a 7B step on an
H100 is 15–25 ms with CUDA graphs, so the §3 budget holds only
graph-captured; `moshi_mlx` already runs the 7B at int4/int8 on an M3
laptop, so the 1–2B student on Apple silicon is a matter of distillation,
not feasibility.

## 8. The $500 programme (execution order, revised 2026-09-12)

Constraints as set: US$500 of Modal credit, unlimited CPU time, a 48 GB
M5 Pro whose memory is the local ceiling and which must not be crashed, and
Modal reserved for GPU work that genuinely saves time. That rules out §7's
64-GPU stage 1 for now; it does *not* rule out answering the plan's
decisive questions, because each of them is a small job. The credit buys
~125 H100-hours; the programme below spends about $200 of it and keeps
$300 in reserve for the one job that will need re-running.

**Rules that apply to every Modal job** (`research/modal_app.py`):
every function appends `{job, gpu, seconds, est_usd}` to `spend.jsonl` on
the volume and `modal run research/modal_app.py::spend` prints the running
total, so credit use is visible without the dashboard; every job is
idempotent (skips work whose result already exists on the volume) and
retried, because GPU functions are preemptible; every training job
checkpoints to the volume at a fixed cadence and resumes from the latest
checkpoint on restart, and is launched with `modal run --detach` so a
laptop sleep cannot kill it; and a workspace spend limit of $450 is set in
the Modal dashboard before the first job — the one thing only the account
owner can do.

**Local vs. Modal split.** Local (M5 Pro): everything that is CPU-bound or
fits comfortably in MPS memory — the AMI benchmark (Sortformer on CPU,
Whisper on the GPU), room simulation, the toy and the scaled conditioning
ablations (33 M parameters, 3 h at 20 k steps), data preparation, the
DuplexChat podcast pipeline, and all scoring. Rule of thumb from the crash
log: keep at most three CPU workers and one MPS job at a time, under
`caffeinate`. Modal: only what the Mac cannot do in reasonable time —
anything with a ≥3B model over hours of audio, and anything that trains a
7B model.

| # | job | where | cost | what it decides |
|---|---|---|---|---|
| 1 | AMI benchmark, all 48 scores + report | local | $0 | the Whisper-pipeline baseline (cpWER, idWER) |
| 2 | toy conditioning ablation, 320 rooms / 2.5 k steps | local | $0 | does additive conditioning work at all |
| 3 | Dixtral on Sortformer masks, 16 meetings | Modal, 16 × H100, ~15 min | ~$12 | the stage-1 target number; is our diarizer good enough to condition on |
| 4 | scaled ablation, 2 000 rooms / 20 k steps, additive vs. FDDT-style | local, MPS, ~6 h | $0 | which injection to use |
| 5 | per-scenario turn statistics from AMI/ICSI; regenerate rooms | local | $0 | simulator fidelity |
| 6 | **Moshi-LoRA stage-1 pilot**: moshi-finetune (LoRA, one H100) with the conditioning module added and the text stream re-targeted to the speaker-attributed room transcript, on 300 h simulated rooms + the 700 h labelled meetings; checkpoints every 500 steps | Modal, 1 × H100, ~30 h | ~$120 | whether a full-duplex-shaped model learns to *listen to a room* under conditioning — the plan's central bet, scored by job 1's harness |
| 7 | pilot re-run / second arm (FDDT variant or Dixtral encoder path) | Modal, 1 × H100 | ~$120 | reserve |
| 8 | assistant-in-the-room scripts (1 000) rendered with CosyVoice 3 / Dia2, 100 h | local | $0 | the §4.3 behaviour spec |

Job 2 went through two invalid versions before a valid one (`docs/RUNS.md`):
a CTC recognizer trained from scratch on 7 h of simulated rooms never learns
the acoustics, so it cannot measure conditioning. The valid ablation freezes
a pretrained listener (wav2vec2-base-960h) and trains only the speaker-tag
outputs and the conditioning — the experiment then measures exactly the two
things conditioning is for, *who* and *where turns start*, and its lesson
(zero-gate the conditioning) is already in §2.

Job 6 replaces §5's 7B stage 1 as the thing this budget proves. It is
deliberately a LoRA on the public Moshi weights rather than full training:
moshi-finetune runs LoRA on one H100 (39.6 GB), the conditioning module is
~5 M parameters, and 30 H100-hours at 440 audio-hours per H100-hour is
~13 k audio-hours of exposure — 13 epochs over the 1 000 h pilot corpus,
enough to see the cpWER curve bend or not. If it bends, §7's stage 1 is
the same recipe with the LoRA removed and the corpus scaled, and the
budget case for it is made with a measured curve rather than an estimate.
If it does not, the Dixtral encoder path (job 7) is the fallback and the
plan changes before real money is spent.

What gets documented, per job, in `docs/RUNS.md`: the exact command, the
Modal app id, GPU-seconds and dollars from `spend.jsonl`, the checkpoint
path, the metric before/after, and one paragraph on what changed in the
plan because of it. Nothing is reported as done without the metric.
