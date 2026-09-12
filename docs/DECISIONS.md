# Decisions

Every non-obvious choice in this project, with the evidence that drove it and
what would reverse it. Newest at the bottom of each section. Numbers marked
*(own run)* come from this repo's benchmarks (`docs/RUNS.md`,
`bench/ami/results/`); everything else is cited in `docs/research/`.

## Live app (livediar)

**Sortformer v2.1 as the diarizer, streamed through NeMo's own step function.**
It is the newest directly loadable streaming diarizer (NVIDIA's
Nemotron-3-Diarization preview of 2026-09-10 is gated and evaluation-only) and
its 80 ms frame rate matches the Mimi codec the model programme needs. Reverse
if a non-gated 8-speaker streaming diarizer appears — the 4-slot cap is the
architecture's hard limit.

**A speaker cap instead of threshold tuning for the "ghost 4th speaker".**
A threshold sweep on cached probabilities showed thresholds trade real talk
time for almost no ghost reduction (Obama 86.9 → 67.9 s of talk to remove one
0.9 s ghost turn), while an establish-rule cap (one continuous turn ≥ 1.5 s)
removed it with every real speaker's numbers unchanged *(own run)*. Known
cost: a ghost that holds 1.5 s before a real third person speaks takes the
seat; the UI cap is user-settable for that reason.

**Whisper large-v3-turbo on MLX for captions, driven by diarizer turns.**
Every closed turn is single-speaker by construction, so slicing that audio and
transcribing it makes attribution exact with no word matching; the GPU takes
the ASR (40× realtime) and leaves the CPU to Sortformer (0.64× realtime).
Merge same-speaker turns within 1.2 s and restrict language detection to an
allowed set, because Whisper chops without the merge and guessed Dutch for a
1.5 s English fragment without the restriction *(own run)*.

**Blind separation is off by default.** SepFormer (WHAMR 16 kHz) on a real
Obama/Trump mixture left the louder voice *worse* than the raw mix and
produced junk words for the quieter one even after a per-word confidence
floor ("months other. had") *(own run)*. Word-level attribution by diarizer
dominance is the default; the separation path stays as `--overlap extract`
for a better separator. Reverse when a separator trained on meeting-like
data (not WSJ) shows clean transcripts on `samples/synth_overlap.wav`.

**Identity from voice profiles, never from slot index.** TitaNet identified
2 s snippets 13/13 for Obama vs Trump at an 11 Hz pitch gap *(own run)*, and
on AMI IS1009c — where the diarizer mixed speakers badly — per-caption
verification cut speaker-attributed WER from 61 % to 26 % *(own run)*. Slot
swaps are unrecoverable inside a mask, so names bind to embeddings. Reverse:
never; this is now a plan-level principle (PLAN §2).

**The AMI benchmark replays cached diarizer probabilities.** One Sortformer
pass per meeting (~2 h for 9 h of audio on this Mac) and every caption
condition scored from the cache through the *same* tracker/captioner code the
server runs. Reverse if the tracker changes its input contract.

## Model programme (PLAN.md)

**Full-duplex, Moshi layout, not a transcript-then-LLM pipeline.** The goal
is "hears a room, decides when to speak, answers in speech". Moshi's 17-stream
serialisation with acoustic delay 2→1 is kept verbatim because changing the
token layout puts the programme on DuplexSLA's 500 k-hour continued-pretraining
cost curve (`docs/research/duplex.md`).

**Diarization conditioning enters as an additive per-frame signal, with a
DiCoW-style per-layer variant ablated against it; QKb dropped.** DiCoW's
query-key biasing hallucinates at init and hurts timestamps (tcpWER 55.8 vs
47.8 on AMI); its per-layer diagonal affine with suppressive init is what
works (`docs/research/conditioning.md`).

**Conditioning is zero-gated and bounded.** Discovered the hard way in the
ablation: a unit-scale conditioning vector added to a pretrained wav2vec2's
encoder input silenced it (all-blank output), and even a *trained* gate
drifted to −0.20 (conditioning RMS 0.25) and collapsed the frozen encoder,
while RMS 0.06 left it intact *(own runs, 2026-09-12)*. The gate is therefore
zero-initialised and bounded (`max_gate` × tanh, default 0.1). Input-level
additive injection into a *frozen* pretrained encoder is brittle; the pilot's
host is not frozen (LoRA adapts around the conditioning), and DiCoW's
per-layer FDDT — the variant that halved cpWER in job 3 — is the fallback
injection if the pilot's additive path underperforms.

**The conditioning question is answered by job 3, not the toy.** Dixtral with
our Sortformer masks: cpWER 20.3 % over 14 AMI meetings vs 41.5 % for the
unconditioned Whisper pipeline on the same masks *(own run, 2026-09-12)*.
The toy ablation's remaining value is engineering (gate bounds, injection
point), not evidence; its Modal copy was cancelled.

**Explicit floor-control channel over implicit silence modelling.** Every
2026 system that measures it wins this way (DuplexSLA, SALMONN-omni,
MiniCPM-o, DuplexOmni). The RL recipe is DuplexPO's windowed factorised
reward (onset MAE 1.99 → 0.69 s on Moshi) rather than a 235B judge in the
loop.

**Compute sized from data, not asserted.** The first draft's 40 k H100-h for
stage 1 was ~150 epochs over the corpus and was withdrawn; at ~440 audio-hours
per H100-hour the range is 2.7 k–22.6 k (`docs/research/infra.md`).

**Commercial-clean data only, and podcasts for scale.** ~700 h of human-
labelled multi-party audio is all that is commercially usable; Fisher,
Seamless Interaction, base Emilia, WHAM! and DiariZen are non-commercial.
DuplexChat's CC-BY podcast pipeline (415 k h) is the route to 20 k+ h
(`docs/research/data.md`).

**Simulator parameters are fitted, not guessed.** AMI train+dev: 13.8 % of
speech time overlapped, 39 % of handovers overlap, median turn 1.53 s, 36 % of
turns under 0.8 s, same-speaker continuation 0.22 (`research/fit_turns.py`).
Published overlap rates span 9–42 % across corpora, so fit per scenario.

## Execution ($500 programme)

**Local by default; Modal only for ≥3B-model-over-hours-of-audio work.**
Measured: Dixtral runs at 3.6–9× realtime on the M5 Pro's GPU (one 14-min
meeting ≈ 1–2 h) versus ~15 min for all 16 meetings on 16 H100s (~$12).
Everything else — diarization, simulation, the ablations, scoring — runs
here for $0.

**Stage-1 pilot is a LoRA on public Moshi weights, trained in segments.**
moshi-finetune runs LoRA on one H100; segments restart from a merged
checkpoint on the volume, so a preempted or timed-out container loses at most
one segment and no in-place FSDP resume is needed. ~$120 of the $500.

**Every Modal job records its spend and is idempotent.** `spend.jsonl` on the
volume, `modal run research/modal_app.py::spend` to sum it; jobs skip work
whose result exists. Because Modal GPU functions are preemptible and capped
at 24 h.

**Conditioning ablation on a frozen pretrained listener.** Two from-scratch
CTC recognizers (word-level at 12.5 Hz, char-level at 50 Hz on Whisper-tiny
features) never learned the acoustics on 7 h of simulated rooms and so could
not measure conditioning *(own runs, dev WER > 100 %)*. Freezing
wav2vec2-base-960h and training only four `<spk k>` outputs plus the
conditioning makes the experiment measure exactly what conditioning is for.

**Long local jobs run under `caffeinate`; GPU jobs are relaunched after any
sleep.** The Mac idle-slept 05:23–11:42 on 2026-09-12 and the Metal compiler
service died, killing every MLX/MPS process; a shell that survived the
hibernate could not spawn Metal-capable children either.
