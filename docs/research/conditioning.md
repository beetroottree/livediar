# Diarization-conditioned ASR and speech-LLMs

Research note for PLAN.md §2 (the conditioning input `d_t`) and §5 (the
stage-1 gate). Question answered: how do the systems that already condition
a transcriber or an LLM on "who is talking" actually do it, what do they
score, and what breaks when the conditioning comes from *our* diarizer
(Streaming Sortformer v2.1, 80 ms frames, 4 arrival-ordered slots) instead
of the offline DiariZen the BUT group uses.

Sources are marked **[verified]** when read directly (paper text, model
card, or the local clone at `research/dixtral_repo/`) and **[inferred]**
when they are my reading between the lines. Numbers without a mark are
verified from the cited table. Written 2026-09-12; the arXiv ids after
2606 are recent enough that camera-ready versions may still move.

## 1. DiCoW — Diarization-Conditioned Whisper

Paper: Polok, Klement, Kocour, Han, Landini, Yusuf, Wiesner, Khudanpur,
Černocký, Burget, "DiCoW: Diarization-Conditioned Whisper for Target
Speaker Automatic Speech Recognition", arXiv 2501.00114 (v1 30 Dec 2024),
Computer Speech & Language vol. 95, 101841, 2026
(https://arxiv.org/abs/2501.00114,
https://www.sciencedirect.com/science/article/pii/S088523082500066X).
Code: training in https://github.com/BUTSpeechFIT/TS-ASR-Whisper, inference
in https://github.com/BUTSpeechFIT/DiCoW, both Apache-2.0 **[verified]**.

### 1.1 STNO masks

The diarizer output is `D ∈ [0,1]^{S×T}`, `d(s,t)` the probability that
speaker `s` is active in frame `t`. For a chosen target `s_k` the four
mutually exclusive frame classes are (paper eqs. 4–7) **[verified]**:

    p_S(t) = Π_s (1 − d(s,t))                       silence
    p_T(t) = d(s_k,t) · Π_{s≠s_k} (1 − d(s,t))      target alone
    p_N(t) = (1 − p_S(t)) − d(s_k,t)                non-target only
    p_O(t) = d(s_k,t) − p_T(t)                      target overlapped

so the mask `M_t = [p_S p_T p_N p_O]ᵀ` has a fixed size regardless of `S`,
which is the whole trick: the model never sees a speaker index, only "is
the one I was asked about active, and is anyone else". Decoding a
recording with `S` speakers means `S` independent forward passes with `S`
different masks. The code that builds it is `_create_stno_masks` in
`research/dixtral_repo/src/data/local_datasets.py` and, identically,
`_stno_from_diar` in `demo/pipeline.py`; note `sil = Π(1−d)` and
`anyone_else = Π_{s≠k}(1−d)` are literal products, so the formulas are
applied to soft values without modification **[verified]**.

Both training and inference use *hard* masks. §6.3 of the paper is
explicit: the CHiME-8 submission had claimed gains from soft masks, but
"subsequent analysis revealed that these improvements were largely due to
the overly aggressive post-processing used to derive hard decisions", so
"we always use hard diarization decisions ... during decoding, as this
approach matches the setup used during training. Incorporating soft
activations into the current framework remains an open problem"
**[verified]**. With one-hot masks eq. 13 below collapses to selecting one
of four affine maps per frame.

### 1.2 The three conditioning mechanisms compared in the paper

**Input masking** (§4.2): `x_masked(t) = x(t)·(p_T + p_O)` — zero the
waveform where the target is silent. Works (Whisper goes from 220 → 52.8
tcpWER on AMI-sdm, Table 4) but introduces artificial silences.

**Query-key biasing, QKb** (§4.3, eqs. 9–12): extend every query with a
constant 1 and every key with `−c`, extend `W_q`, `W_k` with a 1 on the new
diagonal. Then `a_ij = (W_q q_i)ᵀ(W_k k_j) − c` for non-target key frames
and unchanged for target frames. `c` is 0 on target frames and a constant
(initialised to 50) elsewhere; it is applied to encoder self-attention
*and* decoder cross-attention; because the extended projections are
trainable the model can learn to un-suppress non-target frames
**[verified]**. Two problems the paper documents: at initialisation the
model hallucinates (tcpWER 276 / 260 / 991 on AMI / NOTSOFAR / Libri2Mix,
Table 4), and masking attention leaves "holes" in the positional sequence
the decoder cross-attends to. Their fix, *shifted positional embeddings*
(§4.3.1: for a T/N pattern `TTTNNTT` the positions become `1233345`),
improves cpWER but wrecks timestamps: Table 5, AMI-sdm cp/tcp 21.3/55.8
with shift vs 45.9/47.8 without; NOTSOFAR 25.2/65.4 vs 27.3/28.5
**[verified]**. QKb is a dead end for anything that needs timing.

**Frame-level Diarization-Dependent Transformations, FDDT** (§4.4, eq. 13):
per encoder layer `l` and per STNO class `i`, an affine map
`(W_i^l, b_i^l)`; the layer input is replaced by the convex blend

    ẑ_t^l = Σ_{i∈{S,T,N,O}} (W_i^l z_t^l + b_i^l) · p_i(t)

applied *before* every encoder layer (all 32 of Whisper large-v3 /
large-v3-turbo; config `apply_fddt_to_n_layers: -1` = all)
**[verified]**. In the paper `W_i^l ∈ R^{d_m×d_m}` is a full matrix; the
released code defaults to `fddt_is_diagonal: true` (a `d_m` vector times
the hidden state plus bias, `CustomDiagonalLinear`), with `bias_only` as a
further option **[verified, FDDT.py/config.py]**. Initialisation is what
makes it trainable on top of a pretrained Whisper ("suppressive"): in the
paper `W_S^0 = W_N^0 = 0` at the first layer, `W_T^l = W_O^l = I`, all
biases 0, so silence and non-target frames are killed at the input and
the model behaves like input masking from step 0 (Table 4: FDDT at init
78.3 / 89.7 / 102 vs input masking 52.8 / 61.6 / 56.2) **[verified]**.
The v3 generation relaxed this: an extra FDDT *before* the positional
embeddings (`use_pre_pos_fddt`, "initial_fddt") whose S/N diagonal is
initialised to `non_target_fddt_value` = 0.5 (SE-DiCoW paper: "increased
from 0.1 to 0.5"), while the per-layer FDDTs are built with
`non_target_rate=1.0`, i.e. identity for all four classes
**[verified, encoder.py; the HF v3_2 card calls this "less strict
suppressive initialization"]**. Training recipe in the paper: three
stages — CTC-preheat on LibriSpeech-960, FDDT-preheat (only FDDT + CTC
params on the multi-speaker data), then full fine-tune; AdamW, batch 64,
peak LR 2e-7 for pretrained weights and 100× that for FDDT (the Dixtral
config uses `fddt_lr_multiplier: 10.0`), 5k warmup, ≤50k steps, CTC weight
λ = 0.3 **[verified]**.

Ablation, tcpWER with oracle diarization (Table 4) **[verified]**:

| method | AMI-sdm | NOTSOFAR-1 | Libri2Mix |
|---|---|---|---|
| Whisper large-v3-turbo, no conditioning | 220.0 | 260.1 | 588.2 |
| input masking | 52.8 | 61.6 | 56.2 |
| QKb, tuned, no shift | 47.8 | 28.2 | 7.9 |
| FDDT single-domain | 17.8 | 20.9 | 6.3 |
| FDDT multi-domain (AMI:NSF:L2M = 4:4:1) | 17.6 | 19.7 | 6.9 |
| + Co-Attention across target instances | 18.1 | 20.0 | 5.8 |

Co-Attention is a module that lets the `S` parallel target instances
compare notes; it only helps on fully-overlapped synthetic data. The
paper's own reading is that "with fully overlapped speech ... multiple
TS-ASR instances can decide to decode speech from the same speaker"
**[verified]** — the STNO masks of two speakers who talk over each other
the whole time are nearly identical (both mostly O). SE-DiCoW (§1.6) is the
real fix.

Extras in the encoder: a CTC head (one Transformer layer, two stride-2
convs 1500 → 375 frames, linear to vocab) trained jointly and used for
joint CTC/attention decoding — on NOTSOFAR eval, beam 5: 22.2 without the
head, 20.9 with λ = 0.2 (Table 8) **[verified]**. Dixtral keeps the
`additional_self_attention_layer` and `ctc_lm_head` modules but trains
with `ctc_weight: 0.0` **[verified, configs/base.yaml]**.

### 1.3 Mask construction at inference: 50 Hz, from segments

Whisper's mel front-end has a 10 ms hop and the encoder's `conv2` has
stride 2, so encoder frames are 20 ms = 50 Hz, 1500 per 30 s window. The
mask is built directly at that rate. Training path
(`local_datasets.get_stno_mask`): `frame_step = 2 × 160` samples; a
per-speaker binary matrix `[S, n_frames]` is filled from Lhotse
supervisions with `start_frame = floor(start·sr/320)` and
`end_frame = ceil(end·sr/320)`; `n_frames` is the audio padded up to a
multiple of 30 s. Inference path (`demo/pipeline.py::_build_diar_mask`):
same but from `{start, end, speaker}` segments in seconds, i.e. an RTTM,
with `round(t·50)`; speakers are relabelled in *arrival order* (first
segment start) before building the mask; the `[T,4]` STNO tensor is cut
into `[n_chunks, 4, 1500]` blocks, padded with the one-hot *silence* class,
and the batch dimension is `(n_speakers × n_chunks)` with the audio tiled
so that block `i` lines up with mel chunk `i` **[verified]**. There is no
resampling code anywhere — the diarizer is expected to give segment
boundaries in seconds, and the 50 Hz grid is imposed by rounding.

Training-time mask corruption (`collators.py`, config `aug:`)
**[verified]**: (i) Gaussian noise on the 4-way probabilities, variance
0.2, applied to 75 % of samples, then clamp-and-renormalise to a
distribution; (ii) segment flips — walk the mask in random segments of
5–50 frames (0.1–1.0 s), with probability 0.1 per segment blend it toward
a random *other* class with a random α, applied to 30 % of samples;
(iii) SpecAugment applied jointly to the mel features and the mask
(mask repeated ×2 to the mel rate, masked, then averaged back);
(iv) MUSAN noise on 30 %. None of these simulates a *late onset* or a
*delayed* mask — they are symmetric jitter.

### 1.4 Real-diarization results and the error analysis

Table 6 (multi-domain FDDT, tcpWER / tcORC-WER) **[verified]**:

| | AMI-sdm test | NOTSOFAR-1 eval-small | Libri2Mix clean | Libri2Mix both | LibriCSS |
|---|---|---|---|---|---|
| ground-truth diarization | 17.6 / 16.7 | 19.7 / 19.1 | 6.9 / 6.9 | 15.9 / 15.9 | 8.8 / 8.8 |
| real diarization | 25.0 / 18.2 | 33.5 / 22.6 | 8.4 / 8.3 | 20.6 / 20.5 | 11.0 / 8.9 |

cpWER (Table 3): AMI-sdm 17.2 oracle → 23.6 real; NOTSOFAR 19.7 → 33.5.
The real diarizer (Table 7, DER / miss / FA / conf): AMI-sdm 17.3 / 8.7 /
3.8 / 4.8; NOTSOFAR-1 22.0 / 7.0 / 6.2 / 8.8; Libri2Mix-both 10.0 / 1.2 /
8.4 / 0.4; LibriCSS 5.5 / 3.6 / 0.4 / 1.5 **[verified]**.

What the authors conclude from the breakdown, all **[verified]** quotes
from §6.1/§6.3:

- ORC-WER (which ignores speaker attribution) degrades little; cpWER a lot
  — "errors related to speaker label assignment (linked to the confusion
  errors in DER) have a more substantial effect than inaccuracies in
  segment boundaries."
- On NOTSOFAR the real-diarization result is 14.5 % deletions, 10 %
  insertions, 9 % substitutions: "the system struggles with omissions".
- "Real diarization can miss portions of a speaker's speech, particularly
  in scenarios with more than two overlapping speakers. The system
  struggles to recover from missed speech segments because it has not
  encountered such cases during training. Moreover, we initialize the
  FDDT parameters in such a way that the model ignores frames marked as
  silence from the very beginning. This further limits the system's
  ability to recover from diarization errors."
- Aligning predicted to reference speakers "becomes especially
  problematic when the diarization system predicts an incorrect number of
  speakers."

Libri2Mix-both is the FA case: 8.4 % false alarm on a two-speaker
mixture turns the target's T frames into O and the other speaker's
non-speech into N, costing 4.7 tcpWER points — FA on the *target's own
row* is what inserts the other voice's words into the target transcript
**[inferred from Table 6/7]**.

### 1.5 Checkpoints

All under https://huggingface.co/collections/BUT-FIT/dicow. Inputs are
"30 s audio + 4-channel STNO mask"; loading needs `trust_remote_code`
**[verified, cards]**.

| checkpoint | base | params (HF) | licence | notes |
|---|---|---|---|---|
| `BUT-FIT/DiCoW_v1` | Whisper large-v3-turbo | ~0.9 B | CC-BY-4.0 | the CSL paper model |
| `BUT-FIT/DiCoW_v3_2` | large-v3-turbo | 1.0 B | CC-BY-4.0 | pre-positional FDDT, less strict suppressive init, sequential decoding with fallback seeking, decoder frozen during FT |
| `BUT-FIT/DiCoW_v3_3` | large-v3-turbo | 0.9 B | CC-BY-4.0 | + Libri3Mix and synthetic LibriSpeech overlaps, STNO noise + SpecAug, data-segmentation fix; ~50 % rel. tcpWER cut on Libri3Mix vs v1 |
| `BUT-FIT/DiCoW_v3_3_large` | Whisper large-v3 (32-layer decoder) | "2 B" F32 | CC-BY-4.0 | same recipe on the full model; this is the encoder Dixtral loads FDDT from |
| `BUT-FIT/SE-DiCoW` (also `SE_DiCoW`) | large-v3-turbo | ~1 B | card says Apache-2.0, paper says CC-BY-4.0 | needs STNO + enrollment audio |
| `BUT-FIT/DiCoW_v3_MLC` | — | — | — | multilingual (MLC-SLM) variant, not examined |

Card numbers (tcpWER, 5 s collar; v1 → v3.3): Libri2Mix-both 21.6 → 9.7,
LibriSpeechMix-2 17.9 → 3.1, AMI-SDM 21.4 → 18.7, NOTSOFAR-1 small-SC
29.8 → 26.6 **[verified]**. The DiariZen diarizer the pipeline ships with
(`BUT-FIT/diarizen-wavlm-large-s80-md`) is **CC-BY-NC-4.0** — the one
non-commercial component in the stack **[verified, DiCoW README]**.

### 1.6 SE-DiCoW: enrollment resolves full overlap

Polok et al., "SE-DiCoW: Self-Enrolled Diarization-Conditioned Whisper",
ICASSP 2026, arXiv 2601.19194 (https://arxiv.org/abs/2601.19194)
**[verified]**. The diarizer output is scanned for the 30 s window that
maximises `Σ_t p_T(t)` for the target; that window is encoded in parallel
with its own STNO mask, and at every encoder layer the main stream
cross-attends to it (query = main, key/value = enrollment), the result is
concatenated with the main stream, passed through a 2-layer MLP with
residual, then the normal encoder layer runs. The code for this is
`SpeakerCommunicationBlock` / `CrossAttentionEnrollBlock` in
`layers.py`, gated by a `tanh` gate initialised at 0 so the block is a
no-op at step 0 **[verified]**. Training: LR 2e-6, batch 96, 40k steps,
loss only on the main stream. tcpWER (oracle → DiariZen), DiCoW v3.3 vs
SE-DiCoW **[verified, paper Table]**: AMI-SDM 14.5 → 14.3 / 18.6 → 18.5;
NOTSOFAR small-SDM 16.0 → 15.8 / 26.6 → 26.1; Libri3Mix-clean 16.0 → 9.7 /
31.6 → 29.3; Libri3Mix-both 27.7 → 19.9 / 38.6 → 35.6. The gain is
entirely in the overlap-heavy synthetic sets; on meetings it is noise.
Important aside: DiariZen "models a powerset of 11 classes with at most
two active speakers at a time", so on Libri3Mix one speaker is
structurally missing (DER 27.6, MSCE 0.99) — that is why real-diarization
Libri3Mix is 3× worse than oracle **[verified]**.

### 1.7 Streaming variants: none exist

Searches for a streaming/online DiCoW (arXiv, GitHub, HF, 2025–2026) found
nothing; the encoder is full-attention Whisper over 30 s windows with
sequential long-form decoding, and the papers do not report RTF
**[verified absence as of 2026-09-12]**. The only streaming
diarization-conditioned ASR with public weights is NVIDIA's (§3.4).

### 1.8 SA-DiCoW: the serialized-output adaptation (arXiv 2510.03723)

Kocour, Karafiát, Polok, Klement, Burget, Černocký, "Adapting
Diarization-Conditioned Whisper for End-to-End Multi-Talker Speech
Recognition", arXiv 2510.03723 v2 (Feb 2026),
https://arxiv.org/abs/2510.03723, code
https://github.com/BUTSpeechFIT/SOT-DiCoW **[verified]**. Runs the DiCoW
encoder once per speaker with that speaker's STNO mask, applies a
per-speaker affine `H̄_u = W_u Ĥ_u + b_u`, and *concatenates along time*
(`d_m × T·|U|`) into a single decoder that emits one serialized
transcript with combined speaker-timestamp tokens `⟨|s1_2.2|⟩`
(8 speakers × 1501 timestamps added to the vocab, three output heads:
lexical, time, speaker). Concatenation beat weighted-sum / average /
masked-average by a mile (NOTSOFAR 21.0 vs 47–59). Two-stage training
(1k steps new params at 2e-4, then ~5k full at 2e-6), batch 192 effective,
speaker-order permutation augmentation so the model cannot memorise slot
→ identity. Results vs per-speaker DiCoW (tcpWER, oracle): LibriMix-2
3.9 vs 4.8, LibriMix-3 18.0 vs 32.1, NOTSOFAR 21.0 vs 18.0; cpWER AMI-SDM
18.1 vs 16.3, AMI-IHM-mix 14.4 vs 13.1. Error decomposition on NOTSOFAR:
leakage 5.1 % and omission 11.0 % vs DiCoW's 3.5 / 2.1 **[verified]**.
This is the closest published precedent for PLAN's single-pass K-stream
design, and it says: joint decoding wins on 3-way overlap and loses 1–3
points on real meetings, mainly through omissions.

## 2. Dixtral — DiCoW encoder inside Voxtral Mini 3B

Paper: Polok, Cornell, Udupa, Černocký, Watanabe, Burget, "Grounding
Spoken LLMs in Multi-Speaker Audio via Diarization Conditioning",
Interspeech 2026, arXiv 2606.18134 (16 Jun 2026), CC-BY-4.0
(https://arxiv.org/abs/2606.18134). Code
https://github.com/BUTSpeechFIT/Dixtral (Apache-2.0; local clone at
`research/dixtral_repo/`). Weights: `BUT-FIT/Dixtral` (TS-ASR, 5 B,
BF16, Apache-2.0), `BUT-FIT/Dixtral_QA`, and `BUT-FIT/Dixtral_TS-ASR`
(the demo's two named checkpoints) **[verified]**.

### 2.1 How DiCoW plugs into Voxtral

Voxtral Mini 3B = Whisper large-v3 encoder (50 Hz) → MLP adapter (two
linears + GELU) that groups 4 consecutive frames → 12.5 Hz audio tokens →
Ministral 3B decoder; 32k context, ≈40 min of audio (Voxtral report,
arXiv 2507.13264) **[verified]**. In code the grouping is literally
`audio_hidden_states.reshape(-1, intermediate_size)` with
`intermediate_size = 4·d_model`, i.e. 1500 encoder frames → 375 LLM tokens
per 30 s window, and Voxtral's `avg_pooler` exists but is not on that
path **[verified, modeling_dixtral.py:316,553]**. Because Voxtral learned
its adapter with a frozen Whisper and DiCoW learned FDDT with a frozen
decoder, the two compose: `DixtralEncoder` subclasses `VoxtralEncoder`
and adds exactly the DiCoW modules (`initial_fddt`, `fddts[0..31]`,
diagonal, all layers, `non_target_fddt_value: 0.5`, `fddt_init:
suppressive`, FDDT weights hot-loaded from `BUT-FIT/DiCoW_v3_3_large`)
**[verified, configs/base.yaml, modeling_dixtral.py:380–425]**.

Frozen vs trained (`params_to_keep_frozen_keywords`): frozen =
`language_model` (the whole Ministral), `audio_tower.conv1/conv2`,
`audio_tower.embed_positions`, `ctc_lm_head`; the paper adds that the
modality adapter is frozen too. Trained = the 32 encoder layers, the 33
FDDT blocks (10× LR), and the leftover `additional_self_attention_layer`.
Best MT-ASR variant adds LoRA on the decoder (`training.use_lora: True`,
`configs/train/w_lora.yaml`); rank/α are not stated in the paper and the
public checkpoint ships with the LoRA *merged* (comment in
`demo/pipeline.py`) **[verified]**. QA fine-tune (`train_qa/dixtral_ft.yaml`)
re-inits from `BUT-FIT/Dixtral`, LoRA on, augmentation off, LR 4e-5.

A useful mode switch: to reason over the *whole* mixture, set
`p_T = 1` on every frame (all other classes 0) and the encoder behaves as
plain Voxtral **[verified, paper §2.2.1]**. That is how the demo's
transcript-conditioned "ask about the whole meeting" path works.

### 2.2 Training data and compute

Lhotse cutsets, weights 3:1:3:1:1 **[verified, base.yaml]**: NOTSOFAR-1
SDM train in 120 s and 30 s cuts, AMI-SDM train in 120 s and 30 s cuts,
Libri2Mix train-clean-100 *noisy* 30 s. Mixer 6 is held out
(out-of-domain). Text normalisation "voxtral" for training,
"whisper_nsf" for eval; language forced to `en`. 20k steps (config cap
25k), peak LR 6e-5, 5k warmup, cosine to 0, effective batch 32 (per-device
1 × grad-acc 4 × 8 GPUs), bf16, gradient checkpointing, on **eight 24 GB
A5000s**; 2-minute utterances fit; AMI decoding is chunked into 5-minute
pieces at diarization-derived pauses (`utils/chunk_longform_cutset.py`);
QA fine-tuning on long-form audio needed H100s **[verified]**.
Augmentation is the DiCoW recipe of §1.3 verbatim.

### 2.3 NSF-QA

https://hf.co/datasets/popcornell/NSF-QA, built on NOTSOFAR-1
**[verified]**. Content QA (entity / topic / yes-no / detail, answerable
from text) and paralinguistic QA (emotion from emotion2vec 9-class labels
on close-talk audio with GT segmentation; gender from session metadata),
plus per-speaker summaries (<50 words, five Gemini-2.5-Flash references,
ROUGE-L max). Train counts: entity 1,111, topic 1,139, yes/no 1,557,
detail 775, emotion 1,196, summaries 390; eval 1,092 / 1,114 / 1,515 /
753 / 1,145 / 379. Gemini 2.5 Flash both writes the references and judges
(binary correct/incorrect → accuracy); the authors flag the circularity
and argue it favours Gemini **[verified]**.

### 2.4 Numbers

Table 2, cpWER %, **DiariZen** diarization for DiCoW and Dixtral
**[verified]**:

| system | NSF-1 small | AMI SDM | LSMix-1 | LSMix-2 | LSMix-3 | Mixer6 CH4 | avg |
|---|---|---|---|---|---|---|---|
| Voxtral Mini Transcribe V2 | 54.4 | 42.3 | 2.0 | 28.2 | 42.3 | 19.4 | 31.4 |
| VibeVoice | 35.8 | 33.7 | 2.1 | 50.8 | 72.8 | 16.0 | 35.2 |
| DiCoW v3.3 | 26.6 | 18.6 | 1.8 | 3.1 | 21.7 | 11.9 | 14.0 |
| Gemini 3.0 Flash | 39.1 | 56.3 | 4.5 | 23.3 | 84.7 | 58.3 | 44.4 |
| **Dixtral** | 29.1 | 19.8 | 2.1 | 3.6 | 23.5 | 14.4 | 15.4 |

The "29.0 / 19.8 / 16.0 absolute" in PLAN §1 are the macro-average gaps
44.4−15.4, 35.2−15.4, 31.4−15.4. Dixtral is 1.4 points *behind* the pure
DiCoW it was built from; the LLM buys QA, not transcription.

Table 3, oracle diarization ablation (cpWER; columns as above minus avg)
**[verified]**: random-init encoder ("wo/ swap") 26.3 / 19.8 / 1.9 / 2.3 /
3.6 / 12.6; full encoder swap 26.4 / 21.0 / 1.9 / 2.4 / 4.6 / 20.4; FDDT
swap only 26.3 / 17.1 / 1.8 / 2.2 / 3.5 / 16.2; + LoRA on decoder
**21.3 / 16.4** / 2.0 / 2.6 / 4.8 / 14.5; after QA+summary FT 24.3 / 19.4 /
2.2 / 3.1 / 5.6 / 26.1. The full encoder swap converges fastest (32.0 %
NSF-1 at 2k steps vs 37.7 FDDT-only vs 38.8 random) but ends worse
because it moves Voxtral's embedding space away from the frozen adapter;
QA fine-tuning costs transcription accuracy (Mixer6 14.5 → 26.1).

Table 4, NSF-QA (accuracy %, ROUGE-L) **[verified]**: far-field Dixtral
zero-shot 54.6 content / 25.4 emotion / 43.2 gender / 24.4; +LoRA
zero-shot 56.9 / 22.2 / 73.3 / 15.4; fine-tuned 73.0 / 47.6 / 95.5 / 41.4;
Gemini 3.0 Flash far-field 55.1 / 29.3 / 74.1 / 23.7; close-talk Voxtral
Mini v1 68.3 / 25.5 / 49.7 / 24.1, close-talk Gemini 68.1 / 34.1 / 75.0 /
26.4.

### 2.5 VRAM, latency, diarizer sensitivity

Not reported **[verified absence]**: the paper has no RTF, no latency, no
VRAM table; the README's guidance is "recordings longer than ~5 minutes
should be chunked" and the demo offloads DiariZen to CPU so that "the
diarization and Dixtral models never occupy VRAM at the same time".
**[inferred]**: 5 B parameters in bf16 ≈ 10 GB of weights, so a 24 GB
card runs inference with the 4-speaker batch the demo uses
(`TRANSCRIBE_BATCH_SIZE = 4`, audio tiled, only the mask differs). The
paper's decoding-cost argument is that `S` separate passes of length `N`
are `O(S·N²)` vs `O((SN)²)` for one serialized transcript, so per-target
passes scale better as the LLM grows. On diarizer sensitivity the paper
punts entirely: "the models' sensitivity to this specific system has
already been studied [DiCoW]" — there is no Dixtral experiment with any
diarizer but DiariZen and no over/under-segmentation study
**[verified]**.

## 3. Other speaker-activity-conditioned recognisers and LLMs

### 3.1 DM-ASR (arXiv 2604.22467)

Li, Cheng, Zhu, Wang, Liu, Li, "DM-ASR: Diarization-aware Multi-speaker
ASR with Large Language Models", Apr 2026
(https://arxiv.org/abs/2604.22467) **[verified]**. Whisper-large-v3-turbo
encoder + 2-layer MLP + a small LLM (Gemma-3 270M, Qwen3-0.6B, Qwen3-1.7B).
Diarization enters as *text*, not as a mask: each diarized segment becomes
one dialogue turn "transcribe speaker `<|spk_idx_k|>` within
`<|start_of_time|>` t₁ t₂", timestamps discretised to 0.1 s tokens,
subsequent turns reuse the KV cache. Not streaming (15–25 s chunks).
Robustness trick worth copying: during training the speaker label and the
timestamps in the prompt are perturbed with p = 0.1, so the model learns
to override wrong cues from acoustics and context. 2,900 h real data
(AliMeeting, AISHELL-4, MISP2025, RAMC, HKUST, AMI, ICSI, Fisher, MLC-SLM),
no simulation. 1.7 B model: AMI-IHM cpWER 17.16 / tcpWER 18.09 (its own
DER 27.87), AMI-SDM 23.45 / 24.97 (DER 14.83), Fisher 16.76 / 16.89,
AliMeeting cpCER 21.60. No code or weights announced.

### 3.2 G-STAR (arXiv 2603.10468)

Peng et al., "G-STAR: End-to-End Global Speaker-Tracking Attributed
Recognition", submitted to EMNLP 2026
(https://arxiv.org/abs/2603.10468) **[verified]**. The one LLM system that
consumes *Streaming Sortformer* directly: the tracker is initialised from
`nvidia/diar_streaming_sortformer_4spk-v2` and fine-tuned on 90 s chunks;
its per-frame speaker embeddings `V(t)` are *interleaved* into the
acoustic token stream every K frames (`E = Interleave(U, V; K)`) before a
Qwen2-7B-Instruct decoder with LoRA (r 64, α 16, from FireRed-LLM);
output is SOT `[<t_st>, w, <t_ed>, <spk=k>]*` with `k` the AOSC
arrival-order slot, so identity is consistent across chunks. Three-stage
training, hierarchical CE (timestamps ×1.5, speaker tags ×2). AMI local
(20 s, oracle VAD): cpWER 24.86 / DER 19.00; AMI full-meeting: 30.85 /
32.23; Fisher 10.29 / 8.18 local. Not evaluated as a true streaming
system ("future work will extend ... to real-time streaming"), no latency
numbers, "we will release the model and code" but nothing linked.
Interleaving beat other fusions by ~4 cpWER in their ablation — a data
point for PLAN's "sum into the frame embedding" choice: interleaving as a
separate token every K frames is a live alternative **[inferred]**.

### 3.3 CALM (arXiv 2601.22792)

Shakeel, Fukumoto, Maeda, Lin, Watanabe, ICASSP 2026
(https://arxiv.org/abs/2601.22792) **[verified]**. Not diarization-
conditioned: target-speaker extraction from an ECAPA/RawNet3 192-d
enrollment embedding applied as FiLM (`γ(E_s) ⊙ H + β(E_s)`) to Conformer
encoder states, plus dynamic-vocabulary contextual biasing. Offline. The
relevant number for us is the FiLM-on-embedding mechanism, which is the
identity half of PLAN's `d_t`: LibriSpeech2Mix biased-WER 12.7 → 4.7,
CSJMix2 biased-CER 16.6 → 8.4, AMI-IHM-mix 34.7 → 22.1 on biased words
with overall WER 37.4 → 39.1. No weights yet.

### 3.4 NVIDIA `multitalker-parakeet-streaming-0.6b-v1`

https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1
(NVIDIA Open Model License), paper Wang, Park, Medennikov et al.,
"Speaker Targeting via Self-Speaker Adaptation for Multi-talker ASR",
Interspeech 2025, arXiv 2506.22646 **[verified]**. The only *streaming*
system with public weights that eats Sortformer activity:

- Fast-Conformer (NEST-pretrained, 0.6 B) hybrid transducer/CTC,
  cache-aware streaming; `att_context_size` in 80 ms frames: `[70, 0]` =
  80 ms latency, `[70, 13]` = 1.12 s (the evaluated setting).
- One ASR instance per speaker, all fed the same mixed audio; per-speaker
  activity `y_spk_k ∈ (0,1)^{B×T×1}` is injected at the pre-encode layer
  as `f_inj(X, y) = FFN(X ⊙ y)` (paper eq. 2) — the earlier offline
  Sortformer paper (arXiv 2409.06656) used additive sinusoidal kernels
  `Ã = A/‖A‖₂ + Γᵀ P`, `κ_{k,z} = sin(2πkz/M)`, and measured 0.78 %
  runtime overhead and AMI-test cpWER 26.71 with Sortformer vs 26.83 with
  oracle labels **[verified]**.
- The glue is `SpeakerTaggedASR` in
  `nemo/collections/asr/parts/utils/multispk_transcribe_utils.py`
  (NVIDIA-NeMo/Speech) **[verified from source]**: the diarizer's
  `(B, T, 4)` sigmoid stream is passed **raw** by default
  (`binary_diar_preds: False`; when true it is thresholded at 0.5) via
  `asr_model.set_speaker_targets(active, inactive)`; a speaker counts as
  active if its probability exceeds 0.5 anywhere in a window of the last
  `cache_gating_buffer_size = 2` chunks (`cache_gating`); there is an
  alternative `masked_asr` path that multiplies the waveform (mask
  repeated ×8 to 10 ms) or the pre-encoded features by the activity; and
  there is **no resampling** — the code asserts that the diarizer and ASR
  `window_stride` and subsampling factors are equal (both 8× on 10 ms →
  80 ms) and raises otherwise.
- Streaming Sortformer v2 settings in the tutorial: `chunk_len 6`,
  `chunk_right_context 7`, `spkcache_len 188`, `fifo_len 188` = the 1.04 s
  preset; ASR at 1.12 s.
- cpWER with Streaming Sortformer v2 (card): AMI-IHM 21.26, AMI-SDM
  **37.44**, CH109 15.81, Mixer6 23.81; single-speaker HF-leaderboard
  average 7.44. Paper (LibriSpeechMix, streaming, 560 ms): 1-mix 4.0,
  2-mix 5.6, 3-mix 11.0; CH109 26.21 at 1.12 s vs cascaded 27.38. No
  oracle-vs-Sortformer ablation in the paper.

The AMI-SDM 37.4 vs DiCoW's 18.6 is the price of streaming plus a 0.6 B
CTC/transducer model plus a diarizer that (§4) misses a lot on far-field
AMI. It is the number PLAN's stage-1 gate should be compared against as
the *streaming* floor, not Dixtral's offline 19.8 **[inferred]**.

### 3.5 Others seen in the sweep (not activity-conditioned at the frame level)

TagSpeech (ACL 2026, arXiv 2601.06896): frozen LLM, separate semantic and
speaker streams, interleaved time anchors, SOT; beats Qwen-Omni/Gemini on
AMI/AliMeeting DER. SpeakerLM (AAAI 2026, arXiv 2508.06372): end-to-end
who-spoke-when-and-what MLLM with a speaker registration mechanism
(7 B, 7,639 h per DM-ASR's comparison, AliMeeting cpCER 16.05).
"Balancing ASR and diarization in end-to-end LLMs" (Interspeech 2026,
arXiv 2606.13095): dual encoder, feature interleaving, +18 % AliMeeting.
"Diarization-Guided Qwen-ASR Adaptation" (arXiv 2607.08208): cascaded
clustering diarizer + Qwen3-ASR-1.7B, two-speaker MLC-SLM. None report
streaming, none take a per-frame activity mask, none release weights as of
the search **[verified abstracts only]**.

## 4. Sortformer v2.1 as the conditioning source

Card: https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1;
papers arXiv 2409.06656 (Sortformer), 2507.18446 (Streaming Sortformer,
Interspeech 2025) **[verified]**.

Architecture: 17-layer NEST Fast-Conformer (8× subsampling of a 10 ms mel
→ **80 ms frames**) + 18-layer Transformer (d 192) + feed-forward to 4
sigmoids per frame; 117 M (card) / 123 M (paper); arrival-order speaker
cache (AOSC) + FIFO + current chunk with right context; trained on ~5,000 h
real + simulated (Fisher, **AMI**, VoxConverse, ICSI, AISHELL-4, DIHARD-III
dev, NIST SRE-2000, DiPCo, AliMeeting, **NOTSOFAR-1**; NIST SRE 04–10 and
LibriSpeech simulations); augmentation is random permutation of the cache
speakers. The official presets (units of 80 ms frames):

| preset | chunk_len | right ctx | fifo | update period | cache | latency | RTF (RTX 6000 Ada) |
|---|---|---|---|---|---|---|---|
| very high | 340 | 40 | 40 | 300 | 188 | 30.4 s | 0.002 |
| low | 6 | 7 | 188 | 144 | 188 | 1.04 s | 0.093 |

The paper adds a 0.32 s preset at RTF 0.18; livediar's `presets.py` has
`low` = the official one and `ultra` (chunk 3, right 1 → 0.32 s)
**[verified, local]**. Input-buffer latency = (chunk_len + right_ctx) × 80 ms.

DER at 1.04 s, v2.1 (card) **[verified]**: DIHARD-III ≤4 spk 15.09, ≥5
spk 41.42, full 20.21; CALLHOME 2/3/4 spk 6.65 / 11.25 / 13.35, full
11.19; CH109 5.09; AliMeeting near/far 12.60 / 15.60; AMI-IHM 16.67,
**AMI-SDM 20.57**; NOTSOFAR-1 ≤4 spk 17.26, ≥5 spk 36.76, full 28.75.
What changed from v2 to v2.1 is not documented anywhere I could find; the
only evidence is the meeting-corpus rows (AliMeeting near 19.98 → 12.60)
and the fact that AMI and NOTSOFAR are now in the training list — so the
AMI DER is in-domain and optimistic for us **[inferred]**.

Post-processing (NeMo `examples/speaker_tasks/diarization/conf/post_processing/`),
six operations tuned by Optuna per dataset **[verified, YAML]**:

| config | onset | offset | pad_onset | pad_offset | min_duration_on | min_duration_off |
|---|---|---|---|---|---|---|
| `diar_streaming_sortformer_4spk-v2_dihard3-dev` | 0.56 | 1.0 | 0.063 | 0.002 | 0.007 | 0.151 |
| `diar_streaming_sortformer_4spk-v2_callhome-part1` | 0.641 | 0.561 | 0.229 | 0.079 | 0.511 | 0.296 |
| `sortformer_diar_4spk-v1_dihard3-dev` (offline) | 0.64 | 0.74 | 0.06 | 0.0 | 0.1 | 0.15 |

Note the CallHome set pads onsets by 229 ms — the tuner found that the
model turns on late. There is no v2.1-specific YAML; livediar's
`turns.py` replaces all of this with per-slot hysteresis on the raw
sigmoid stream **[verified, local]**.

Known failure modes:

- **Missed speech dominates on far-field meetings.** Our own AMI harness
  (`bench/ami/results/*.json`, Sortformer v2.1 streaming): EN2002b DER
  25.7 = miss 19.1 + FA 4.8 + conf 1.8; ES2004a DER 21.5 = miss 15.0 +
  FA 4.5 + conf 1.9 **[verified, local]**. The independent benchmark
  arXiv 2509.26177 says missed speech is the largest component for every
  model except Sortformer, whose distinguishing weakness is *speaker
  confusion*, and that boundary precision on long segments, not detection
  of short ones, is where models fail **[verified]**. Both patterns matter
  for DiCoW-style conditioning (§5).
- **Overlap onset lag.** livediar's README records that "the diarizer
  itself can miss the second voice for a second at overlap onset; no
  attribution logic can recover words the diarizer never flagged as
  shared", and that in overlap "Sortformer's probabilities saturate for
  both talkers" so they carry no dominance information **[verified, local
  observation; not quantified in any NVIDIA paper]**.
- **Four-slot cap.** Hard-wired; DIHARD ≥5 spk 41 %, NOTSOFAR ≥5 spk
  37 %. Arrival-order errors grow with speaker count (the Sort-loss paper
  says "arrival time estimation is not always correct", hence the hybrid
  PIL+sort loss). Slot swaps and merges at overlaps are the confusion
  errors — livediar binds identity to TitaNet profiles for exactly this
  reason **[verified, local README]**.
- **Training-length mismatch.** Offline Sortformer is trained on 90 s
  windows and degrades on long recordings; the streaming version avoids
  this via the cache but 2509.26177 still had to chunk offline v2 at 12
  min.
- Primarily English; degrades on noisy / out-of-domain audio (card).

NVIDIA's own uses of Sortformer as conditioning are the MS-Canary
integration in 2409.06656 (kernels into the encoder, sorted serialized
transcripts) and the streaming Parakeet of §3.4; I found no NVIDIA paper
feeding Sortformer into an LLM, and the one third-party LLM that does is
G-STAR (§3.2) **[verified absence]**.

## 5. Pitfalls of feeding Sortformer masks into DiCoW / Dixtral

**Frame rate.** Sortformer emits at 12.5 Hz; DiCoW's FDDT wants 50 Hz;
Voxtral's LLM tokens are 12.5 Hz. The ratio is an exact 4, so the
conversion is `repeat_interleave(4)` on the per-slot activity *before*
the STNO products (never after — averaging STNO classes across a boundary
produces soft masks the model was only trained to tolerate as noise)
**[inferred from code]**. Equivalent and simpler: turn the binarised 80 ms
stream into `{start, end, speaker}` segments in seconds and hand them to
`_build_diar_mask`, which rounds to 50 Hz. Either way boundaries are
quantised to 80 ms (±40 ms jitter), which is inside the 0.1–1.0 s
segment-flip augmentation DiCoW trained with, so it is benign. Window
alignment matters more: the mask is cut into 1500-frame blocks that must
start where the 30 s mel windows start, and padding must be one-hot
silence, not zeros (`_stno_chunks` does this) **[verified]**. For PLAN's
own model there is no conversion at all — one Sortformer frame per Mimi
frame per Voxtral-rate token.

**Mask lag.** At the `low` preset the activity for frame `t` exists only
1.04 s after the audio (0.32 s at `ultra`). DiCoW/Dixtral are offline so
this is invisible in batch evaluation, but in a live pipeline the last
~1 s of every window has no mask; holding the last state or padding with
silence both bias toward deletions (silence frames are suppressed from
step 0 by the initialisation, §1.2) **[inferred]**. For the K-stream LM
the choice is between delaying the audio stream by the diarizer latency
(adds directly to response latency) and training with delayed
conditioning `d_{t−L}`; the second is free at inference and can be made
an augmentation over `L ∈ {0, 4, 13}` frames **[inferred]**.

**Under-segmentation: missed onsets and missed overlap.** This is the
Sortformer failure (miss 15–19 % on AMI) landing on DiCoW's documented
blind spot ("struggles to recover from missed speech segments because it
has not encountered such cases during training") **[verified both
sides]**. A late onset labels the target's first syllables S or N, which
the FDDT maps toward zero — those words are deleted, and because Whisper
decodes autoregressively a truncated first word can also derail the
segment. The BUT augmentations are symmetric jitter and do not simulate
"the diarizer switched on late"; if the DiCoW path is used with our masks
it needs either `pad_onset` (NeMo's CallHome tuning found 0.23 s) and a
lower onset threshold, or a fine-tune whose corruption includes
delayed-onset shifts of 0–1 s **[inferred]**. Missed *overlap* (second
voice not flagged for ~1 s) is the same failure from the other speaker's
perspective: their frames become N instead of O, and the FDDT is
initialised to suppress N.

**Over-segmentation: false alarms and spurious slots.** FA of 4.5–4.8 %
on our AMI runs. FA on a *non-target* row only turns T into O, which
DiCoW handles (it was trained on plenty of O); FA on the *target's* row
inserts the other speakers' words into the target's transcript
(Libri2Mix-both: 8.4 % FA → +4.7 tcpWER) **[verified numbers, inferred
mechanism]**. A phantom fifth slot cannot happen (4 max) but a phantom
*fourth* slot from a slot swap does, and any slot the diarizer invents
produces an extra transcript that cpWER counts fully as insertions; DiCoW
notes speaker-count mismatch is the hardest alignment case.

**Confusion / slot swaps.** DiCoW's own analysis: confusion errors hurt
cpWER far more than boundary errors (ORC-WER barely moves). Sortformer's
distinguishing weakness per 2509.26177 is confusion. A mid-meeting slot
swap concatenates two people's words under one label for the rest of the
meeting; nothing in DiCoW/Dixtral can undo it because the mask *is* the
identity. PLAN's TitaNet binding (identity carried by the embedding, not
the slot) is the right layer to fix this, and SE-DiCoW's Libri3Mix result
(16.0 → 9.7 with an enrollment reference) is the evidence that an
identity reference resolves what the activity mask cannot **[verified
results, inferred transfer]**.

**Soft vs hard.** DiCoW was trained and evaluated only on hard masks, and
its authors say soft activations are an open problem; the Gaussian-noise
augmentation (σ² 0.2 on 75 % of samples) means mildly soft masks are
in-distribution but saturated 0.9/0.9 overlap probabilities from Sortformer
are not what the products in §1.1 expect (`p_T = 0.9·0.1 = 0.09`,
`p_O = 0.81` — the mask says "overlapped" with a large N residue).
Binarise with hysteresis (livediar `turns.py`) before the STNO products
when driving DiCoW/Dixtral; feed raw sigmoids only to a conditioning
module we train ourselves, as NVIDIA does (`binary_diar_preds: False`)
**[inferred]**.

**Diarizer-shape mismatch.** DiariZen (what all BUT numbers use) is a
powerset model with at most two simultaneous speakers, at 16 ms-class
resolution, run offline with clustering across the whole recording;
Sortformer allows four simultaneous speakers, decides at 80 ms with ≤1 s
of lookahead, and cannot revise the past. Expect more O frames, more
misses on far-field, and no global consistency. There is no published
Dixtral-with-Sortformer number; the closest is Parakeet-streaming's
AMI-SDM 37.4 vs Dixtral+DiariZen 19.8, which bounds what the diarizer
swap alone might cost **[inferred]**.

**Arrival order is free.** Dixtral's demo relabels DiariZen's opaque ids
into arrival order before building masks; Sortformer's slots are already
arrival-ordered, so the slot index can feed the mask builder directly
**[verified]**.

**Licences.** DiCoW weights CC-BY-4.0, Dixtral Apache-2.0, Sortformer
NVIDIA Open Model License, Voxtral Apache-2.0; only DiariZen is
CC-BY-NC, and replacing it with Sortformer removes the non-commercial
component **[verified]**.

## Implications for PLAN.md

1. **The stage-1 gate needs a different baseline number.** "cpWER ≤
   Dixtral" must mean Dixtral *driven by Sortformer v2.1 masks*, not the
   paper's 19.8 on AMI with DiariZen. That baseline does not exist in the
   literature and is cheap to produce: `demo/pipeline.py` takes any
   `{start, end, speaker}` segment list, so `bench/ami_bench.py` can emit
   Sortformer segments in that form and run `BUT-FIT/Dixtral_TS-ASR` on a
   24 GB GPU (or Modal) with no training. Do this before touching the
   K-stream model; it also measures how much of the gap is the diarizer
   (compare against Dixtral with AMI reference RTTMs, and against the
   streaming Parakeet's 37.4).

2. **Keep additive-affine, drop QKb from the ablation list.** DiCoW's
   evidence is unambiguous: per-layer affine FDDT (not bias-only, not
   input-only) with identity-for-T/O and scaled-down-for-S/N
   initialisation is what lets a pretrained backbone survive the new
   input; QKb hallucinates at init and destroys timing. For §2's "summed
   into the frame embedding" the concrete recipe is: one FDDT-style
   4-class diagonal affine at the input plus one per backbone layer (or
   at least every k-th), `W_T = W_O = I`, `W_S = W_N = 0.5·I`, 10× LR on
   these parameters, FDDT-only warm-up before unfreezing. G-STAR's
   interleave-a-speaker-token-every-K-frames is the one alternative worth
   keeping in the ablation.

3. **Corruption in §4 must include Sortformer-shaped errors.** Copy the
   BUT recipe (Gaussian σ² 0.2 on 75 %, segment flips 0.1–1 s on 30 %/10 %,
   SpecAugment jointly on audio and mask) and add the three things it
   lacks: delayed onsets (shift a speaker's rise 0–1 s late), delayed
   conditioning (`d_{t−L}`, L ∈ {0, 4, 13} frames), and slot swaps with
   the identity embedding kept correct — the last is what makes identity
   binding, not the slot, carry the name. DM-ASR's p = 0.1 label
   perturbation is the same idea for prompt-side cues.

4. **The frame-rate coincidence is exact end to end.** Sortformer 80 ms =
   Mimi 80 ms = Voxtral's post-adapter 80 ms token. Only the optional
   DiCoW-encoder side path needs the ×4 repeat, and it needs binarised
   masks; the main path can take raw sigmoids since we train it.

5. **Identity embeddings are load-bearing, not decoration.** SE-DiCoW
   shows an enrollment reference resolves full-overlap ambiguity that the
   activity mask cannot (Libri3Mix 16.0 → 9.7); CALM shows FiLM on a
   192-d speaker embedding is a working injection. The §5 ablation
   "identity as embedding vs text prompt" should be scored on the
   overlap-heavy subset, where the difference will show.

6. **Single-pass K-stream has a known cost.** SA-DiCoW, the closest
   precedent (per-speaker encodings concatenated, one decoder, up to 8
   speakers), gains on 3-way overlap and loses 1–3 cpWER on real meetings
   through omissions. Budget that into the stage-1 target rather than
   expecting parity with per-target decoding.

7. **Sortformer's cap is 4 and its AMI DER is in-domain.** K = 8 in §2
   needs a different diarizer (mago-ai's 8-speaker Ultra-Sortformer exists
   on HF, unexamined), and the 20.6 AMI-SDM DER on the card is with AMI in
   training — our measured 21–26 miss-dominated DER is the honest input
   quality. Report tcpWER with the 5 s collar alongside cpWER so numbers
   are comparable with the BUT tables (meeteval does both).
