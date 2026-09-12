# Full-duplex spoken dialogue modelling — what exists, with numbers

Research note for PLAN.md §1, §2 and §5. Written 2026-09-12 from primary
sources fetched that day. Every claim carries a URL. Tags: **[V]** = read
directly in the cited source (paper HTML/PDF text, model card, README);
**[I]** = inferred, or reported second-hand by an abstract/summary and not
confirmed against the body; **[A]** = absence claim (searched, not found —
the weakest kind of evidence).

Sections: 1 Moshi and its public code · 2 Kyutai's RL post-training and
Seamless Interaction · 3 The 2025–26 field and how each model decides to
speak · 4 Benchmarks and metric definitions · 5 Multi-party (3+) work ·
Implications for PLAN.md.

---

## 1. Moshi (Kyutai, arXiv 2410.00037)

Paper: https://arxiv.org/abs/2410.00037 (HTML body used:
https://arxiv.org/html/2410.00037v1). Code: https://github.com/kyutai-labs/moshi.
Finetuning: https://github.com/kyutai-labs/moshi-finetune.

**Backbone.** Helium is a 7B text LM: 32 layers, model dim 4096, 32 heads,
4096-token context, pretrained on 2.1T tokens of public English text [V].
On top of it the *Depth Transformer* that emits the codebook tokens within
a frame has 6 layers, dim 1024, 16 heads [V]; the linear layers of the depth
transformer are *not shared* across codebook index — "a different set of
weights for each index k" [V]. Total per-step token layout is K = 17
sub-sequences: 1 text + 8 Moshi audio + 8 user audio [V]. The GitHub README
states the temporal transformer is 7B and needs ~24 GB GPU memory in
PyTorch without quantisation [V].

**Mimi codec.** 24 kHz input, 12.5 Hz frame rate (80 ms hop), 1.1 kbps at
the 8 codebooks Moshi uses, split-RVQ where codebook 1 is a *semantic*
codebook distilled from WavLM and codebooks 2–8 are acoustic; fully causal
SeaNet encoder/decoder with Transformer blocks in the bottleneck; 80 ms codec
latency [V paper + README]. The README compares it favourably to
SpeechTokenizer (50 Hz, 4 kbps) and SemantiCodec (50 Hz, 1.3 kbps) [V].
(Mimi is trained with more quantizers than Moshi consumes; the exact
released count is not restated in the fetched text — treat "32" as [I].)

**Delays (Table 1 of the paper) [V].** Acoustic delay between the semantic
codebook and the acoustic codebooks: 2 steps during audio pre-training, 1
step for post-training, Fisher fine-tuning and instruct fine-tuning. Text
delay (how far the inner-monologue text leads its own audio): sampled in
±0.6 s during pre-training, fixed at 0 for all later stages; +2 s in the
TTS-mode variant. "Introducing a delay of 1 or 2 steps between the semantic
and acoustic features greatly improves the quality of the generation" [V].
Hence the headline latency: 160 ms theoretical = 80 ms frame + 80 ms
acoustic delay, ~200 ms measured on an L4 [V README + paper].

**Inner monologue.** At each 80 ms step the text token is predicted *before*
the 8 audio tokens of Moshi's own stream; word ends are marked with PAD /
EPAD tokens that "split the decision of ending a word", and in English
conversational speech ~65% of text-stream tokens are padding [V]. Padding
tokens get weight 0.5 in the cross-entropy; the text embedding / output
linear get a 0.75 learning-rate multiplier [V]. Only Moshi's own stream has
a text stream; the user's stream is audio-only [V].

**How Moshi decides to speak.** There is no turn token. Silence is what the
audio tokens decode to; during silence the text stream emits PAD. "Forcing a
EPAD token will make Moshi start talking immediately" [V] — i.e. the
speak/listen policy is implicit in the joint distribution over the two
audio streams and the text stream.

**Training stages [V].**
1. Helium text pre-training, 2.1T tokens.
2. Audio pre-training on 7M hours of unsupervised audio, single-stream,
   1M steps at batch = 16 h of audio.
3. Multi-stream pre-training, 100k steps at batch = 8 h: single-channel
   data is turned into two streams by running PyAnnote diarization, picking
   one speaker at random, and masking the waveform into "one with the
   speaker, and one with the residual (potentially several speakers)".
4. Fisher fine-tuning: 2,000 h of two-channel telephone speech, upsampled
   8 → 24 kHz, 10k batches.
5. Instruct fine-tuning on >20k hours of *synthetic* two-party dialogue.
   Recipe: Helium writes transcripts prompted with Wikipedia / StackExchange
   context (and Fisher-style conversational prompts); a *multi-stream
   streaming TTS* built from the Moshi architecture and trained on Fisher
   renders both channels; Moshi's voice is a single actor who recorded
   monologues in >70 speaking styles; user voices are randomly sampled per
   example.
Augmentations on the user stream: gain variation (50% of samples), DNS
noise at −30 to +6 dB (30%), simulated echo of Moshi's own output at 0.2
amplitude and 100–500 ms delay plus reverb (30%) [V].

**Release.** Weights CC-BY 4.0; Python code MIT, Rust backend Apache 2.0.
Variants: moshiko (male) and moshika (female) in PyTorch bf16/int8, MLX
int4/int8/bf16, Candle int8/bf16 [V].

**moshi-finetune [V, from README].** Supports both LoRA (`lora.enable:
true`, rank typically ≤128, the recommended default) and full fine-tuning.
Data format: *stereo* WAV where the **left channel is Moshi, the right
channel is the user**, plus a `.jsonl` manifest of `{"path": ..., "duration":
...}` and per-file `.json` transcripts with word timestamps (produced by the
provided `annotate.py`). Recommended LR 2e-6. Memory: one H100 at
`batch_size: 16`, `duration_sec: 100` peaks at ~39.6 GB; 8×H100 with FSDP
~23.7 GB per GPU. There is no K>2 speaker convention anywhere in the repo —
a room model needs its own loader [V by absence in README, A].

## 2. Kyutai's RL post-training (moshika-rl-seamless) and Seamless Interaction

Model cards: https://huggingface.co/kyutai/moshika-rl-seamless and
https://huggingface.co/kyutai/personaplex-rl-seamless. Paper: "Multi-Faceted
Interactivity Alignment in Full-Duplex Speech Models", Ohashi, Zeghidour,
Défossez, Kharitonov (Kyutai / Gradium), https://arxiv.org/abs/2606.11167
(HTML body https://arxiv.org/html/2606.11167v1). Samples:
https://huggingface.co/datasets/kyutai/interactivity-alignment-samples.

**Recipe [V].** GRPO on the base model (kyutai/moshika-pytorch-bf16, or
NVIDIA's PersonaPlex-7B). Four "interactivity axes" each get a training set
mined automatically with VAD from two-channel human dialogue: **pause
handling**, **turn-taking**, **backchanneling**, **user interruption**.
Prompts are user-channel segments of at least τ_min = 4–5 s (pause / turn /
backchannel) or 3 s (interruption); a random 0–30 s of preceding context is
prepended with a linear schedule (the "without context" ablation
generalises worse). The model card adds: "up to 2,000 relevant segments"
per axis [V card].

**Rewards are rule-based, per axis [V].**
- Pause: −1 if the model produces speech >1 s during the user's within-turn
  pause, else 0.
- Turn-taking: R = −d, the negative response delay after the user's turn end.
- Backchanneling: F1 between generated short utterances (≤1 s) and
  ground-truth backchannel positions.
- Interruption: negative delay between the user's barge-in and the model's
  yield/response.
- Plus an LLM-judge reward (Qwen3-235B, 0–2 semantic relevance of the
  model's text stream) to stop content quality collapsing. The authors note
  the RL "optimizes the text token stream" — the policy gradient is taken
  on the inner monologue, which is why it transfers to other Moshi-family
  models but is architecture-specific [V].

**Optimisation [V].** 32 segments × 16 completions = 512 generations per
epoch, 100 epochs, 32 H100 with FSDP, LR 2e-7 cosine, KL β = 0.01, clip
ε = 0.2. This is a *small* RL run — tens of GPU-hours per epoch class.

**Data [V].** Fisher (2,000 h telephone) and Seamless Interaction (4,000 h).
The released checkpoints are the Seamless-trained ones; Fisher-trained
variants "caused safety degradation in some cases".

**Results, Full-Duplex-Bench v1 (Candor pause set etc.) [V paper tables].**
Moshi base → +RL(Seamless): pause TOR 0.528 → 0.463 (lower better);
turn-taking latency 0.162 → 0.160 s; backchannel frequency 0.074 → 0.101;
turn-taking TOR 0.739 → 0.958; interruption latency 1.377 → 0.409 s;
GPT-4o quality 3.44 → 3.63. PersonaPlex base → +RL: pause TOR 0.444 →
0.356; turn latency 0.219 → 0.086 s; interruption latency 0.271 → 0.223 s;
GPT-4o 4.50 → 4.53. UTMOSv2 speech quality unchanged (2.56 → 2.64). On
FDB-v2 (dynamic examiner) PersonaPlex+RL is best on all four task families,
e.g. Daily: turn-taking fluency 4.017, instruction following 3.197 [V].
Ablations: dropping any single axis dataset re-creates that axis's failure
(over-speaking without pause data, under-speaking without turn data);
dropping the LLM-judge reward costs the most content quality (GPT-4o 3.05)
[V]. No human evaluation; rule-based rewards flagged as non-scalable [V].

**Licences.** moshika-rl-seamless: CC BY-NC 4.0, research only [V card].
personaplex-rl-seamless: CC BY-NC 4.0 *and* the NVIDIA Open Model License
[V card]. The non-commercial term is inherited from the dataset.

**Seamless Interaction dataset (Meta).** Paper
https://arxiv.org/abs/2506.22554; page
https://ai.meta.com/research/seamless-interaction/; code/download
https://github.com/facebookresearch/seamless_interaction; HF
https://huggingface.co/datasets/facebook/seamless-interaction. 4,000+ hours,
4,000+ participants, ~65,000 interactions, in-person face-to-face dyads with
4K video [V Meta page]. Audio: 48 kHz 16-bit WAV, one file per participant,
released as "speaker-bleed denoised" separate channels (the participants
are co-located, so the raw channels bleed — inferred from the denoising
[I]); VAD at 100 Hz and time-aligned transcripts per participant in JSONL
[V HF card + GitHub]. Splits: *improvised* (at least one professional actor,
scripted prompts) and *naturalistic* (ordinary pairs), each with
train/dev/test; ~27 TB total, WebDataset batches of 50–100 GB on HF or
per-file S3 [V GitHub]. Licence **CC-BY-NC 4.0** [V GitHub + HF]. Known
issues: ~10% of interactions have metadata problems (timestamp misalignment,
participant-ID errors) [V GitHub]. Strictly dyadic.

**A second RL recipe worth knowing — DuplexPO (NTU/NTU/HKUST/NVIDIA,
https://arxiv.org/abs/2607.07148) [V PDF text].** Argues "when to speak"
should be learned separately from "what to say": sample *dynamics-critical
windows* (turn transitions, backchannels, barge-ins) from long human
dialogues (Fisher + "Seamless-Naturalistic-HQ"; 24.6k Fisher and 43.1k
Seamless windows), teacher-force the history, roll out only inside the
window, score with the Factorized Conversational Dynamics Reward:
R_on = exp(−τ²/2σ²) on onset error τ, R_bc = proximity of any produced
speech to the annotated backchannel interval, R_off = −clip((ℓ−ℓ*)/H, 0, 1)
for speech continuing ℓ after a barge-in, R_reg = −clip(Σβ_k·1[bad pattern
k]) — then GRPO with clipped group advantages (±5) and KL to the rollout
policy. Model: Qwen2.5-7B-Instruct + 600M Parakeet streaming encoder,
CosyVoice2 codec at 25 Hz with the LLM stepping at 12.5 Hz (2 audio tokens
per frame), 64 A800-80GB, built on the Nemotron-VoiceChat recipe [V].
Results (window-level, Fisher): onset MAE Moshi 1.99 s → DuplexPO 0.69 s;
turn-taking initiation 87.7% → 100%, yield 76.5% → 98.7%; backchannel
initiation 83.9% → 97.8%, yield 66.7% → 100%. Same on Seamless: Moshi
2.16 s / 80.1 / 69.0 / 76.2 / 71.3 → DuplexPO 1.03 s / 98.0 / 93.6 / 99.5 /
93.3. On FDB-v3: Moshi latency 0.44 s with Voiced-Interrupt-Rate 12.0% and
yield 71.9%; DuplexPO 0.24 s, VIR 5.0%, yield 100% [V]. The relevant
design lesson: reward on *windows* around events, not whole dialogues, and
factorise the reward per event type. A related paper, ASPIRin
(https://arxiv.org/abs/2604.10065), projects the action space into
speech-vs-silence before GRPO [I from DuplexPO's related work].

## 3. The 2025–26 field: what is public and how each model decides to speak

Two surveys give the taxonomy. Chen & Yu,
https://arxiv.org/abs/2509.14515 ("engineered synchronisation" — modular,
external scheduler — vs "learned synchronisation" — end-to-end; formalises
joint P(S_E, S_A) next-token-pair vs conditional P(S_A | S_E) objectives;
Moshi's inner monologue is the hierarchical case) [V PDF text]. The
second survey, https://arxiv.org/abs/2606.19453, defines an architectural
hierarchy **L0** external VAD/EoT scheduler, **L1** hidden-state predictor on
the LLM, **L2** token-level synchronisation in the sequence, **L3** shared
continuous latent (unrealised); a five-state machine {Idle, Listen, Speak,
Wait, Dual}; and a T×I×R ontology whose *intent* axis already includes
"Third-party" and whose six acid tests include "third-party rejection". It
states multi-party remains an open problem for every surveyed system [V
via fetch summary, I on exact wording].

Mechanisms, by model:

- **Moshi / PersonaPlex** — implicit; parallel streams, PAD/EPAD in the
  text stream (§1). PersonaPlex (NVIDIA,
  https://huggingface.co/nvidia/personaplex-7b-v1, paper
  https://arxiv.org/abs/2602.06053) is Moshi fine-tuned on Fisher (<10k h)
  with a *voice prompt* (audio tokens) and a *text persona prompt* prepended;
  reported 170 ms smooth-turn latency, 240 ms interruption response, 95%
  interruption success on FDB; NVIDIA Open Model License + CC-BY-4.0 [V
  card].
- **DuplexSLA** (StepFun + PKU/NTU/SJTU/UNSW/Imperial,
  https://arxiv.org/abs/2605.20755, code https://github.com/hyzhang24/DuplexSLA)
  — *explicit labels on a third channel*. Backbone "7B speech-LM,
  initialized from Step-Audio 2 mini" [V Table 1; the Step-Audio-2-mini
  card says 8B, Apache 2.0 — https://huggingface.co/stepfun-ai/Step-Audio-2-mini
  [V]; the 7B/8B discrepancy is the paper's rounding, I]. Clock: 160 ms
  chunks. Per chunk: user channel = 2 causal acoustic features at 80 ms
  stride; assistant channel = "TA4" = 1 text anchor + 4 audio tokens at
  40 ms; action channel = ≤10 text tokens carrying delayed transcript,
  planning text, the **three control labels `response` / `interrupt` /
  `backchannel`**, and JSON tool calls; silence is `<vad_silence>` /
  `<tts_pad>` anchors plus silence audio codes; chunk always ends with
  `<|action_end|>` so the clock never slips [V]. Serialization:
  `<|user_audio_begin|> U U <|user_audio_end|> <|assistant_audio_begin|> T A A A A <|assistant_audio_end|> ⟨action⟩ <|action_end|>` [V].
  Training: CPT ~500k h (320k h duplex dialogue, 2×90k h user/assistant-side
  ASR for alignment, 1.92M text samples) then post-training ~50k h (36k h
  interrupt/backchannel/pause, 14k h tool calls) with loss masks/weights on
  state tokens [V]. DuplexSLA-Bench: 2,100 cases; turn-taking subset
  accuracy/delay: normal 96.0% / 0.27 s, pause 93.33% / 0.27 s, interrupt
  99.33% / 0.40 s, backchannel 98.33% / 0.32 s; tool calling 85.56% at 0.64 s
  vs an ASR+LLM cascade 91.33% at 2.77 s [V]. **Release status: GitHub is
  MIT but the README says inference code, recipes and checkpoints are "not
  yet released"** as of 2026-09-12 [V]. Paper CC-BY 4.0.
- **Step-Audio 2 / 2.5** (https://arxiv.org/abs/2507.16632,
  https://arxiv.org/abs/2605.23463) — Step-Audio 2 is turn-based (no duplex
  claim) [V abstract]; 2.5 has ASR / TTS / *Realtime* branches trained with
  RLHF using a generative reward model, but the report gives **no**
  turn-taking mechanism, size, or duplex benchmark [V by absence in the HTML,
  A].
- **Qwen3-Omni** (https://arxiv.org/abs/2509.17765, Apache 2.0) — Thinker
  MoE 30B-A3B, Talker MoE 3B-A0.3B, AuT audio encoder 650M at 12.5 Hz, MTP
  80M, Code2Wav ConvNet 200M; 234 ms theoretical first-packet latency;
  input/output codec rates 12.5 Hz [V Table 1 of the report]. The word
  "duplex" does not occur in the report [V, grep of the PDF text]; the
  GitHub question on interruption handling
  (https://github.com/QwenLM/Qwen3-Omni/discussions/158) is unanswered [V].
  Qwen3.5-Omni (https://arxiv.org/abs/2604.15804) claims "semantic
  interruption through native turn-taking intent recognition" but is API-only
  and gives no mechanism [V]. Qwen3-Omni is therefore an *initialisation*
  with the right frame rate, not a duplex model.
- **DuplexOmni** (XJTU/PKU/Meituan, https://arxiv.org/abs/2606.09186) —
  shows how to duplex a Qwen3-Omni init: 480 ms time slices, explicit tokens
  `[THINK]`, `[CUT]` (stop assistant audio), `[WAIT]`, overlap-onset marker;
  3.02M synthetic conversations (~70% Chinese), speech from Qwen3-TTS
  tokenised with *Mimi*; FDB v1.5 72.6, latency 0.506 s; weights promised,
  CC-BY 4.0 [V via fetch].
- **SALMONN-omni** (ByteDance/Tsinghua, https://arxiv.org/abs/2505.17060) —
  Llama-3-8B-Instruct + LoRA r=32, Mamba streaming encoder, CosyVoice2-0.5B;
  80 ms blocks; per block 1 text token + 12 speech tokens (480 ms); explicit
  `<think>` and `<shift>` state tokens ("your LLM is secretly a full-duplex
  predictor"); barge-in-vs-backchannel F1 0.88 → 0.90 with RL; code
  https://github.com/bytedance/SALMONN, CC BY-SA 4.0 [V via fetch].
- **MiniCPM-o 4.5** (https://arxiv.org/abs/2604.27393) — 9B (Qwen3-8B
  backbone); "Omni-Flow": fixed windows, a **binary listen/speak control
  token predicted first** in each window; ablation over 1.0 / 0.2 / 0.1 s
  windows found 1.0 s best; open weights [V via fetch].
- **Fun-Audio-Chat** (https://arxiv.org/abs/2512.20156,
  https://huggingface.co/FunAudioLLM/Fun-Audio-Chat-8B) — parallel
  speech/text input streams; LLM core at 5 Hz, "Speech Refined Head" at
  25 Hz; duplex data synthesised by augmenting half-duplex dialogues; no
  explicit state tokens described [V via fetch].
- **Freeze-Omni / VITA-1.5** (https://github.com/VITA-MLLM/Freeze-Omni) —
  a classification head on the last LLM layer predicts interrupt/continue
  states; LLM cannot listen and speak simultaneously [V README summary].
- **Duplex S2S with next-token-pair prediction** (NVIDIA,
  https://arxiv.org/abs/2505.15670) — TinyLlama-1.1B, NanoCodec 12.5 Hz ×
  4 FSQ codebooks, jointly predicts user and agent channels; 94.5% barge-in
  success, 0.69 s; "first openly available duplex S2S model with training
  and inference code" [V via fetch].
- **User-stream routing study** (https://arxiv.org/abs/2605.10199,
  Qwen3-1.7B + LoRA r=16) — *channel fusion* (sum user embeddings into the
  stream, Moshi-style) beats *cross-attention routing* on QA (LLaMAQ 50.7 vs
  38.3) and takeover (1.000), but under overlap "CF-Duplex is more prone to
  producing semantically incoherent continued generation, whereas XA-Duplex
  largely avoids this failure mode" [V via fetch].
- **Data.** DuplexChat (https://arxiv.org/abs/2607.04941): 282,634 h English
  + 132,723 h Japanese of *speaker-separated* two-party podcast dialogue
  made by diarization + separation + restoration, CC-BY-4.0 — the largest
  open duplex corpus, but strictly two-speaker clips [V].

## 4. Benchmarks and metric definitions

**Full-Duplex-Bench v1** (https://arxiv.org/abs/2503.04721, code
https://github.com/DanielLin94144/Full-Duplex-Bench) [V HTML]. Four
static tasks: *pause handling* (Candor 216 + synthetic 137 items), *smooth
turn-taking* (Candor 119), *backchanneling* (ICC 55), *user interruption*
(200 synthetic, GPT-4o text + ChatTTS). Metrics: **TOR** (takeover rate) =
mean over items of a binary "model took the floor" (0 for silence or a
backchannel, 1 otherwise) — lower is better for pause handling, higher for
turn-taking/interruption; **response latency** = mean seconds from user
speech end to model onset; **backchannel frequency** = events/s while not
taking over; **JSD** of backchannel timing vs human in 200 ms windows;
**GPT-4o score** 0–5 for coherence/relevance. Headline v1 numbers: Moshi
pause TOR 0.985 with ~0.26 s latency (it interrupts almost every pause);
dGSLM 0.934; Freeze-Omni 0.642 at 0.953 s; Gemini Live 0.255 [V].

**FDB v1.5, overlap handling** (https://arxiv.org/abs/2507.23159, ICASSP
2026) [V HTML]. Four overlap scenarios: user interruption, user
backchannel, **talking to others** (side conversation), **background
speech**. Metrics: categorical behaviour {Respond, Resume, Uncertain,
Unknown}; stop latency (user onset → model stops); response latency;
prosodic adaptation. "Talking to others": Moshi Resume 0.19 / Respond 0.20
/ Unknown 0.59, stop latency 0.87 s; Freeze-Omni Resume 0.25 / Respond
0.58; Gemini Resume 0.99 / Respond 0.00; GPT-4o Resume 0.02 / Respond 0.91,
stop 0.18 s. "Background speech": Moshi Resume 0.07 / Respond 0.21 /
Unknown 0.71; GPT-4o Respond 0.93 [V]. These two cells are the only place
in the dyadic benchmark family where a *third voice* is tested — and no
model handles it: the two strategies are "respond to everything" or
"ignore everything".

**FDB v2** (https://arxiv.org/abs/2510.07838, ACL 2026) [V HTML]. Dynamic:
GPT-Realtime is the *examiner* that speaks, pursues staged goals and
interrupts; two pacings (Fast: examiner may interrupt/backchannel; Slow:
waits for end-of-turn); four families Daily / Correction / Entity Tracking /
Safety; Gemini 2.5 Flash scores per-event **turn-taking fluency** (1–5),
**instruction following** (1–5) and a global **task competence** (1–5);
transport is 48 kHz 16-bit mono PCM in 10 ms frames over WebRTC adapters;
human agreement r = 0.59–0.69. GPT-Realtime ≈ 4.0+, Moshi 2.7–3.8,
Freeze-Omni 2.6–3.5 [V].

**FDB v3** (https://arxiv.org/abs/2604.04847): tool use under real-world
disfluency; DuplexPO reports its metrics as **turn-taking latency**,
**Voiced Interrupt Rate** (share of agent onsets overlapping voiced user
speech, excluding internal pauses and the last 0.5 s of a user segment) and
**Yield Rate** (agent stops before the user resumes) [V DuplexPO text].

**Others.** FD-Bench (https://arxiv.org/abs/2507.19040): LLM+TTS simulated
user, 293 conversations, 1,200 interruptions, ~40 h; Moshi, Freeze-Omni,
VITA-1.5 [V abstract]. MTR-DuplexBench (https://arxiv.org/abs/2511.10262,
Findings ACL 2026): multi-round segmentation of continuous duplex dialogue;
conversation, quality, instruction following, safety [V abstract]. FLEXI
(https://arxiv.org/abs/2509.22243): six scenarios including
*model-initiated* emergency interruption; "significant gaps between open
source and commercial models in emergency awareness, turn terminating, and
interaction latency" [V abstract]. EchoChain
(https://arxiv.org/abs/2604.16456): state-update reasoning under
interruption [I]. A French duplex benchmark
(https://arxiv.org/abs/2609.10765) [I]. **HumDial, ICASSP 2026 challenge**
(https://arxiv.org/abs/2604.21406): 100+ h of dual-channel human-recorded
dialogue, eight scenarios (5 interruption, 4 rejection) in Chinese and
English, HumDial-FDBench protocol; includes *third-party speech* and
*speech directed at others* sub-scenarios; the overview states "performance
also degrades in multi-speaker environments, where overlapping speech from
third parties complicates speaker identification and leads to incorrect
decisions"; winner "Cookie ASR" 76.6 overall, Gemini 2.5 79.8 on
interruption but weak on rejection [V via fetch].

## 5. Multi-party (3+) spoken dialogue and turn-taking

Nothing found that trains a *speech-output* full-duplex LM on 3+-party
audio [A: searches on arXiv/HF for multi-party duplex / speech-to-speech,
2025–26]. What exists is turn-taking prediction and text-output agents.

**ModeratorLM (Amazon AGI)** — "Adaptive Turn-Taking for Real-time
Multi-Party Voice Agents",
https://cdn.amazon.science/e1/d2/f690424240e9bbb55b37d95b2b96/scipub-approval152129-45879557-adaptive-turntaking-for-realtime-multiparty-voice-agents.pdf
[V PDF text]. The closest thing to PLAN.md's stage 3 in the literature.
Qwen3-4B-Instruct-2507 (or -Thinking-2507 for the CoT variant) + an
in-house block-wise streaming speech encoder; **multi-channel audio is
downmixed to mono** before encoding; each chunk's audio embeddings are
appended to the LLM context *together with speaker-labelled transcripts
of that chunk*; per chunk the LLM emits either a **turn-taking control
token + text response** or an empty sequence (no turn); chunk length is
randomly 0.5–3 s in training. Data: ASR alignment on ~90k h public speech
(projection only), then "conversation pre-training" on **AMI + Fisher** with
the assistant role rotated over every non-initiating speaker (N−1 instances
per N-party conversation), then role-conditioning on RolePlayConv —
synthetic 3–6-speaker conversations from 125 roles written by Amazon Nova
Pro and rendered with TTS; LoRA, 13.4M trainable params. Evaluation on
**NOTSOFAR-1** (real meetings, ~4 speakers, ~6 min) with one speaker
designated "assistant": turn-taking precision / recall / F1 / false-positive
rate — **Moshi 0.14 / 0.10 / 0.11 / 0.66** (fails outright in a room:
constant false starts), multi-party baseline without role 0.58 / 0.33 /
0.38 / 0.05, ModeratorLM 0.77 / 0.51 / 0.57 / 0.01, ModeratorLM-Think 0.81 /
0.74 / 0.76 / 0.01. Two ablations matter for us: with **no transcripts** the
model "degrades substantially", with per-channel Kyutai-STT-2.6B hypotheses
(WER 6.7%) only slightly — the policy leans on who-said-what text; and
fixed chunking looks better only because it leaks utterance boundaries.
Output is text (a TTS is assumed downstream) [V].

**Triadic VAP** (https://arxiv.org/abs/2507.07518) [V HTML]. First VAP for
three parties: three per-speaker 16 kHz channels through a frozen CPC
encoder, a 1-layer self-attention per channel, then a 3-layer 64-d
cross-channel Transformer; predicts a 600 ms future window with 2 bins per
speaker → 2⁶ = 64 joint states. Data: TEIDAN, Japanese triads, 54
discussions from 18 triads, ~288 min (223 spontaneous + 64 "attentive
listening"), overlap 31.98% / 21.17%; **not released**. Next-speaker
accuracy: spontaneous 62.43% vs 55.56% baseline; attentive 87.50% vs
81.25%; combined 86.83% vs 77.58%.

**MuVAP (KTH, Qi & Skantze)** (https://arxiv.org/abs/2606.16731, code
https://github.com/Haotian-Qi/MuVAP) [V PDF text]. Monaural audio + one
camera, arbitrary N, causal. Key idea, **Role-Relative Projection**: instead
of the 2^(bins×N) joint VAP state, project onto {current floor-holder, next
floor-holder}, so the objective is independent of N. Per-speaker
attribution comes from active-speaker-detection face tracks. New AVCC
corpus: 30 h 52 min of unedited YouTube/Twitch conversations, 17 h 31 m
two-speaker + 13 h 21 m three-speaker; pre-training on Fisher (1,960 h) and
ASD sets (140 h). Numbers: on Fisher, stereo speaker-based VAP macro-F1
0.799 vs mono role-relative 0.778 (the cost of losing channels is ~2
points); AVCC silent shift/hold macro-F1 2-spk 0.696, 3-spk 0.670 (VAP
baseline 0.672 / 0.655); active 0.641 / 0.652; **next-speaker accuracy at
mutual silence 2-spk 0.637 (0.666 with GVAP conditioning, 0.702 with oracle
previous speaker), 3-spk 0.477 / 0.508 / 0.547 vs chance 0.5 / 0.333**;
previous-speaker inference 0.837 / 0.760. Turn-taking statistics they
report (median s): gaps Fisher 0.32, AVCC-3 0.43; pauses 0.40 vs 0.61;
overlaps 0.56 vs 0.50. Their reading: 3-party next-speaker is dominated by
*self-selection*, which is "very hard, if not impossible, to predict".

**LLMs on AMI turn-taking (NTT + CMU)** (https://arxiv.org/abs/2606.17542)
[V PDF text]. AMI (100 h, K = 4); three tasks — addressee detection (acc),
turn-change prediction (acc), next-speaker prediction (F1) — with
utterance-level text context c_i = (s_<i, t_<i). Full set: naive 47.6 /
63.7 / 25.0; SVM 56.4 / 66.5 / 40.1; Qwen3-8B 51.8 / 64.6 / 50.0;
Qwen3-14B 52.3 / 66.4 / 51.1; Qwen3-32B 47.3 / 65.3 / 49.7; Qwen2.5-Omni-7B
50.0 / 61.6 / 37.0; **Qwen3-Omni-30B 32.1 / 41.6 / 13.7** (raw audio-video
hurt badly); Gemini 2.5 Pro 55.7 / 68.3 / 47.7. Human-eval subset: humans
66.6 acc addressee, 75.0 turn-change, **60.1 F1 next speaker**; Qwen3-14B
69.4 F1 next speaker (significantly above humans), 51.5 addressee
(significantly below). Ablation: conversational context is the decisive
input for next-speaker. Related: ICNLSP 2025 next-speaker with LLMs on
AMI/ICSI (https://aclanthology.org/2025.icnlsp-1.7/) and an addressee
benchmark (https://arxiv.org/abs/2501.16643) [I].

**Engineered multi-party agents.** Furhat social robot
(https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1766383/full):
ReSpeaker DOA + face tracking + Azure STT + GPT-3.5 choosing an addressee;
gaze-based turn policy (take the turn if the last speaker looks at you, or
after long silence); addressee accuracy 92.6% in parallel dyads but 79.3%
in a group; voice-ID only 18–27%; 1.35 s latency [V via fetch]. RESPOND
(Microsoft, https://arxiv.org/abs/2603.21682): streaming ASR + incremental
semantics with "backchannel intensity" and "turn-claim aggressiveness"
dials [V abstract]. S-MARC (Berkeley, https://arxiv.org/abs/2602.11065):
streaming hierarchical behaviour/intent reasoning over duplex dialogue,
dyadic, macro-F1 0.535 synthetic / 0.578 Candor subset [V PDF text].

---

## Implications for PLAN.md

1. **§2's "80 ms coincidence" is real and the delay values are now
   pinned.** Use Moshi's post-training configuration — acoustic delay 1
   step, text delay 0 — as the starting point, and keep the per-codebook
   depth-transformer weights (they are per *index*, so a K-speaker mixture
   stream costs no extra depth-transformer parameters). What Moshi does
   *not* give us: any K>2 loader — moshi-finetune is stereo with
   left=Moshi/right=user — so the stage-3 data pipeline is ours to write;
   the memory numbers (39.6 GB on one H100 at 16×100 s) are the right
   budget anchor for a 7B LoRA run.

2. **The floor token has strong precedent, and the best precedent is a
   separate channel.** The field split: implicit (Moshi, PersonaPlex, NTPP)
   vs explicit (DuplexSLA `response/interrupt/backchannel` on a rate-limited
   action channel; SALMONN-omni `<shift>`; MiniCPM-o binary listen/speak
   per window; DuplexOmni `[CUT]/[WAIT]`). Every explicit system that
   reports numbers beats Moshi on pause TOR / VIR, and ModeratorLM shows
   Moshi's implicit policy collapses in a room (F1 0.11, FP 0.66). PLAN's
   {listen, backchannel, speak, yield} maps onto DuplexSLA's three labels
   plus yield; the recommendation is to give it DuplexSLA's *shape* — a
   text action channel on the assistant timeline, capped per frame — because
   that channel can also carry the speaker-attributed monologue tags
   `<spk:NAME>` and, later, tool calls, without touching the audio streams.
   Keep Moshi's silence anchors (PAD/EPAD or `<vad_silence>`) as well: the
   RL recipes below act on the text stream.

3. **Stage 4 RL is cheap and the two recipes are directly portable, but the
   room changes the reward definitions.** Kyutai's run is 32 H100 × 100
   epochs × 512 generations at LR 2e-7, rule rewards per axis plus an
   LLM-judge term; DuplexPO's factorised window reward (onset Gaussian,
   backchannel proximity, yield penalty, participation regulariser) with
   GRPO is the cleaner formulation. The room version needs one new mask:
   *addressed-to-agent*. Pause/turn/backchannel rewards only make sense on
   turns directed at the model, and the "regularised participation" term
   becomes the main term ("don't take the floor when two humans are
   talking to each other"). AMI carries addressee labels (used by the NTT
   paper), so the stage-4 windows can be mined from real meetings rather
   than only the §4.3 human slice. Budget 10k H100-h is generous by these
   anchors.

4. **Licences constrain which data goes into which stage.** Seamless
   Interaction is CC-BY-NC 4.0 and taints derived checkpoints (Kyutai's are
   NC for exactly this reason); Fisher is LDC. For open stage-3 pretraining
   of the own-audio stream, DuplexChat (415k h, CC-BY-4.0, speaker-separated
   podcasts) is the obvious large corpus, with the caveat that it is
   two-speaker clips — useful for teaching the model to *speak* in duplex,
   not for the K-speaker floor policy.

5. **Evaluation (§6) should add three externally comparable numbers.**
   (a) NOTSOFAR-1 turn-taking P/R/F1/FP in ModeratorLM's protocol — the
   only published multi-party duplex number; the bar is F1 0.76 / FP 0.01
   and Moshi is at 0.11 / 0.66. (b) AMI next-speaker F1 and addressee
   accuracy as a probe on the inner monologue — humans 60.1 / 66.6, text
   LLM 69.4 / 51.5 — which tests whether the speaker-attributed monologue
   carries the conversational state. (c) FDB v1.5 "talking to others" and
   HumDial "third-party speech" cells, where every current model either
   answers everything or ignores everything; a diarization-conditioned
   model should be the first to split those. Adopt FDB-v3's VIR / yield /
   latency and DuplexPO's onset-MAE as the stage-3/4 floor metrics rather
   than inventing a floor-F1.

6. **What the multi-party turn-taking literature says the floor token
   should predict.** Next-speaker at three parties is ~0.5 accuracy even
   with oracle information (MuVAP 0.547; humans 60 F1 on AMI) because
   self-selection is not predictable; shift/hold is barely harder at 3 than
   at 2 (0.670 vs 0.696). So the floor token should be "should *I* take the
   floor now", not "who speaks next" — which is what PLAN §2 already does,
   and MuVAP's role-relative projection (current vs next floor-holder,
   independent of K) is a good target parameterisation for the auxiliary
   head if we want one. ModeratorLM's ablation — no transcripts degrades
   substantially, ASR hypotheses barely — is direct evidence for the
   speaker-attributed monologue being load-bearing for the policy, and its
   role prompt cutting false positives (FP 0.05 → 0.01) argues for adding a
   persona/role prompt (PersonaPlex-style text prompt) to the enrollment
   prompt in §2.

7. **Backbone-init ablation (§5 stage 0).** Moshi: 7B, CC-BY-4.0, Mimi at
   12.5 Hz, already duplex, but its user stream is single-speaker and its
   floor policy must be re-learned. Qwen3-Omni: Apache 2.0, AuT at 12.5 Hz,
   stronger LM, zero duplex machinery; DuplexOmni demonstrates it can be
   duplexed with Mimi tokens and 480 ms slices. Step-Audio-2-mini (8B,
   Apache 2.0) is DuplexSLA's init, but DuplexSLA's own weights are
   unreleased. The routing study is the one architectural warning: summing
   the room stream into the sequence (what PLAN does with mixture tokens +
   additive STNO) is better for semantics but fragile under overlap, while
   a cross-attention listening path is more robust — which is the argument
   for keeping the Dixtral/DiCoW encoder as the second path rather than
   dropping it after the ablation.

8. **Scale sanity.** DuplexSLA needed ~500k h of CPT to make a 7B model
   fluent in a *new* serialization and 36k h just for interrupt/backchannel/
   pause behaviour; Moshi used 7M h + 2k h Fisher + 20k h synthetic. PLAN's
   stage-3 budget (30k H100-h) is plausible only because it starts from a
   duplex init and keeps Moshi's serialization; changing the token layout
   (e.g. to TA4 at 40 ms) would put us on DuplexSLA's cost curve.
