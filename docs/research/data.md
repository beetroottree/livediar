# Data inventory for the room-aware full-duplex LM (PLAN.md §4)

Researched 2026-09-12. Every row is what the public page or paper says
today, with the URL it came from; "commercial" means the licence as written
permits it without a separate agreement. Hours are audio hours unless noted.
Overlap figures are the corpus authors' own where they publish one.

Reading key for the tables: **chan** = what audio channels exist
(IHM = per-speaker headset, SDM = single distant mic, MC = array);
**labels** = what supervision ships with the audio.

## 1. Multi-party corpora (§4.1 real audio)

| Corpus | Hours / sessions | Speakers | Lang | Overlap | Chan | Labels | Licence, access | Commercial |
|---|---|---|---|---|---|---|---|---|
| [AMI](https://groups.inf.ed.ac.uk/ami/corpus/) | 100 h, ~170 meetings | 4–5 / meeting | en | 2-spk overlap 22.1% train / 21.0% test ([src](https://arxiv.org/pdf/2510.03630)) | IHM + SDM + MC arrays + video | Manual transcripts with word timings, dialog acts (incl. backchannel class), heads/gaze | CC BY 4.0, direct download; HF mirror [edinburghcstr/ami](https://huggingface.co/datasets/edinburghcstr/ami) | Yes |
| [ICSI](https://groups.inf.ed.ac.uk/ami/icsi/) | ~72 h speech, 75 meetings | 3–10 / meeting (53 unique) | en | 2-spk overlap 9.0% train / 13.6% test ([src](https://arxiv.org/pdf/2510.03630)) | IHM + distant table mics | Manual transcripts, MRDA dialog acts (backchannel tag) | CC BY 4.0 from the AMI site (free); also LDC [LDC2004S02](https://catalog.ldc.upenn.edu/LDC2004S02) | Yes (Edinburgh copy) |
| [CHiME-6](https://www.chimechallenge.org/datasets/chime6) | 40:05 train / 4:27 dev / 5:12 eval, 20 dinner parties | 4 / session | en | CHiME-5 reported 22.9% overlap ([src](https://arxiv.org/pdf/2307.13012)) | Binaural per speaker + 6 Kinect 4-mic arrays | Manual transcripts, utterance timestamps (re-synced in CHiME-6) | CC BY-SA 4.0 since 2024-01-01 ([openslr 150](http://openslr.org/150/)); `chime-utils` fetches it | Yes |
| [DiPCo](https://zenodo.org/records/8122551) | ~5.3 h, 10 sessions of 15–45 min | 4 / session (32 total) | en | dinner-party style, not summarised | Close-talk per speaker + 5×7-mic arrays | Manual transcripts, timestamps | CDLA-Permissive 1.0, Zenodo direct | Yes |
| [Mixer 6](https://www.chimechallenge.org/challenges/chime8/task1/data) (CHiME-7/8) | 36 h calls + 27 h interviews used in DASR | 2 / session | en | low (interview) | 14 mics incl. distant | Transcripts | LDC2013S03; CHiME PDF request form to LDC | No without LDC for-profit membership |
| [NOTSOFAR-1](https://huggingface.co/datasets/microsoft/NOTSOFAR) | 315 meetings × ~6 min: 174.5 h SC (5 devices) + 126 h MC; splits 110/35/170 mtgs | 4–8 / meeting, 35 unique | en | "extra overlapping speech" flagged per meeting; no single % ([paper](https://arxiv.org/html/2401.08887v1)) | Close-talk + 5 SC devices + 4×7-mic arrays | Double-annotated human transcripts from close-talk; word timestamps via aligner | CC BY 4.0; HF (token) or AzCopy via [NOTSOFAR1-Challenge](https://github.com/microsoft/NOTSOFAR1-Challenge). Plus 1,000 h simulated train set (LibriVox + 15k real ATFs) | Yes |
| [AliMeeting](http://openslr.org/119/) | 118.75 h (104.75/4/10), 240 sessions of 15–30 min | 2–4 / session | zh | 42.27% train / 34.76% eval ([src](https://arxiv.org/pdf/2110.07393)) | IHM + 8-ch array | Manual per-speaker transcripts with timestamps | CC BY-SA 4.0, openslr direct | Yes (share-alike) |
| [AISHELL-4](https://www.openslr.org/111/) | 120 h, 211 sessions | 4–8 / session | zh | 19.04% train / 9.31% eval ([src](https://arxiv.org/pdf/2104.03603)) | 8-ch circular array; headsets used for transcription but not all released | Transcripts + speaker VAD | CC BY-SA 4.0, openslr direct | Yes (share-alike) |
| [MagicData-RAMC](https://www.openslr.org/123/) | 180 h, 351 dialogues | 2 / call, 663 total | zh | phone dialogue, moderate | Per-speaker phone channels | Manual transcripts, topics, speaker meta | CC BY-NC-ND 4.0 | No |
| [Fisher English](https://catalog.ldc.upenn.edu/LDC2004S13) (P1 + [P2](https://catalog.ldc.upenn.edu/LDC2005S13)) | ~1,960 h, 11,699 calls ≤10 min | 2 / call, thousands unique | en (Spanish set separate) | ~10% simultaneous speech typical of CTS | 2 channels, one per side, 8 kHz | Manual transcripts with segment times (word alignments available from later work) | LDC non-member research licence or membership; fee | No (needs LDC for-profit membership) |
| [CALLHOME AmEng](https://catalog.ldc.upenn.edu/LDC97S42) | ~56 h, 120 calls | 2+ / call | en (+ 5 other CALLHOME langs) | intimate, high | 2 channels | Transcripts [LDC97T14](https://catalog.ldc.upenn.edu/LDC97T14) | LDC, corpus-specific agreement | No |
| [Switchboard-1 R2](https://catalog.ldc.upenn.edu/LDC97S62) | ~260 h, 2,400 calls | 543 | en | CTS | 2 channels | Transcripts; SWBD-DAMSL acts on 1,155 calls | LDC | No |
| [VoxConverse](https://github.com/joonson/voxconverse) | dev 20.3 h / 216 clips, test 53.5 h / 310 clips | up to ~20 / clip | en | present, unquantified in summary | Single YouTube mix | RTTM only, no transcripts | CC BY 4.0; audio must be pulled from YouTube (site no longer hosts wavs) | Labels yes; audio is YouTube ToS |
| [DIHARD III](https://catalog.ldc.upenn.edu/LDC2022S12) | ~34 h dev + ~33 h eval, 11 domains | varies | en, zh | domain-dependent, up to very high | Single | RTTM + UEM, no transcripts | LDC licence via challenge registration | No |
| [MSDWild](https://github.com/X-LANCE/MSDWILD) | 80 h, 3,000+ clips | few per clip | multilingual | frequent turn-taking | Single + video | RTTM, face boxes | Research-only agreement PDF, Google Drive | No |
| [M3SD](https://huggingface.co/datasets/Igor97/MISP-M3SD) | 770+ h, 1,372 clips | varies | 16 langs | meetings/debates/home | Single + video by ID | RTTM (semi-auto, partially verified) | Apache 2.0 on HF, gated agreement; video via script | Labels yes; video source ToS |
| [Ego4D AV](https://ego4d-data.org/docs/benchmarks/av-diarization/) | ~50 h annotated of 750 h conversational | wearer + others | en (transcription) | in-the-wild | Egocentric single | Speaker segments, transcripts, gaze/looking | Ego4D licence, 48 h approval; data may not appear in any product | No |
| [Seamless Interaction](https://huggingface.co/datasets/facebook/seamless-interaction) | 4,000+ h, 65k interactions, 4,000+ participants | 2 / session | en | dyadic, naturalistic + 1,300 h actors | Per-participant denoised channels, 48 kHz + video | Time-aligned transcripts (JSONL), 5k emotion/behaviour annotations | CC BY-NC 4.0, HF open | No |
| [CANDOR](https://betterup.com/research/candor-research) | 850+ h, 1,656 video chats | 2 / call | en | back-channels ~1,000/h | Per-participant recordings | Transcripts, turn segmentation, surveys | CC BY-NC 4.0, request form | No |
| [DuplexChat](https://arxiv.org/html/2607.04941v1) | 282,634 h en + 132,723 h ja after filtering | 2 / clip | en, ja | simultaneous speech 10% en / 21% ja; overlapping transitions 48–50%; backchannels 3.1/min en | Separated per-speaker streams (DialogueSidon) | Pyannote community-1 diarization + Parakeet ASR; only URLs/segments released | CC BY 4.0 metadata + code ([repo](https://github.com/sarulab-speech/DuplexChat)); audio re-fetched from podcast feeds | Metadata yes; audio depends on each feed |
| [SPoRC](https://huggingface.co/datasets/blitt/SPoRC) | 1.1M episodes (May–Jun 2020); speaker turns for 370k | many | en | podcast | Audio not redistributed, RSS URLs | Whisper transcripts, pyannote turns, audio features | Non-commercial | No |
| [Spotify Podcast Dataset](https://podcastsdataset.byspotify.com/) | 100k episodes, ~47k h | many | en (+pt) | podcast | Single | ASR transcripts | Discontinued; no new access | No |
| [IMDA NSC](https://www.imda.gov.sg/how-we-can-help/national-speech-corpus) Parts 3–6 | P3 900 h conv, P4 900 h code-switch (Singlish ↔ zh/ms/ta), P5 1,500 h themed, P6 1,300 h simulated calls; ~10,600 h total with prompted parts | 2 / conv, ~1,000+ | en-SG, zh/ms/ta CS | conversational, separate close mics per speaker | Per-speaker close-talk (+ boundary mic in same-room setup) | Manual transcripts, speaker meta | Singapore Open Data Licence, register on site ([paper](https://www.isca-archive.org/interspeech_2019/koh19_interspeech.pdf)) | Yes |

Other Mandarin meeting/dialogue material worth knowing: WenetSpeech
`Test_Meeting` (15 h far-field from 197 real meetings, CC BY 4.0, test-only
by convention), and the M2MeT challenge which is AliMeeting plus
AISHELL-4 ([summary](https://arxiv.org/pdf/2202.03647)).

## 2. Single-speaker pools with transcripts (§4.2 simulation sources)

| Corpus | Hours | Speakers | Lang | Labels | Licence | Commercial |
|---|---|---|---|---|---|---|
| [LibriSpeech](http://www.openslr.org/12/) | 960 h + 5 h dev/test, 16 kHz | 2,484 | en | Transcripts; word alignments via published MFA/NeMo manifests | CC BY 4.0 | Yes |
| [LibriTTS-R](https://google.github.io/df-conformer/librittsr/) | 585 h, 24 kHz, restored | 2,456 | en | Transcripts with punctuation, sentence-level | CC BY 4.0 ([openslr 141](http://www.openslr.org/141/)) | Yes |
| [Common Voice 26 (scripted)](https://github.com/common-voice/cv-dataset) | 42,388 h total, 28,893 h validated, 26+ langs; plus Spontaneous Speech v4 (4 langs) | hundreds of thousands | multi | Sentence text, demographics; no timing | CC0 | Yes |
| [Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset) | 101k h (zh/en/ja/ko/de/fr); Emilia-YODAS +114k h → Emilia-Large 216k h | in-the-wild | 6 | Whisper transcripts, DNSMOS-filtered, per-utterance | Emilia CC BY-NC 4.0; **Emilia-YODAS CC BY 4.0** | Only the YODAS half |
| [GigaSpeech](https://huggingface.co/datasets/speechcolab/gigaspeech) | 10k h supervised (podcast/YouTube/audiobook), 40k h total | many | en | Transcripts, segment times | Apache 2.0 (agreement on HF) | Yes |
| [MLS](https://www.openslr.org/94/) | 50k h; 44.5k h en, 6k h de/nl/fr/es/it/pt/pl | ~6k | 8 | Transcripts | CC BY 4.0 | Yes |
| [YODAS2](https://huggingface.co/datasets/espnet/yodas2) | ~422k h long-form, 149 langs, 24 kHz | YouTube | multi | Uploader captions (manual/auto subsets) | CC BY 3.0 videos; also [yodas-granary](https://huggingface.co/datasets/espnet/yodas-granary) cleaned | Yes |
| [WenetSpeech](http://www.openslr.org/121/) | 10k+ h labelled, 22k h total | many | zh | OCR/ASR-verified transcripts with confidence | CC BY 4.0 | Yes |
| [AISHELL-1](https://www.openslr.org/33/) | 178 h | 400 | zh | Transcripts | Apache 2.0 | Yes |
| [AISHELL-3](https://www.openslr.org/93/) | 85 h, 44.1 kHz | 218 | zh | Transcripts + pinyin | Apache 2.0 | Yes |
| [Granary](https://huggingface.co/datasets/nvidia/Granary) | 643k h pseudo-labelled ASR, 25 EU langs | CC pools | multi | Whisper-v3 two-pass, filtered | CC BY 4.0 sources | Yes |

## 3. Simulation tooling (§4.2)

| Tool / set | What it gives | Key parameters | Licence |
|---|---|---|---|
| [NeMo speech_data_simulator](https://github.com/NVIDIA-NeMo/Speech/tree/main/tools/speech_data_simulator) | Multi-speaker sessions from single-speaker manifests with word alignments; RTTM/CTM/JSON outputs; optional multichannel | `num_speakers`, `session_length`, `turn_prob`, `overlap_prob`, `mean_overlap`, `mean_silence`, `dominance_var`, sentence-length neg-binomial, `add_bg`, `use_rir` (gpuRIR or pyroomacoustics), enrollment/speaker-sampling knobs ([tutorial](https://github.com/NVIDIA-NeMo/NeMo/blob/main/tutorials/tools/Multispeaker_Simulator.ipynb)) | Apache 2.0 |
| [LibriMix](https://github.com/JorisCos/LibriMix) / [SparseLibriMix](https://github.com/popcornell/SparseLibriMix) | 2/3-spk mixtures from LibriSpeech + WHAM! noise; Sparse variant adds controllable overlap 0–100% | SNR, overlap ratio, mode (min/max) | MIT scripts; WHAM! noise CC BY-NC 4.0 |
| [LibriCSS](https://github.com/chenzhuo1011/libri_css) | 10 h real re-recorded LibriSpeech, 7-mic array, 6 sessions × 10-min mini-sessions | Overlap 0S/0L/10/20/30/40% | MIT + CC BY 4.0 |
| [pyroomacoustics](https://github.com/LCAV/pyroomacoustics) | ISM + ray-tracing RIRs, arbitrary rooms/arrays | room dims, absorption/materials, order, mic geometry | MIT |
| [gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR) | Fast ISM on GPU (NeMo's other backend) | as above | AGPL-3.0 (check before shipping) |
| [BUT ReverbDB](https://speech.fit.vut.cz/software/but-speech-fit-reverb-database) | 279 real RIRs: 9 rooms × 31 mics, with noise and LibriSpeech retransmissions | measured positions in metafiles | CC BY 4.0 |
| [openSLR 28 RIRS_NOISES](https://www.openslr.org/28/) | 60k simulated RIRs (small/medium/large rooms) + real RIRs (RWCP, REVERB, AIR) + isotropic/point noises | Kaldi-style lists | Apache 2.0 (simulated); real parts inherit source licences |
| [openSLR 26](https://www.openslr.org/26/) | Simulated RIRs only (subset of 28) | — | Apache 2.0 |
| [MUSAN](https://www.openslr.org/17/) | 6 h noise, 42 h music, 60 h speech | — | CC BY 4.0 / public domain parts |
| [WHAM! noise](http://wham.whisper.ai/) | ~30 h urban ambient, 2-ch | — | CC BY-NC 4.0 |
| [DNS Challenge](https://github.com/microsoft/DNS-Challenge) | 65k+ noise clips (Audioset/Freesound/DEMAND) + RIRs | — | CC BY 4.0 repo; clip-level source licences |
| NOTSOFAR simulated set | 1,000 h LibriVox + 15k real conference-room ATFs, early/late split | — | CC BY 4.0 |

## 4. Synthetic dialogue (§4.3)

**Moshi's recipe** ([paper](https://kyutai.org/Moshi.pdf)): post-train on
diarization-simulated multistream from ~7M h monologue pretraining, then
fine-tune on Fisher (2,000 h, upsampled 8→24 kHz with AudioSR) for duplex
behaviour, then instruct on **20k+ h synthetic dialogue**: Helium fine-tuned
on OpenHermes + real-conversation transcripts writes user/assistant scripts;
a multi-stream streaming TTS renders both sides, the assistant conditioned
on one actor's voice (70+ styles) and users on random voices, with
randomised timing so streams overlap. Kyutai later opened a text-to-dialogue
path via the 2025 Kyutai TTS/Unmute stack.

| Model | Params, langs | Multi-speaker script | Nonverbal / backchannel tags | Weights licence | Commercial |
|---|---|---|---|---|---|
| [Voxtral TTS](https://mistral.ai/news/voxtral-tts/) (Mar 2026) | 4B, 9 langs, 3-s cloning, 70 ms TTFA | Single speaker per call | Reproduces fillers/ums from reference; no explicit tags | CC BY-NC 4.0 | No (API only) |
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | 82M, 8 langs, 54 voices | No | None | Apache 2.0 | Yes |
| [Fun-CosyVoice3-0.5B](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) (Dec 2025) | 0.5B, 9 langs + 18 zh dialects | Via inline speaker tags, one voice per reference | `[laughter]`, `[breath]`, `<strong>` emphasis, instruct-style vocal bursts | Apache 2.0 | Yes |
| [Fish-Speech / OpenAudio S1-mini](https://huggingface.co/fishaudio/openaudio-s1-mini) | 0.5B (S1 4B API), 13 langs | No native multi-spk | Rich: `(laughing) (chuckling) (sighing) (crowd laughing)` etc. | CC BY-NC-SA 4.0 | No |
| [Orpheus 3B](https://github.com/canopyai/Orpheus-TTS) | 3B Llama-based, en (+multilingual ft) | No | `<laugh> <chuckle> <sigh> <cough> <sniffle> <groan> <yawn> <gasp>` | Apache 2.0 | Yes |
| [Dia 1.6B](https://github.com/nari-labs/dia) / [Dia2 1B/2B](https://github.com/nari-labs/dia2) | en only; Dia2 streaming, ~2 min per output | `[S1]`/`[S2]` in one pass | `(laughs) (coughs) (sighs) (whispers)` | Apache 2.0 | Yes |
| [Chatterbox / Chatterbox Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) | 0.5B; Multilingual 23 langs; Turbo 350M | No | Turbo: `[laugh] [chuckle] [cough]` etc. | MIT | Yes |
| [VibeVoice-1.5B / Large](https://github.com/microsoft/VibeVoice) | 1.5B/7B, en + zh | Up to 4 speakers, ~90 min single pass | No explicit tags; spontaneous prosody only | MIT (research release; repo was pulled and restored in 2025) | Yes |
| [MOSS-TTSD v1.0](https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v1.0) (Mar 2026) | 8B, 20 langs | 1–5 speakers, up to 60 min | Dialogue-native prosody; tag support not documented | Apache 2.0 | Yes |
| [Higgs Audio v2](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base) | 3B | Multi-speaker dialogue | Emergent laughter/hums | Boson community licence (Llama-3-style, 100k-MAU cap) | Conditional |
| [Breeze TTS 2](https://github.com/breezeblue-ai/breeze-tts) (Aug 2026) | 3B, 50 langs, tops open-weights arena | Voice design | Expressive | Weights research/non-commercial; code Apache | No |

Practical read: for §4.3 the assistant stream wants one fixed voice with
laughter/backchannel tokens (Orpheus, CosyVoice 3, Chatterbox Turbo all do
this under Apache/MIT); the human streams want *many* voices with
per-speaker separate audio so the room simulator can place them, which
argues for rendering each line independently with a zero-shot cloner
(CosyVoice 3, Chatterbox, Dia2) rather than a single-pass dialogue model
like VibeVoice/MOSS-TTSD whose output is already mixed. Keep MOSS-TTSD or
VibeVoice for a "pre-mixed" ablation only.

## 5. Pseudo-labelling pipelines and reported quality (§4.1)

| Pipeline | Stages | Scale | Reported quality |
|---|---|---|---|
| [DuplexChat-Pipe](https://arxiv.org/html/2607.04941v1) (2026) | Feed filtering → pyannote community-1 diarization → 2-speaker clip extraction at ≥5 s silence gaps → DialogueSidon diffusion separation+restoration → Parakeet ASR | 415k h en+ja | DNSMOS 2.92 en / 3.11 ja; SQ-PESQ 3.23; turn exchanges 6.2/min en, backchannels 3.1/min; no WER reported (ASR used for counting) |
| [Emilia-Pipe](https://arxiv.org/abs/2501.15907) | Standardise → UVR source separation → pyannote diarization → VAD segmentation → Whisper ASR → DNSMOS/length filtering | 216k h | DNSMOS P.808 ≈ 3.26 avg; TTS trained on it matches MLS-trained on WER/SIM; ~1 h raw → minutes of processing |
| [Granary](https://arxiv.org/pdf/2505.13404) (NVIDIA, 2025) | Silero VAD + forced alignment → two-pass Whisper-large-v3 with LID check → hallucination/char-rate/charset filters → EuroLLM translation + QE | 1M h in → 643k h ASR | 14k h Granary-en ≈ 23.5k h MOSEL-en on FLEURS PnC (~10% rel. WER better) |
| [Multi-ASR fusion + SpeechLLM correction](https://arxiv.org/abs/2506.11089) (Interspeech 2025) | N heterogeneous ASRs → LLM/SpeechLLM arbitrates instead of ROVER | — | Pseudo-label WER below single-ASR and voting baselines; downstream semi-supervised ASR improves |
| [LLM diarization correction](https://arxiv.org/pdf/2406.04927v3) | ASR + diarization → fine-tuned Qwen2.5-7B / GPT reassigns speaker labels from text | — | Reduces speaker-attribution errors; zero-shot GPT weaker than fine-tuned 7B |
| [DiCoW](https://arxiv.org/pdf/2501.00114) / [SE-DiCoW](https://arxiv.org/html/2601.19194v1) | Diarization-conditioned Whisper encoder (FDDT + QK-bias) for TS-ASR | — | NOTSOFAR-1 eval-small tcpWER 19.7 with oracle diarization, 33.5 with DiariZen; ~11% abs ORC-WER gain over cascades; SE-DiCoW −52% rel tcpWER on EMMA |
| [SoulX-Transcriber](https://arxiv.org/pdf/2606.02400) / [SpeakerLM](https://arxiv.org/pdf/2508.06372) (2025–26) | End-to-end multi-speaker transcription LMs; SoulX's training data comes from a cascaded VAD → multi-ASR consensus (with confidence) → speaker clustering pipeline | — | Consensus confidence used to gate labels, matching the "mask low-confidence frames" idea in §4.1 |
| [Grounding spoken LLMs via diarization conditioning](https://arxiv.org/html/2606.18134) (2026) | Feeds diarization masks to a spoken LLM the way §2 proposes | — | Directly relevant precedent for K-stream conditioning |

Take-away for §4.1: every recent large pipeline uses pyannote (or NeMo
Sortformer) + a strong single-speaker ASR + a separation/restoration model,
and only DuplexChat keeps *separated* per-speaker streams — which is what a
multi-stream model needs. None of them publish label WER against human
meetings; the honest quality number we own is our own AMI tcpWER, so keep
reporting it.

## 6. Turn-taking / behaviour labels for RL preference data (§4.3)

| Resource | What is labelled | Size | Licence | Use |
|---|---|---|---|---|
| [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) v1/1.5/[v2](https://arxiv.org/pdf/2510.07838)/[v3](https://arxiv.org/abs/2503.04721) | Pause handling, backchannel, turn-taking, user interruption; v1.5 adds overlap; v3 real disfluent human speech + tool use (May 2026) | Hundreds of prompts | Open (GitHub) | Eval; its scoring rules are a reward template |
| [HumDial-FDBench](https://github.com/ASLP-lab/HumDial-FDBench) (ICASSP 2026) | Dual-channel actor-performed dialogues with scripted interruption types (follow-up, negation, repeat, topic switch, silence) and rejection cases (backchannel, third-party speech, speech to others) | Released dataset + leaderboard | Open | Closest public analogue to "assistant in the room, not addressed" |
| [MTR-DuplexBench](https://arxiv.org/pdf/2511.10262), [M3-DuplexBench](https://arxiv.org/html/2607.29125), [EchoChain](https://arxiv.org/pdf/2604.16456) | Multi-round, multilingual, and state-update-under-interruption evaluations | — | Open | Eval |
| [Multi-Faceted Interactivity Alignment](https://arxiv.org/html/2606.11167v1) (2026) | VAD-derived rule rewards from **Fisher + Seamless Interaction**: pause (−1 if >1 s speech), turn-take (−delay), backchannel F1 ±1 s, interruption (−delay), + Qwen3-235B judge; GRPO | Automatic | Corpora as above | Recipe for §4.3 rewards without human raters |
| [DuplexPO](https://arxiv.org/html/2607.07148v1) (2026) | Factorised reward (onset Gaussian, backchannel window, yield penalty, pattern reg.) on 24.6k Fisher + 43.1k Seamless windows | ~68k windows | Not released | Same; shows localized RL on "dynamics-critical windows" beats DPO |
| [Dual-Axis Generative Reward Model](https://arxiv.org/html/2604.14920) (2026) | Human-annotated interaction failures: 100 Seamless + 289 human–machine samples (~10 h) + 146 h synthetic; reward model 86.8% acc on human–human | ~7k samples | [repo](https://github.com/MM-Speech/DualAxisRM) | Off-the-shelf judge for "welcome / late / rude" |
| [CANDOR](https://betterup.com/research/candor-research) | Turn segmentation, backchannel counts, post-conversation ratings (enjoyment, etc.) per participant | 850 h | CC BY-NC | Non-commercial preference signal |
| [Switchboard SWBD-DAMSL](https://catalog.ldc.upenn.edu/LDC97S62) / [AMI dialog acts](https://groups.inf.ed.ac.uk/ami/corpus/) / [ICSI MRDA](https://groups.inf.ed.ac.uk/ami/icsi/) | Backchannel, floor-grabber, floor-holder, interruption acts with timestamps | 1,155 calls / 100 h / 72 h | LDC / CC BY / CC BY | Fit the §4.2 Markov turn model from AMI+ICSI (both commercial-OK) |
| [Seamless Interaction](https://huggingface.co/datasets/facebook/seamless-interaction) | Per-participant channels + 5k self-reported internal-state annotations | 4,000 h | CC BY-NC | Research-only reward mining |

No public set carries the exact "interjection welcome / late / rude" label
§4.3 wants; the 2026 papers all synthesise it from VAD timing on Fisher or
Seamless, and only Dual-Axis paid humans (~10 h). The 500 h human-rated
slice in the plan would be the largest such set by an order of magnitude.

## Implications for PLAN.md §4

**Download first (commercial-clean, direct):** AMI, ICSI (Edinburgh copy),
CHiME-6 (openslr 150), DiPCo, NOTSOFAR-1 (+ its 1,000 h simulated set),
AliMeeting, AISHELL-4, IMDA NSC Parts 3–6 (register, Open Data Licence),
LibriSpeech, LibriTTS-R, Common Voice 26, GigaSpeech, MLS, Emilia-YODAS
(not base Emilia), YODAS2, WenetSpeech, AISHELL-1/3, BUT ReverbDB, openSLR
26/28, MUSAN, DNS noise, LibriCSS; tooling: NeMo simulator + pyroomacoustics
(avoid gpuRIR's AGPL unless the simulator stays internal). Real
multi-party audio with human labels here totals only ~700 h (≈290 h
English meetings + ~240 h Mandarin + ~4,500 h IMDA dyadic), so §4.1's 20k h
target is reachable only through podcasts: run DuplexChat-Pipe's URL list
plus our own feed crawl, keep the separated streams, and mask by
confidence. Fit the §4.2 turn-taking Markov model from AMI + ICSI dialog
acts, not Fisher, so the simulator itself stays commercial-clean.

**Needs registration / agreement, still usable:** NOTSOFAR HF token,
IMDA account, M3SD gated Apache, Ego4D (research only; 48 h approval),
MSDWild (research PDF), VoxConverse (labels CC BY, audio from YouTube),
DIHARD III (LDC challenge licence; eval only).

**Not commercially usable — keep to research ablations and eval:** Fisher,
CALLHOME, Switchboard, Mixer 6 (LDC; would need for-profit membership),
Seamless Interaction, CANDOR, SPoRC, MagicData-RAMC (all NC), Emilia base
(CC BY-NC), WHAM! noise (NC; use MUSAN/DNS instead), and among TTS: Voxtral
TTS, Fish/OpenAudio, Breeze TTS 2, Higgs (capped). This matters for §4.3:
the 2026 RL papers all mine Fisher + Seamless, so an
internal-and-commercial reward set has to come from IMDA NSC (dyadic,
per-speaker channels, commercial) plus the planned 500 h human-rated slice.

**TTS for §4.3:** render human lines per speaker with CosyVoice 3 or
Chatterbox Turbo (Apache/MIT, cloning, laughter tags) and the assistant with
Orpheus or a fixed CosyVoice 3 voice; use Dia2 for English two-party
banter; VibeVoice / MOSS-TTSD only as a pre-mixed baseline. Mandarin and
code-switch scripts: CosyVoice 3 (18 dialects) and MOSS-TTSD.

**Corrections to the plan text:** §4.2 lists Emilia as a source — swap to
Emilia-YODAS or the licence blocks a shipped model; the 15% overlap prior
should be scenario-conditioned (AMI 21–22%, ICSI 9–14%, AliMeeting 35–42%,
AISHELL-4 9–19%, CTS ~10%), and §4.1 should name DuplexChat-Pipe as the
scaling path since it is the only open pipeline that outputs separated
per-speaker audio.
