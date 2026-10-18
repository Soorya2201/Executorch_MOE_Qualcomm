# MeetingGraph, by the numbers

> Every meeting you have is a decision you'll forget. MeetingGraph remembers them for you.
> It listens, writes the notes, and links every decision to the project it changes.
> **Entirely on your laptop's NPU. Nothing leaves the machine.**

This file is the presentation source of truth: every number below was **measured on this laptop**, and it
is updated whenever a new number lands. *Pending* means "being measured now"; nothing here is estimated
unless it says so.

Last updated: 2026-10-18

---

## 1. The problem

- Teams meet, decide, and forget. Notes are written late, by one person, and never connected to the work.
- Cloud meeting assistants fix the note-taking, but they **send every word you say to someone else's
  server**. For a lot of meetings (HR, legal, health, unreleased products, classrooms) that is a non-starter.
- Running AI locally was supposed to fix that, but on a normal laptop it **burns the CPU, the fan and the
  battery**, and is slow.

## 2. The one-line answer

**A meeting memory that runs on the laptop's neural processor (NPU), so it is private by design, quiet,
and fast enough to take notes while you are still talking.**

## 3. The demo, in three acts

1. **Record.** Press ● Record. A small capsule floats on top of everything; its sound wave pulses with your
   voice. Every 30 seconds the new audio becomes transcript lines, on the NPU.
2. **Review.** When you stop, the AI has already written the summary, decisions, action items and the
   product features the meeting touched, each with the exact quote it came from. Anything it is unsure
   about is a yellow question. You approve with one click; nothing is saved without you.
3. **Remember.** Every saved meeting updates a living map of the project: which feature needs which, which
   meeting changed what, and why. Ask a question and the answer cites its sources.

**One more thing:** all of that ran with the laptop's Wi-Fi off.

## 4. The numbers that matter

| | |
|---|---|
| **0** | bytes sent to the cloud. Speech, notes, search and answers all run locally |
| **100 %** | of the AI model's maths on the Hexagon NPU: 8 NPU calls, **0** operators falling back to the CPU |
| **~9 %** | CPU use while the 3.8-billion-parameter model is generating (the NPU does the work) |
| **3-6× faster notes** | after switching to Qwen3-1.7B (8k context, hybrid): biology lecture **398 s → 134 s**, design review 266 s → 47 s, kickoff 162 s → 34 s |
| **124 tokens/s** | prompt reading on the NPU with Qwen3-1.7B's 128-token prefill graph (Phi read ~10 tokens/s) |
| **2.4 s** | to load the meeting model onto the NPU (Qwen3-1.7B, 1.8 GB) |
| **12.2 tokens/s** | writing speed on the NPU (Phi-4-mini 12.2, Qwen3-1.7B 12.0) |
| **5.8× real time** | live speech-to-text: 30 s of audio transcribed in 5.0 s (Whisper-small on the NPU) |
| **10.1 %** | word error rate, Whisper-small on the NPU (17.1 % for Whisper-tiny) |
| **~2 %** | CPU use while Whisper transcribes: encoder and decoder are **1 NPU call each, 0 CPU operators** |
| **1.4 s** | answer time for a short request with the model kept loaded (was 7.3 s when it reloaded every time) |
| **6.76 ms** | per image, DeepLabV3 segmentation on the NPU (first model brought up on this stack) |
| **47.9 tokens/s** | Qwen3-0.6B on the NPU (small-model speed reference) |
| **5.1 GB** | the whole 3.8B-parameter model on disk after 8-bit quantization (about 7.7 GB in bf16) |
| **68** | automated tests, all passing |

## 5. Under the hood: the models and what we did to them

**Hardware:** Dell Latitude 7455 · Snapdragon X Elite X1E80100 · Hexagon NPU (HTP v73) · 32 GB RAM ·
Windows 11 on ARM.
**Software:** ExecuTorch 1.5.1 (Meta's on-device runtime) with the Qualcomm QNN backend, QAIRT 2.37.

| Job | Model | What we did |
|---|---|---|
| Notes, features, answers (**now**) | **Qwen3-1.7B** (Alibaba) | 8-bit KV cache, **8,000-token context**, hybrid: a 128-token prefill graph + a 1-token decode graph, compiled for SC8380 with QAIRT 2.46. Ran on our runner after a fix for its 3-D attention mask (patch 0007) and the 2.46 runtime DLLs |
| Notes, features, answers (before) | **Phi-4-mini-instruct** (Microsoft, 3.8B) | Post-training quantization to **W8A16** (8-bit weights, 16-bit activations), 8-bit KV cache, 2,048-token context, calibrated on our own samples, split into 8 shards, compiled to NPU binaries |
| Speech to text | **Whisper-small** (OpenAI, 244M) | 16a8w quantization, exported encoder + decoder for the NPU, our own NumPy log-mel front end (matches Hugging Face to 1.4e-5) |
| Reference | Whisper-tiny, Qwen3-0.6B, DeepLabV3 | Brought up first to prove the pipeline end to end |

**Accuracy vs the original model** (10 fixed test prompts, same automatic checks):

| Model | Passed | Speed | Same pass/fail as the original |
|---|---|---|---|
| Phi-4-mini, original bf16 on the CPU | 10/10 | slow, CPU-bound | n/a |
| **Phi-4-mini W8A16 on the NPU (v1, shipped)** | **7/10** | **12.2 tok/s** | 7/10 |
| Phi-4-mini, 16-bit KV cache on the NPU (v2, tried) | 7/10 | 10.0 tok/s | 7/10 |

The quantized model keeps math, code, JSON output, summaries and decisions; it loses two trick reasoning
questions and sometimes over-assigns action items. That is why every item carries its source quote and the
app asks the user before saving anything.

## 6. NPU optimizations, one by one

Every step below was made to get more work onto the Hexagon NPU, or to keep it from waiting.

| # | Optimization | What it does | Measured effect |
|---|---|---|---|
| 1 | **W8A16 quantization** (Phi-4-mini) | 8-bit weights, 16-bit activations, 8-bit KV cache; calibrated post-training | 7.7 GB → **5.1 GB**, 7/10 vs 10/10 on the test suite, **12.2 tok/s** |
| 2 | **Whole graph on the NPU** | Every operator is lowered to QNN; nothing falls back to the CPU | Phi: 8 NPU calls, **0 CPU ops**, CPU ~9 %. Whisper: 1 call per encoder/decoder, **0 CPU ops**, CPU ~2 % |
| 3 | **8 NPU shards with weight sharing** | The 3.8B model is split into 8 context binaries that fit the NPU's memory | Model runs at all on a 32 GB laptop |
| 4 | **16a8w Whisper** + our own NumPy log-mel | Speech model quantized for the NPU; front end has no PyTorch dependency at runtime | **5.8× real time**, WER 10.1 % |
| 5 | **Keep-loaded NPU server** (our `--server` patch to the runner) | Model is loaded into the NPU once and stays; requests go over stdin | **7.3 s → 1.4 s** per short request; byte-identical output |
| 6 | **Warm-up on Record** | The LLM starts loading onto the NPU the moment a meeting starts, so it is ready when you press Stop | Model load is hidden behind the meeting |
| 7 | **Stay loaded 15 minutes** | The NPU model is kept for 900 s after the last request, then freed | Back-to-back meetings skip the reload; idle laptops get their memory back |
| 8 | **One NPU at a time** (`NPU_LOCK`) | Whisper and Phi take turns on the NPU instead of fighting over it | No failed loads or stalls while recording and summarizing |
| 9 | **Context-sized requests** | Transcript chunks and prompts are sized to the model's 2,048-token window; compact prompts for small contexts | No truncated prompts; fewer NPU requests per meeting |
| 10 | **Hybrid prefill** | A graph that reads 128 prompt tokens per NPU step instead of 1 | **Done with Qwen3-1.7B: 124 tok/s prompt, notes 3-6× faster.** (Our own Phi hybrid export is still blocked by RAM) |
| 10b | **8k context** | The whole meeting fits in one request instead of 2,500-character chunks | Biology lecture: 5 requests → 2 |
| 10c | **Newer QAIRT runtime, side by side** | Qwen runs with the QAIRT 2.46 DLLs in their own folder; Whisper keeps 2.37 | Nothing that worked was touched |
| 11 | **Memory-safe exporter** | Re-ordered so the exporter never holds three copies of the model; frees each stage | Peak: 3 copies → 2 |
| 12 | **Verified calibration checkpoint** | The slow calibration step is saved once and fingerprint-checked on resume | A crashed export resumes without re-calibrating |
| 13 | **Clear memory errors** | When the NPU model cannot load, the app says how many GB are free vs needed | No silent "model failed" |

**Phi-4-mini vs Qwen3-1.7B, same transcripts, the app's full notes + features pipeline:**

| Meeting | Phi-4-mini (kv, 2k) | **Qwen3-1.7B (hybrid, 8k)** | Speed-up |
|---|---|---|---|
| Biology lecture, 2,051 chars | 398 s, 5 requests | **134 s, 2 requests** | 3.0× |
| Atlas design review, 1,285 chars | 266 s, 3 requests | **47 s, 3 requests** | 5.7× |
| Atlas kickoff, 508 chars | 162 s, 2 requests | **34 s, 3 requests** | 4.8× |
| Atlas sprint sync, 492 chars | n/a | **42 s** | |
| Model load (warm) | 4.2 s | **2.4 s** | 1.7× |

Honest quality note: on the kickoff, Phi listed 5 features and Qwen 1 (Qwen merged "login & offline sync" into one
name, which the transcript check then removed); on the other meetings Qwen found as much or more (6 grounded
organelles in the lecture) and filled decisions and questions that Phi left empty. Every item still has to pass
the transcript check, and the one-feature-per-minute rule tops up any shortfall.

Before the switch (Phi): one real meeting 387 s, one Ask answer 407 s, a short request 1.4 s with the model loaded.

## 7. Engineering we had to invent

- **Windows on ARM did not run ExecuTorch's Qualcomm runners out of the box.** Fixed a heap-corruption
  crash (aligned free), registered the NPU backend in the runner's own registry, and patched Whisper's
  export for non-tiny models. Patches are in `patches/`.
- **A keep-loaded NPU server.** The stock runner loads the 5 GB model for every single request. We added a
  `--server` mode that loads once and answers requests over stdin, with output **byte-identical** to the
  stock runner: **7.3 s → 1.4 s** per short request.
- **Fast prompt reading (hybrid mode), in progress.** Phi currently reads prompts one token per NPU step
  (~10 tokens/s), which is ~80 % of the time per meeting. Hybrid mode adds a graph that reads many tokens
  per step. Its export needed more than 32 GB RAM + 64 GB page file, so we **re-ordered the exporter to
  never hold three copies of the model at once**, added a **verified calibration checkpoint**, and cut the
  prefill block from 128 to 32 tokens. Calibration now completes; the final QNN compile still runs out of
  memory (~62 GB resident at that point), so it needs a bigger machine or a smaller model.
- **A crash in the audio library.** Recording crashed the app at random: the `soundcard` library reserves
  16 bytes for a Windows structure that is 24 bytes on 64-bit, corrupting memory. Found by reproducing it
  outside the app, fixed in our recorder.
- **Small-model output, handled.** Phi sometimes answers with a bare list instead of the requested object;
  the parser now accepts it (this alone had been dropping every feature from the notes).
- **No surprise windows.** Runner processes start without console windows, so live transcription no longer
  flashes a terminal every 30 seconds.

## 8. Privacy and trust, by design

- Everything (audio, transcripts, notes, the project map, the search index) lives in one local folder.
- The AI proposes; **the user approves**. Every decision and feature carries a quote from the transcript,
  and quotes that cannot be found in it are flagged.
- Saves are transactional: a failed save leaves the project exactly as it was.
- Full transcripts are kept with each saved meeting for reference (one switch to turn that off).

## 9. Design

- Built on Microsoft's **Fluent 2** design language: Segoe UI Variable, Fluent brand blue, 4/8 px radii,
  layered soft shadows, line icons, tinted Windows 11 title bar, own taskbar icon.
- A hand-drawn **doodle wallpaper** (SVG) sits faintly behind the white panels.
- The recorder capsule's sound wave is painted at 60 fps from the live microphone level.
- Fits the laptop's 1280×752 work area (2560×1600 at 200 % scaling); an automatic check renders every
  page at 3 window sizes and fails on any clipped or overlapping text.

## 10. Pending numbers (being measured now)

| What | Status |
|---|---|
| Hybrid prompt reading | **done with Qwen3-1.7B: 124 tok/s.** Our own hybrid Phi export: calibration checkpointed, QNN compile needs more RAM |
| Hybrid Phi: time to notes for a 2,000-character meeting | *pending* (today: 398 s in 5 requests; target: under a minute) |
| Hybrid Phi: accuracy on the 10-prompt suite | *pending* (must stay at v1's 7/10) |
| Whisper + Phi on the NPU at the same time (live notes while recording) | *serialized by NPU_LOCK; concurrent run pending* |
| Qwen3-1.7B accuracy on the 10-prompt suite | *pending* |
| Ask with Qwen3-1.7B (8k context) | *pending* |

## 11. Talk track (60 seconds)

> "Every meeting ends with decisions, and most of them are forgotten by Friday. Cloud assistants remember
> them for you, by sending every word to a server. We wanted the opposite.
> This is MeetingGraph. It runs on the neural processor inside this laptop. [Record.] While I talk, it is
> transcribing, six times faster than I can speak. [Stop.] It has already written the notes, and every
> decision points back to the sentence it came from. I approve, and the project map updates: what changed,
> what depends on what, and why.
> A language model with an 8,000-token memory, on the NPU, writing the notes in about two minutes.
> And the Wi-Fi has been off the whole time."

---

### Changelog of numbers

- 2026-10-17: Phi-4-mini W8A16 on NPU: 12.2 tok/s, NPU 100 %, CPU ~9 %. Whisper-small WER 10.1 %, tiny 17.1 %.
  DeepLabV3 6.76 ms. Qwen3-0.6B 47.9 tok/s.
- 2026-10-18: v1 vs v2 vs bf16: 7/10, 7/10, 10/10; v2 10.0 tok/s (kept v1). Live ASR 5.8× real time.
  Keep-loaded server 7.3 s → 1.4 s. Meeting notes before hybrid: 2,051 chars in 398 s (5 requests).
  47 tests passing.
- 2026-10-18 (later): Whisper delegation verified: encoder + decoder 1 NPU call each, 0 CPU ops, CPU ~2 %,
  6.0 s for the test clip, WER 12.3 % on our own meeting recording. Real meeting notes: 387 s. Ask: 407 s.
  Hybrid export: calibration checkpointed, compile out of memory. Review shows only transcript-backed
  features, at least one per minute of speech. 68 tests passing.
- 2026-10-18 (evening): switched the meeting model to Qwen3-1.7B (8k, hybrid, QAIRT 2.46): load 2.4 s, prompt
  124 tok/s, decode 12.0 tok/s. Notes: biology 398 s → 134 s, design review 266 s → 47 s, kickoff 162 s → 34 s.
  Runner patch 0007 (3-D attention mask).
