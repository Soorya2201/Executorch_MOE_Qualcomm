# LEARNINGS: running LLMs and speech models on the Snapdragon X Elite NPU with ExecuTorch

A record of everything done in this project, what each step taught, and how to explain it. Written so that
someone (including future me) can redo the work and defend it in front of senior engineers.

Measured numbers live in [meetinggraph/QUANTIFY.md](meetinggraph/QUANTIFY.md). Setup steps, patches and gotchas
are in [README.md](README.md). This file explains the *why*.

---

## 1. The big picture in one paragraph

A Hugging Face model (`.safetensors`) is only a table of numbers. To run it on the Hexagon NPU we used
ExecuTorch's Qualcomm example pipeline (`examples/qualcomm/oss_scripts/llama/`). It loads those numbers into an
NPU-shaped re-implementation of the model, records the computation as a fixed graph, quantizes it to integers,
compiles it with Qualcomm's QNN compiler into Hexagon v73 machine code, and packs the result into one `.pte`
file. A small C++ runner (`qnn_llama_runner.exe`) then drives the token loop on the CPU, while every matrix
multiplication runs on the NPU.

```
microsoft/Phi-4-mini-instruct (.safetensors, bf16, 7.7 GB)
  -> convert_weights.py        rename/split HF weights into the static model's names
  -> static_llama.py           NPU-shaped model: fixed shapes, KV cache as I/O, 1x1 convs
  -> torch.export              a fixed list of ops (the graph)
  -> PT2E quantization         W8A16: 8-bit weights, 16-bit activations, calibrated on sample text
  -> QNN / QAIRT compiler      Hexagon v73 context binaries, 8 shards, 0 ops left on the CPU
  -> kv_llama_qnn.pte (5.1 GB) one file: program + NPU binaries + metadata
  -> qnn_llama_runner.exe      CPU: tokenizer + sampling loop; NPU: all the maths
```

---

## 2. What the "Llama-style" code in `oss_scripts/llama` is

**It's an NPU-friendly re-implementation of the decoder-only transformer.** Llama, Qwen, Phi, Gemma, Granite and
SmolLM all share the same skeleton (embedding → N × [norm, attention, norm, MLP] → norm → output head), so
one model file, `model/static_llama.py`, serves them all. Each model only adds a config and a weight mapping.
"Static" means every tensor shape is fixed when the model is exported, because the NPU compiler needs fixed shapes.

### The files and what each one does

| File (under `C:\et\executorch\examples\`) | Role |
|---|---|
| `qualcomm/oss_scripts/llama/llama.py` | The export entry point: `--decoder_model phi_4_mini-w8a16 --model_mode kv --max_seq_len 2048 ...` |
| `qualcomm/oss_scripts/llama/__init__.py` | Registry of supported models (`@register_llm_model("phi_4_mini-w8a16")`): HF repo id, config path, weight converter, number of shards, quant recipe. **Patch 0003 adds our W8A16 entry here** |
| `models/phi_4_mini/convert_weights.py` | Maps Hugging Face names to Meta/Llama names. Phi stores Q, K and V fused (`qkv_proj`) and gate+up fused (`gate_up_proj`); the converter **splits** them into `wq/wk/wv` and `w1/w3` |
| `models/phi_4_mini/config/config.json` | Architecture: dim 3072, 32 layers, 24 query heads, 8 KV heads (grouped-query attention), vocab 200,064, partial RoPE 0.75 |
| `qualcomm/oss_scripts/llama/model/static_llama.py` | The NPU-shaped model (`LlamaAttention`, `FeedForward`, `LlamaDecoderLayer`, `LlamaModel`) |
| `qualcomm/oss_scripts/llama/static_llm_quant_recipe.py` | Per-model quantization recipes. **Patch 0003 adds `Phi4MiniW8A16QuantRecipe`** |
| `qualcomm/oss_scripts/llama/wrappers/llm_wrappers.py` | Orchestrates calibrate → quantize → lower → compile (kv and hybrid). **Patch 0006 changes this for memory** |
| `qualcomm/oss_scripts/llama/runner/*.cpp` | The C++ runtime: `runner.cpp` (setup), `prompt_processor.cpp` (prefill), `token_generator.cpp` (decode), `kv_manager.cpp` (KV cache buffers). **Patches 0005 and 0007 change these** |

### What makes `static_llama.py` different from the Hugging Face model (the parts to explain)

1. **Linear layers become 1×1 `Conv2d`** (`wq_conv`, `wk_conv`, `w1_conv`, ...). The maths is identical, but the
   Hexagon tensor processor runs convolutions with its fastest kernels. That's why the quant recipe targets
   `aten.conv2d`.
2. **The KV cache is an explicit input and output**, not hidden state. Each step takes the past K/V
   (`[1, 8 heads, 128, 2047]` for Phi) and returns **only the new token's K/V** (`output_new_cache_only`). The runner
   writes that slice into a CPU-side buffer, so the whole cache never has to be copied back out of the NPU.
   K is stored transposed (`[heads, head_dim, seq]`) so that Q·Kᵀ is a plain matrix multiply.
3. **Fixed shapes and an explicit attention mask.** The context length (2,048 for Phi, 8,000 for our Qwen) and the
   tokens per step (AR length: 1 for decode, 32 or 128 for prefill) are baked in. Positions not yet used are
   masked out with an attention-mask input instead of changing tensor sizes.
4. **Two graphs ("hybrid" mode).** `prefill_forward` reads the prompt AR tokens per NPU call; `kv_forward` writes
   the reply one token per call. Both share the same weights inside the `.pte`. In "kv" mode only the 1-token
   graph exists, so a 1,000-token prompt costs 1,000 NPU calls. That was Phi's bottleneck.
5. **Sharding.** The 32 layers are split into 8 NPU contexts (`num_sharding = 8`) because one context can only hold
   so much; a token passes through them in sequence. Weight sharing lets the prefill and decode graphs use the
   same copy of the weights.
6. **Metadata methods** (`get_max_seq_len`, `get_vocab_size`, `get_kv_io_bit_width`, ...) are compiled into the
   `.pte`, so the runner configures itself from the model file.

---

## 3. How to present this so senior engineers trust it

Senior people trust **precise claims with evidence and stated limits**. They distrust "we built an AI that
runs on the NPU" without numbers, and they distrust anything that hides the trade-offs.

### Say what you built on, and what you added

> "We use Meta's ExecuTorch and its Qualcomm QNN backend, the official PyTorch path to the Hexagon NPU. The
> model architecture code is ExecuTorch's static Llama implementation; we didn't rewrite the transformer.
> Our work is everything needed to make it run and be useful on a Windows-on-ARM laptop: seven patches to
> ExecuTorch, a new 8-bit quantization recipe for Phi-4-mini, a persistent NPU server mode, a memory-safe
> exporter, and the application on top."

Being clear about the boundary between upstream work and your own is the biggest credibility signal. Claiming
upstream work as your own is the fastest way to lose a senior audience.

### Lead with verifiable numbers, each with its method

| Claim | How it was measured (say this) |
|---|---|
| "100 % of the model's maths on the NPU, 0 CPU operators" | ExecuTorch delegation report on the `.pte`: every method is a single QNN delegate call; Inspector/ETDump traces show no CPU kernels |
| "~9 % CPU while generating" | Task Manager / perf counters during a run; the CPU only tokenizes and samples |
| "12.2 tokens/s decode, 124 tokens/s prompt" | The runner's own `PyTorchObserver` timing line, temperature 0 |
| "7/10 vs 10/10 for the original" | A fixed 10-prompt suite with automatic checks, run on the bf16 model on the CPU and the W8A16 model on the NPU |
| "Notes 3-6× faster with Qwen3-1.7B" | The app's real pipeline (`scripts/llm/bench_meeting_notes.py`) on the same transcripts, both models |
| "WER 10.1 % (Whisper-small)" | `jiwer` against reference transcripts |

### State the trade-offs before they ask

- "Quantization costs accuracy: 7/10 vs 10/10. It lost two trick reasoning questions and sometimes over-assigns
  action items. So every extracted item carries its source quote, and the app removes anything the transcript
  doesn't back."
- "Export runs on x64 Python under emulation and needs a lot of memory. Our own hybrid Phi export still runs out
  of memory at the QNN compile step on 32 GB + 64 GB page file."
- "The Qwen3-1.7B we ship was compiled elsewhere with QAIRT 2.46; we made our runner compatible (patch 0007) and
  run it with the 2.46 runtime libraries."

### Questions to expect, with honest answers

| Question | Answer |
|---|---|
| Why not ONNX Runtime + QNN EP, or Qualcomm AI Hub? | The brief was ExecuTorch. It also gives one PyTorch-native path from training code to device, with the same `.pte` on Android/Linux/Windows. AI Hub is a valid alternative for prebuilt models |
| How do you know it's really on the NPU? | Delegation report (0 CPU ops), Inspector traces, and the NPU engine counter at 100 % while the CPU sits at ~9 % |
| What's W8A16, and why not 4-bit? | 8-bit per-channel weights, 16-bit activations. The stock recipe for Phi is mostly 4-bit; we chose 8-bit to protect summary quality and measured the cost (7/10) |
| What is calibration? | Running sample text through the float model to record each activation's range, which sets the integer scales. Bad calibration data means clipped or imprecise activations |
| Why is prompt reading slow in kv mode? | One NPU call per prompt token. Hybrid mode adds a graph that reads 128 tokens per call: 124 tok/s instead of ~10 |
| Why 8 shards? | Per-context memory limits on the HTP; sharding keeps each compiled graph within them |
| What happens with a long meeting? | Transcripts are chunked to fit the context (2,048 tokens for Phi, 8,000 for Qwen) and the parts are merged |
| Privacy? | Audio, transcripts, notes and search all stay in one local folder; nothing is sent to a network service |

### Phrases to avoid

"Fully optimized", "no accuracy loss", "real time" (unless you give the factor: 5.8× real time), "our model"
(it's Microsoft's or Alibaba's model, quantized and compiled by us), and any number that hasn't been measured.

---

## 4. Timeline: everything done, in order

### Day 1 (2026-10-17): the NPU stack

1. **Toolchain on Windows-on-ARM:**
   - Git, CMake, **x64** Python 3.12 (export dependencies have no ARM64 wheels) and VS 2022 Build Tools (ARM64 + x64 MSVC, Clang);
   - QAIRT **2.37** (the version ExecuTorch v1.5.1 pins);
   - ExecuTorch v1.5.1 built from source;
   - Developer Mode turned on for git symlinks.
2. **First model, DeepLab v3 (INT8):** 6.76 ms/image on the NPU, 93.0 % pixel accuracy. Found and fixed a
   **heap-corruption crash** in `qnn_executor_runner` on Windows: `free()` was called on `_aligned_malloc`
   memory, and the ETDump was written in text mode (**patch 0001**).
3. **LLM runner on Windows:** "QnnBackend is not registered", because on Windows the backend DLL keeps its own
   registry, so the exe must call `QnnExecuTorchBackendRegister` (**patch 0002**). First LLM: Qwen3-0.6B at
   47.9 tok/s.
4. **Whisper on the NPU:** the export hard-coded `whisper-tiny` and a 384-wide decoder input (**patch 0004**).
   Whisper-small reached WER 10.1 %, 5.8× real time, and we wrote our own NumPy log-mel front end (matches
   Hugging Face to 1.4e-5).
5. **Phi-4-mini W8A16:**
   - a new model entry plus recipe (**patch 0003**);
   - page file raised to 64 GB;
   - export with 8 calibration samples, about 25 min of calibration;
   - result: a 5.1 GB `.pte` at 12.2 tok/s.
6. **MoE investigation:** ExecuTorch v1.5.1 has no MoE pipeline for the Qualcomm backend (MoE targets CUDA and
   MLX). The routing ops exist in QNN. Written up in `docs/moe-pipeline-brainstorm.md`.
7. **File organizer app** (NPU LLM used only as a classifier; code does every file move). Later dropped.
8. Ollama removed at the owner's request: ExecuTorch only.

### Day 2 (2026-10-18): MeetingGraph and performance

9. **Phi v1 vs v2 vs original:**
   - an automatic 10-prompt suite;
   - bf16 on the CPU 10/10; v1 (8-bit KV cache) 7/10 at 12.2 tok/s; v2 (16-bit KV cache) 7/10 at 10.0 tok/s;
   - kept v1, since the 16-bit cache gave no accuracy gain.
10. **Keep-loaded server** (**patch 0005**, `--server`):
    - the model loads once and requests arrive over stdin;
    - a short request went from 7.3 s to 1.4 s, with output byte-identical to the one-shot runner.
11. **MeetingGraph app:**
    - Whisper and Phi backends;
    - merged a teammate's live-recording and knowledge-map work;
    - Fluent 2 redesign;
    - recorder capsule with a live sound wave;
    - an automatic overlap and clipping check of the UI at 3 window sizes.
12. **Bugs fixed:**
    - a random recording crash: the `soundcard` library reserved 16 bytes for a 24-byte Windows `PROPVARIANT`;
    - console windows flashing every 30 s;
    - Phi returning a bare list where an object was expected;
    - six more bugs, each with a regression test.
13. **Hybrid Phi export (prefill graph):**
    - kept running out of memory, so we re-ordered the exporter (never three fp32 copies at once), added
      garbage collection between stages, and added a verified calibration checkpoint (**patch 0006**);
    - calibration now completes and is saved;
    - the final QNN compile still fails with `bad allocation` (~62 GB resident). **Not solved on this machine.**
14. **Trust rules in the app:**
    - features not backed by the transcript are removed (a prompt example, "Mitochondria", had been copied into a
      casual chat's notes);
    - at least one feature per minute of speech, every one grounded in the transcript.
15. **Switch to Qwen3-1.7B (8k context, hybrid)** (compiled with QAIRT 2.46 by someone else, downloaded as a `.pte`):
    - "Using newer context binary on old SDK": installed the 2.46 runtime DLLs in a separate folder (`npu-run-246`);
    - the runner crashed in prefill because the model's attention mask is 3-D: read the last two dims (**patch 0007**);
    - result: notes 3-6× faster than Phi (398 s → 134 s on the biology lecture).

---

## 5. Concepts learned (short definitions worth knowing cold)

- **ExecuTorch:** PyTorch's on-device runtime. Models are exported ahead of time into a `.pte` (FlatBuffer)
  holding the program, constants and backend blobs.
- **Delegate / backend:** a part of the graph handed to a hardware backend. With QNN, the whole model becomes one
  delegate call per method.
- **QNN / QAIRT:** Qualcomm's AI runtime and SDK. **HTP** is the Hexagon Tensor Processor (the NPU); **v73** is the
  X Elite's HTP version; **SC8380XP** is ExecuTorch's name for this chip.
- **Context binary:** the compiled NPU program. **It's tied to the QAIRT version that built it:** an older runtime
  refuses a newer binary; a newer runtime generally loads an older one.
- **PT2E quantization:** prepare (insert observers) → calibrate (run data) → convert (quantize/dequantize nodes,
  the "QDQ" graph) → lower.
- **W8A16 / 16a8w:** 8-bit weights, 16-bit activations. **Per-channel:** one scale per output channel (better for
  weights). **Per-tensor:** one scale per tensor.
- **KV cache:** stored keys and values of past tokens, so each new token only computes itself. Its bit width
  (8 vs 16) trades memory for precision.
- **kv vs hybrid mode:** kv has only the 1-token graph; hybrid adds a multi-token prefill graph. **AR length** is
  the tokens per NPU call.
- **Weight sharing:** the prefill and decode graphs use one copy of the weights.

---

## 6. Debugging lessons (what actually saved time)

1. **Read the model's own metadata before blaming the runtime.** The Qwen crash was solved by listing the
   `.pte`'s input shapes (`Runtime.load_program(...).metadata("kv_forward")`). The mask was `[1, 1, 8000]`, not
   `[1, 1, 1, 8000]`, and the runner had hard-coded dims 2 and 3.
2. **Version strings are in the binary.** Searching the `.pte` bytes found `v2.46.0.260424121129`, which named
   the exact SDK to download.
3. **Never replace a working setup to test a new one.** The 2.46 runtime went into `npu-run-246`; Whisper and the
   Phi setup kept running on 2.37.
4. **Reproduce crashes outside the app.** The audio crash was found by calling `soundcard` alone in a loop, then
   reading the ctypes structure definition.
5. **Measure memory, don't guess.** Export failures were out-of-memory at different stages; logging resident
   tensors and commit charge showed where the copies were.
6. **Checkpoint long jobs, and verify the checkpoint.** A calibration that takes 25+ minutes under emulation is
   saved once and trusted only after replaying it reproduces the original quantization exactly.
7. **Background jobs die for non-code reasons:**
   - the laptop sleeps on battery (keep it awake with `SetThreadExecutionState`, plugged in, lid open);
   - a parent tool kills children under memory pressure (run long exports as detached processes).
8. **Check the claim before writing it down.** For example, Phi's load time was 4.2 s from the log, not the
   ~60 s first assumed. Every number in QUANTIFY.md comes from a log or a measured run.

---

## 7. App-level lessons (small models in a real product)

- **Small models copy prompt examples.** An example feature name in the prompt ("Mitochondria") showed up in an
  unrelated meeting's notes. Use placeholders, not realistic examples.
- **Ground everything.** Every decision and feature carries an exact quote; anything the transcript doesn't back
  is removed before the user sees it.
- **Size prompts to the context.** With 2,048 tokens, the transcript must be chunked and the JSON reply budgeted;
  8,000 tokens let a whole meeting fit in one request.
- **Prompt reading dominates on-device latency.** With kv mode, about 80 % of meeting time was reading the
  prompt. Hybrid prefill gave the 3-6× speed-up.
- **Keep the model loaded** and load it when recording starts, so it's ready when the user presses Stop.
- **Serialize NPU use** (`NPU_LOCK`) so Whisper and the LLM don't fight over it.

---

## 8. Mistakes and what I'd do differently

- Started the Phi hybrid export several times before understanding its memory profile. **Next time:** profile peak
  memory on a small model (Qwen3-0.6B) with the same flags first, then scale up.
- Assumed a number (model load ~60 s) before checking the log, and corrected it before publishing. **Always** quote
  the log.
- Calibrated on generic encyclopedia text. Meeting-domain calibration text (`build_calib_meeting.py`, prepared) would
  likely help summary quality. **Not yet run.**
- Exported on a 32 GB laptop. Large hybrid exports belong on a Linux x64 machine with 128 GB+ RAM; only the
  runtime needs to run on the laptop.

---

## 9. Open items

- Qwen3-1.7B: run the 10-prompt accuracy suite and test Ask with the 8k context.
- Our own hybrid export (Phi, or Qwen3-1.7B at 8-bit) on a bigger machine.
- Meeting-domain calibration and a per-layer weight SQNR report (`qdq_weight_sqnr.py`, prepared).
- MoE on the QNN backend: design notes only (`docs/moe-pipeline-brainstorm.md`).

---

## 10. Where everything is (this laptop)

| What | Path |
|---|---|
| ExecuTorch source + builds | `C:\et\executorch` (`build-x86_64-windows` for export, `build-arm64-windows` for runners) |
| QAIRT 2.37 | `C:\Qualcomm\AIStack\QAIRT\2.37.0.250724` |
| QAIRT 2.46 runtime (Qwen3-1.7B) | `C:\et\incoming\qairt246`, copied into `C:\et\llm\npu-run-246` |
| Phi-4-mini W8A16 `.pte` | `C:\et\llm\phi4mini_w8a16_2k\kv_llama_qnn.pte` |
| Qwen3-1.7B 8k `.pte` | `C:\et\llm\qwen3_1_7b_ctx8k\` |
| Hugging Face weights | `C:\Users\qc_de\.cache\huggingface\hub\` |
| Export logs | `C:\et\llm\export_*.log` |
| Full Day-1 change log | `docs/works.txt` |
| App | `meetinggraph/` (config in `meetinggraph/config.yaml`) |
| Patches 0001-0007 | `patches/` (apply in order on ExecuTorch v1.5.1) |
