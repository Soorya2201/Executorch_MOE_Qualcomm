# Executorch_MOE_Qualcomm

Running PyTorch **ExecuTorch** models on the **Qualcomm Hexagon NPU** of a Snapdragon X Elite
Windows-on-ARM laptop, plus groundwork for a **Mixture-of-Experts (MoE) pipeline** on the Qualcomm
QNN/HTP backend.

| | |
|---|---|
| Device | Dell Latitude 7455: Snapdragon X Elite X1E80100, 32 GB, Windows 11 ARM64 |
| NPU | Hexagon HTP v73, ExecuTorch target `SC8380XP` (SoC ID 60) |
| ExecuTorch | v1.5.1 (`3b60683923245cf472b7323426920e15623ba361`) |
| QAIRT / QNN | 2.37.0.250724 (the version pinned by v1.5.1's `qnn_config.sh`) |
| Based on | [aaishikasb/qc-executorch-guide](https://github.com/aaishikasb/qc-executorch-guide), Compute track |

## Results

| Model | Precision | Result on the NPU |
|---|---|---|
| DeepLab v3 (segmentation) | 8a8w | 6.76 ms/image, 93.0 % PA / 64.7 % mIoU |
| **Qwen3-0.6B** (LLM) | 16a4w, kv, 512 ctx | **47.9 tok/s** decode, 47.5 tok/s prompt, TTFT 0.41 s |
| **Whisper-tiny** (speech-to-text) | 16a8w | 51 ms encode per 30 s, about 7 ms/token, WER 17.1 % |
| **Whisper-small** (speech-to-text) | 16a8w | 134 ms encode per 30 s, about 49 ms/token, **WER 10.1 %** |
| Phi-4-mini-instruct | **W8A16** (new recipe), kv, 2048 ctx | 12.2 tok/s decode, ~10 tok/s prompt, 7/10 on our test prompts |
| **Qwen3-1.7B** (MeetingGraph's LLM) | 8-bit KV, **hybrid (128-token prefill)**, 8000 ctx, QAIRT 2.46 | **124 tok/s prompt**, 12.0 tok/s decode, loads in 2.4 s; meeting notes **3-6x faster than Phi** |

All of these compile to a **single QNN delegate call per method with zero CPU kernels**
(`scripts/pte_delegation_report.py`). During Qwen generation the NPU compute engine reads 100 % while
the runner uses about 60 % of one core (5 % of the CPU) for its host loop.

**DeepLab v3 (INT8, 8a8w) on the NPU**

| Metric | Value |
|---|---|
| Latency | **6.76 ms / inference** (about 148 inferences/s), 10 warm-up + 100 iterations |
| Sustained vs burst HTP mode | same (6.75 ms). Power mode isn't the bottleneck |
| Accuracy (20 VOC 2012 val images) | **93.0 % pixel accuracy, 64.7 % mIoU** (optimistic: calibration also drew from val) |
| NPU evidence | ETDump: `DELEGATE_CALL` = 6.787 ms of `Method::execute` 6.788 ms (99.99 % in the QNN delegate, no CPU ops). Windows: compute-only adapter at 100 % during runs |

See [`docs/works.txt`](docs/works.txt) for the full log of everything that was done, and
[`results/`](results/) for raw runner logs.

## Bug found and fixed: `qnn_executor_runner` crashes on Windows

[`patches/0001-...patch`](patches/0001-qnn_executor_runner-fix-windows-heap-corruption-and-etdump-text-mode.patch)

The runner finished every inference, then died with `0xC0000374` (heap corruption) in `ntdll.dll`.

1. **Crash:** the ETDump buffer comes from `flatcc_builder_finalize_aligned_buffer()`, which on MSVC
   is `_aligned_malloc` (`flatcc/portable/paligned_alloc.h`: *"_aligned_malloc is not compatible with
   free"*). The runner released it with plain `free()`. Fix: `flatcc_builder_aligned_free()`.
   This is Windows-only, because Linux/Android use `posix_memalign`, where `free()` is correct.
2. **Silent corruption:** the ETDump file was opened with `fopen(..., "w+")`, which is text mode on
   Windows, so every `0x0A` became `0x0D 0x0A`. Evidence: logged size 3384, file 3388, exactly 4 CRLFs.
   This also broke the ExecuTorch Inspector. Fix: `"wb"`.

After the fix, every test exits 0, ETDump sizes match exactly, and the Inspector reads the profiles.

Apply it to an ExecuTorch v1.5.1 checkout:
```powershell
cd executorch
git apply ..\Executorch_MOE_Qualcomm\patches\0001-qnn_executor_runner-fix-windows-heap-corruption-and-etdump-text-mode.patch
```

## More Windows fixes (patches 0002-0004)

- **[0002](patches/0002-llm-runners-register-QnnBackend-on-windows.patch): `Backend QnnBackend is not registered`**
  in `qnn_llama_runner` / `qnn_multimodal_runner`. On Windows the backend DLL keeps its own registry,
  so the exe must call `QnnExecuTorchBackendRegister(register_backend)` at startup, as
  `qnn_executor_runner` and `qnn_whisper_runner` already do. The call is a no-op on Linux/Android.
- **[0003](patches/0003-phi_4_mini-w8a16-recipe.patch): `phi_4_mini-w8a16`**, a Phi-4-mini model entry
  whose recipe keeps every weight at 8-bit per-channel with 16-bit activations (the stock recipe is mostly 4-bit).
- **[0004](patches/0004-whisper-hf_model-arg-and-config-sized-inputs.patch): Whisper beyond tiny.** `whisper.py`
  hard-coded `openai/whisper-tiny` (added `--hf_model`) and `whisper_model.py` hard-coded the decoder's
  example `encoder_hidden_states` as d_model 384, so small (768) failed with a channel mismatch.

- **[0005](patches/0005-qnn_llama_runner-server-mode.patch): `qnn_llama_runner --server`.** Loads the
  `.pte` once and answers requests over stdin (`<prompt_file>\t<seq_len>\t<output_file>` -> `OK\t...`), so an app
  does not reload a 5 GB model per request: 7.3 s -> 1.4 s for a short request. Replies are byte-identical to the
  one-shot runner (verified), and each request starts a fresh conversation.
- **[0006](patches/0006-hybrid-export-memory-safe-and-calibration-checkpoint.patch): hybrid export within
  32 GB RAM.** Hybrid mode (a 128-token prefill graph + the kv graph) built three fp32 copies of the model at once
  and ran out of RAM + a 64 GB page file. The exporter now frees the calibration copy before building the prefill
  copy (two copies at peak) and garbage-collects between stages. It also saves the calibration result
  (`calibration_encodings.pt`) and trusts it only after replaying it reproduces the original encoding override
  exactly, so a crash after calibration costs minutes instead of a full re-calibration.

- **[0007](patches/0007-qnn_llama_runner-3d-attention-mask.patch): 3-D attention masks.** Newer exports (the
  QAIRT 2.46 Qwen3-1.7B 8k model) store the mask as `[1, AR, CL]`, not `[1, 1, AR, CL]`. The runner read fixed
  dims 2/3, took the 8000-token context as the prompt block size and crashed in prefill; it now reads the last two
  dims. A model compiled with a newer QAIRT also needs that QAIRT's runtime DLLs next to the runner
  ("Using newer context binary on old SDK" otherwise): `scripts/llm/setup_qwen3_1_7b.ps1` sets that up.

Apply all patches in order (each applies cleanly on v1.5.1 plus the earlier ones): `git apply ..\Executorch_MOE_Qualcomm\patches\*.patch`

## MeetingGraph (`meetinggraph/`)

A private meeting memory that runs on the NPU: record or import a meeting, get a transcript (Whisper-small on the
NPU), notes and feature changes (Qwen3-1.7B, 8k context, on the NPU; Phi-4-mini W8A16 before), approve them, then browse the project map and ask
cited questions. Nothing leaves the laptop. **Every measured number is in
[meetinggraph/QUANTIFY.md](meetinggraph/QUANTIFY.md)** (kept up to date); setup is in
[meetinggraph/README.md](meetinggraph/README.md).

Phi-4-mini accuracy on 10 fixed prompts ([results/llm/phi_eval](results/llm/phi_eval)): original bf16 on the CPU
10/10; W8A16 on the NPU 7/10 at 12.2 tok/s (shipped); 16-bit KV cache 7/10 at 10.0 tok/s.

## Windows-on-ARM build gotchas (all solved)

| Symptom | Cause | Fix |
|---|---|---|
| `install_executorch.py`: `unknown type name 'uint'` in `vec128_uint_aarch64.h` | VS generator defaults to ARM64 on an ARM64 host; the Python wheel must be x64 | `CMAKE_GENERATOR="Visual Studio 17 2022"` + `CMAKE_GENERATOR_PLATFORM=x64` (the platform setting is ignored without the generator) |
| `build.ps1` dies on a CMake deprecation warning | PowerShell 5.1 turns native stderr into terminating errors when piped (`*>&1 \| Tee-Object`) | run it via `cmd /c "... > log 2>&1"` |
| Export: `ModuleNotFoundError: pydot` | `install_executorch.py` skips `backends/qualcomm/requirements.txt` | `pip install -r backends/qualcomm/requirements.txt pandas` |
| Export: `Failed to initialize QNN backend for kHtpBackend` | `envsetup.ps1` adds the **ARM64** QNN DLLs (host shell arch), but export runs in **x64** Python | put `lib\x86_64-windows-msvc` first on PATH (see `scripts/et-env.ps1`) |
| QAIRT zip: "End of Central Directory not found" | download truncated mid-stream | resume with `curl.exe -C -` |
| HF `snapshot_download` writes 0-byte weights / stalls | Xet transfer backend issue | `scripts/hf_curl_download.ps1`: `curl.exe` straight into the HF cache blob, check SHA-256, link with `mklink` |
| LLM export: `ModuleNotFoundError: lm_eval` | LLM requirements live in a bash script | `pip install hydra-core tiktoken sentencepiece lm_eval==0.4.5 blobfile safetensors`, then `install_requirement_helper.py` |
| `--calib_tasks wikitext` crashes in lm_eval (`'<' not supported between int and NoneType`) | lm_eval 0.4.5 vs transformers 5.0rc | `--calib_samples` with JSON from `scripts/make_calib_samples.py` |
| `UnicodeDecodeError: 'charmap'` reading calibration JSON | ExecuTorch opens files without `encoding=` (cp1252 on Windows) | `PYTHONUTF8=1` (in `et-env.ps1`) |
| Whisper export: `please install 'librosa' and 'soundfile'` | audio decoding deps not installed | `pip install soundfile librosa jiwer` |
| Whisper export "fails" with exit 1 | the script ends with `exit("Finish compile_only ...")` | check the log for that line; it's a success |
| `--shared_buffer` on the LLM runner | rpcmem zero-copy path | **crashes on Windows (0xC0000005)**; leave it off |

## NPU runner test matrix

`scripts/test_npu_runner.ps1`. Results in `docs/works.txt` §7 and `results/npu_runner_tests/`.

| Case | Result |
|---|---|
| Real inputs / benchmark / profiling | PASS (exit 0) |
| No input list | PASS (default inputs, smoke test only) |
| Run from another directory | PASS (needs a **writable** working dir for `inference_speed.txt`) |
| Output folder missing | **exit 0 but writes nothing**: create the folder first |
| Missing / wrong-size input, missing model | clear `F ... assert failed`, then `abort()` → `0xC0000409` (intended stop, not corruption) |
| Corrupt `.pte` | `Failed to parse model file`, exit 1 |

## MoE on the Qualcomm NPU

ExecuTorch v1.5.1 has **no MoE model pipeline for the Qualcomm backend**. MoE models
(`qwen3_5_moe`, `gemma4_31b`, `muse_glimmer`) target CUDA and MLX/Metal only. All routing ops do
exist in QNN 2.37 (`TopK`, `Gather`, `ScatterElements`, `OneHot`, `MatMul`), and a toy MoE layer
lowers in ExecuTorch's tests. The design space, difficulty and a phased plan are in
[`docs/moe-pipeline-brainstorm.md`](docs/moe-pipeline-brainstorm.md).

## Running the models

```powershell
scripts\llm\run_qwen3_npu.ps1 -Prompt "Explain NPUs in 3 sentences." -SeqLen 512   # LLM on the NPU
scripts\stt\run_whisper_npu.ps1 [-Model small] [-Wav C:\path\talk.wav]                 # speech-to-text + WER
scripts\llm\export_phi4mini_w8a16.ps1                                                  # Phi-4-mini W8A16 export
```
Paths inside the scripts point at this machine's layout (`C:\et\...`); adjust them to yours.

## File organizer (`organizer/`)

A local file organizer that uses the NPU LLM only as a **classifier**:
`scan -> digest (first + last page only) -> rules -> similarity -> Phi-4-mini (NPU, fixed folder list,
strict JSON) -> plan -> [review] -> move + undo log`.

- `--review on`: asks about unsure files and confirms the plan. `--review off`: sorts everything.
- The model never writes commands; code does every move. Duplicates (sha256) are left in place;
  `undo` restores the last run.
- The LLM's self-reported confidence is trusted only when it agrees with content similarity (a 0.6B test
  model was confidently wrong).
- `python organize.py plan|run|undo --config config.json [--review on|off] [--no-llm]`;
  sandbox test: `python tests\make_sandbox.py <dir>`.
- Runs on native ARM64 Python (needs `pypdf python-docx openpyxl`). Waiting on the Phi-4-mini export.

## Repo layout

```
patches/    0001-0006, git apply onto ExecuTorch v1.5.1
scripts/    et-env.ps1, calibration/HF download helpers, delegation report, test matrix,
            llm/ (Qwen run, Phi exports v1/v2/hybrid, NPU test suite, CPU reference, metrics), stt/ (Whisper)
meetinggraph/  NPU meeting memory app (QUANTIFY.md = presentation numbers)
organizer/  NPU file organizer app
docs/       works.txt (full change log), moe-pipeline-brainstorm.md
results/    runner, benchmark, build and export logs (DeepLab, LLM, STT)
```

## Not included

No model weights, `.pte` files, datasets, or Qualcomm SDK binaries (`QnnHtp.dll`, Hexagon `.so`).
Get QAIRT from Qualcomm under its own license.

## Reproduce (short)

1. Install Git, CMake, **x64** Python 3.12, VS 2022 Build Tools (ARM64 + x64 MSVC, Clang), QAIRT 2.37.0.250724.
2. Enable Developer Mode; `git clone -c core.symlinks=true --branch v1.5.1 --recurse-submodules https://github.com/pytorch/executorch.git`
3. `git apply` the patches, adjust paths in `scripts/et-env.ps1`, then `. .\scripts\et-env.ps1`
4. `python install_executorch.py`; `pip install -r backends\qualcomm\requirements.txt pandas`
5. Clear `CMAKE_GENERATOR*`, then `cmd /c "powershell -File backends\qualcomm\scripts\build.ps1 -Release > build.log 2>&1"`
6. `python -m examples.qualcomm.scripts.deeplab_v3 --build_folder build-x86_64-windows --soc_model SC8380XP --backend htp --compile_only --download`
7. Put the ARM64 runner, backend DLL, QNN ARM64 DLLs and v73 skel in one folder with the `.pte` and run `qnn_executor_runner.exe`.
