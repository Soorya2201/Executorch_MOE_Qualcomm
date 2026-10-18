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

## Windows-on-ARM build gotchas (all solved)

| Symptom | Cause | Fix |
|---|---|---|
| `install_executorch.py`: `unknown type name 'uint'` in `vec128_uint_aarch64.h` | VS generator defaults to ARM64 on an ARM64 host; the Python wheel must be x64 | `CMAKE_GENERATOR="Visual Studio 17 2022"` + `CMAKE_GENERATOR_PLATFORM=x64` (the platform setting is ignored without the generator) |
| `build.ps1` dies on a CMake deprecation warning | PowerShell 5.1 turns native stderr into terminating errors when piped (`*>&1 \| Tee-Object`) | run it via `cmd /c "... > log 2>&1"` |
| Export: `ModuleNotFoundError: pydot` | `install_executorch.py` skips `backends/qualcomm/requirements.txt` | `pip install -r backends/qualcomm/requirements.txt pandas` |
| Export: `Failed to initialize QNN backend for kHtpBackend` | `envsetup.ps1` adds the **ARM64** QNN DLLs (host shell arch), but export runs in **x64** Python | put `lib\x86_64-windows-msvc` first on PATH (see `scripts/et-env.ps1`) |
| QAIRT zip: "End of Central Directory not found" | download truncated mid-stream | resume with `curl.exe -C -` |
| HF `snapshot_download` writes 0-byte weights / stalls | Xet transfer backend issue | `curl.exe` straight into the HF cache blob, check SHA-256, link with `mklink` |

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

## Repo layout

```
patches/   runner fix (git apply onto ExecuTorch v1.5.1)
scripts/   et-env.ps1 (environment), input prep, scoring, ETDump inspection, test matrix,
           page-file registry setter (needs admin; reboot to apply)
docs/      works.txt (full change log), moe-pipeline-brainstorm.md
results/   runner, benchmark, build and export logs
```

## Not included

No model weights, `.pte` files, datasets, or Qualcomm SDK binaries (`QnnHtp.dll`, Hexagon `.so`).
Get QAIRT from Qualcomm under its own license.

## Reproduce (short)

1. Install Git, CMake, **x64** Python 3.12, VS 2022 Build Tools (ARM64 + x64 MSVC, Clang), QAIRT 2.37.0.250724.
2. Enable Developer Mode; `git clone -c core.symlinks=true --branch v1.5.1 --recurse-submodules https://github.com/pytorch/executorch.git`
3. `git apply` the patch, adjust paths in `scripts/et-env.ps1`, then `. .\scripts\et-env.ps1`
4. `python install_executorch.py`; `pip install -r backends\qualcomm\requirements.txt pandas`
5. Clear `CMAKE_GENERATOR*`, then `cmd /c "powershell -File backends\qualcomm\scripts\build.ps1 -Release > build.log 2>&1"`
6. `python -m examples.qualcomm.scripts.deeplab_v3 --build_folder build-x86_64-windows --soc_model SC8380XP --backend htp --compile_only --download`
7. Put the ARM64 runner, backend DLL, QNN ARM64 DLLs and v73 skel in one folder with the `.pte` and run `qnn_executor_runner.exe`.
