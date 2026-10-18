# Phi-4-mini: v2 (W8A16, 16-bit KV) vs v1 (W8A16, 8-bit KV) on the NPU, then the original bf16 on the CPU.
# Runs one after another so the NPU timings aren't disturbed by the CPU reference.
$ErrorActionPreference = "Continue"
$py = "C:\et\organizer\.venv\Scripts\python.exe"
$out = "C:\et\llm\eval"
New-Item -ItemType Directory -Force $out | Out-Null
"start $(Get-Date -Format s)" | Out-File "$out\progress.txt" -Encoding utf8

& $py C:\et\llm\test_phi_npu.py --model_dir C:\et\llm\phi4mini_w8a16kv16_2k --save "$out\results_npu_v2.json" *> "$out\npu_v2.log"
"v2 done $(Get-Date -Format s)" | Out-File "$out\progress.txt" -Append -Encoding utf8
& $py C:\et\llm\test_phi_npu.py --model_dir C:\et\llm\phi4mini_w8a16_2k --save "$out\results_npu_v1.json" *> "$out\npu_v1.log"
"v1 done $(Get-Date -Format s)" | Out-File "$out\progress.txt" -Append -Encoding utf8

. C:\et\et-env.ps1
python C:\et\scripts\pte_delegation_report.py C:\et\llm\phi4mini_w8a16kv16_2k\kv_llama_qnn.pte *> "$out\delegation_v2.txt"
python C:\et\llm\phi_reference_suite.py --save "$out\results_cpu_bf16.json" *> "$out\cpu_bf16.log"
"cpu done $(Get-Date -Format s)" | Out-File "$out\progress.txt" -Append -Encoding utf8
