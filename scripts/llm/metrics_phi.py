"""Metrics table: Phi-4-mini NPU exports vs the original bf16 model on the CPU (same 10 cases, same checks).

  python C:\\et\\llm\\metrics_phi.py C:\\et\\llm\\eval
"""
import difflib
import json
import sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\et\llm\eval")
names = [("cpu_bf16", "CPU bf16 (original)"), ("npu_v1", "NPU v1 (W8A16, KV 8-bit)"),
         ("npu_v2", "NPU v2 (W8A16, KV 16-bit)")]
runs = {k: json.loads((d / f"results_{k}.json").read_text(encoding="utf-8"))
        for k, _ in names if (d / f"results_{k}.json").exists()}
ref = runs.get("cpu_bf16")
cases = list(next(iter(runs.values())))


def sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.split(), b.split()).ratio()


print(f"{'case':16s}" + "".join(f"{k:>12s}" for k in runs))
for c in cases:
    print(f"{c:16s}" + "".join(f"{('PASS' if r[c]['pass'] else 'fail'):>12s}" for r in runs.values()))
print()
for k, label in names:
    if k not in runs:
        continue
    r = runs[k]
    line = f"{label:28s} pass {sum(v['pass'] for v in r.values()):2d}/{len(r)}"
    tps = [v.get("tok_per_s", 0) for v in r.values() if v.get("tok_per_s")]
    if tps:
        line += f"   decode {sum(tps) / len(tps):5.1f} tok/s"
    if ref and k != "cpu_bf16":
        agree = sum(r[c]["pass"] == ref[c]["pass"] for c in cases)
        exact = sum(r[c]["answer"].strip() == ref[c]["answer"].strip() for c in cases)
        s = sum(sim(r[c]["answer"], ref[c]["answer"]) for c in cases) / len(cases)
        line += f"   same verdict as CPU {agree}/{len(cases)}   identical text {exact}/{len(cases)}   word overlap {s:.0%}"
    print(line)
if ref:
    for k in ("npu_v1", "npu_v2"):
        if k in runs:
            diff = [c for c in cases if runs[k][c]["pass"] != ref[c]["pass"]]
            print(f"\n{k} differs from CPU on: {', '.join(diff) or 'none'}")
