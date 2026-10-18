"""Side-by-side: original bf16 (CPU) vs NPU exports, per test case.

  python C:\\et\\llm\\compare_phi.py results_cpu_bf16.json results_npu_v1.json results_npu_v2.json
"""
import json
import sys
from pathlib import Path

runs = [(Path(p).stem.replace("results_", ""), json.loads(Path(p).read_text(encoding="utf-8")))
        for p in sys.argv[1:]]
cases = list(runs[0][1])
print(f"{'case':20s}" + "".join(f"{name:>16s}" for name, _ in runs))
for c in cases:
    print(f"{c:20s}" + "".join(f"{('PASS' if r.get(c, {}).get('pass') else 'FAIL'):>16s}" for _, r in runs))
print(f"{'TOTAL':20s}" + "".join(f"{sum(v['pass'] for v in r.values()):>13d}/{len(r):<2d}" for _, r in runs))

ref_name, ref = runs[0]
for name, r in runs[1:]:
    diff = [c for c in cases if r.get(c, {}).get("pass") != ref[c]["pass"]]
    print(f"\n{name} disagrees with {ref_name} on {len(diff)} case(s): {', '.join(diff) or 'none'}")
    for c in diff:
        print(f"  [{c}] {ref_name}: {ref[c]['answer'][:160]!r}\n  [{c}] {name}: {r[c]['answer'][:160]!r}")
