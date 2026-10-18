"""Ground truth: original bf16 Phi-4-mini on the CPU, run on the SAME cases and checks as test_phi_npu.py.

  python C:\\et\\llm\\phi_reference_suite.py --save C:\\et\\llm\\results_cpu_bf16.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from test_phi_npu import CASES  # noqa: E402  (same prompts, budgets and checks)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
REPO = "microsoft/Phi-4-mini-instruct"

ap = argparse.ArgumentParser()
ap.add_argument("--save", required=True)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16).eval()
torch.set_num_threads(os.cpu_count())
saved = {}
for name, prompt, gen_budget, check in CASES:
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(enc["input_ids"], attention_mask=enc["attention_mask"],
                             max_new_tokens=gen_budget, do_sample=False)
    ans = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    ok = bool(check(ans))
    saved[name] = {"pass": ok, "answer": ans}
    print(f"=== {name}: {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f}s)\n{ans[:400]}\n", flush=True)
Path(args.save).write_text(json.dumps(saved, indent=1), encoding="utf-8")
print(f"SUMMARY: {sum(v['pass'] for v in saved.values())}/{len(saved)} passed")
