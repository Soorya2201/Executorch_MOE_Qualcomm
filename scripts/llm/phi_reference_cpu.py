"""Reference: original (unquantized, bf16) Phi-4-mini on the CPU, greedy, same prompts as the NPU test.
Tells apart model limitations from quantization damage."""
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_HUB_OFFLINE", "1")
REPO = "microsoft/Phi-4-mini-instruct"
PROMPTS = {
    "reasoning_cot": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
                     "How much does the ball cost? Think step by step, then give the final answer on the last line.",
    "reasoning": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
                 "How much does the ball cost? Answer with the amount only.",
}

tok = AutoTokenizer.from_pretrained(REPO)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.bfloat16).eval()
print(f"loaded in {time.time() - t0:.0f}s", flush=True)
torch.set_num_threads(os.cpu_count())
for name, prompt in PROMPTS.items():
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)  # transformers 5 returns a dict
    ids = enc["input_ids"]
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, attention_mask=enc["attention_mask"], max_new_tokens=260, do_sample=False)
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    print(f"\n=== {name} (bf16 CPU, {time.time() - t0:.0f}s) ===\n{text}", flush=True)
