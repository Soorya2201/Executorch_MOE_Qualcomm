"""Phi-4-mini (W8A16, NPU) test suite: one qnn_llama_runner call per case, with an automatic check.

  C:\\et\\organizer\\.venv\\Scripts\\python.exe C:\\et\\llm\\test_phi_npu.py [--only name]
"""
import argparse
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

RUNNER = r"C:\et\llm\npu-run\qnn_llama_runner.exe"
MODEL_DIR = r"C:\et\llm\phi4mini_w8a16_2k"  # overridden by --model_dir
PTE = "kv_llama_qnn.pte"                    # --pte (hybrid exports: hybrid_llama_qnn.pte)
EVAL_MODE = "0"                             # --eval_mode: 0 = kv, 1 = hybrid


def has(*words):
    return lambda a: all(w.lower() in a.lower() for w in words)


def json_with(**expected):
    def check(a):
        m = re.search(r"\{.*\}", a, re.S)
        if not m:
            return False
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return False
        return all(str(obj.get(k, "")).strip().lower() == str(v).lower() for k, v in expected.items())
    return check


def _strict_actions(a):
    """Valid JSON; Arun->Friday and Lee->Wednesday present; no task invented for Priya."""
    m = re.search(r"\[.*\]", a, re.S)
    if not m:
        return False
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return False
    owners = [(str(i.get("owner", "")).lower(), str(i.get("due", "")).lower()) for i in items if isinstance(i, dict)]
    has_arun = any("arun" in o and "friday" in d for o, d in owners)
    has_lee = any("lee" in o and "wednesday" in d for o, d in owners)
    invented = any("priya" in o for o, _ in owners)
    return has_arun and has_lee and not invented


MEETING = """[00:01] Priya: Let's finalize the launch. Marketing copy is still pending.
[00:03] Arun: I'll finish the marketing copy by Friday.
[00:05] Priya: Good. We agreed to move the launch date to November 12.
[00:07] Lee: I can set up the demo laptop with the NPU models by Wednesday.
[00:09] Priya: Budget stays at 20k. Any open questions?
[00:10] Arun: Do we need legal review for the new privacy text? Nobody knows yet."""

CASES = [
    ("reasoning", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
     "How much does the ball cost? Answer with the amount only.", 120, has("0.05")),
    ("reasoning_cot", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
     "How much does the ball cost? Think step by step, then give the final answer on the last line.", 300,
     lambda a: "0.05" in a.strip().splitlines()[-1] if a.strip() else False),
    ("math", "What is 17 * 23? Reply with just the number.", 60, has("391")),
    ("json_strict", "Classify this file into exactly one folder from [Finance, School, Career, Personal]. "
     "File: 'Offer letter from Acme Corp, salary and start date'. Reply with only JSON: "
     '{"folder": "<name>", "confidence": <0-1>}', 120, json_with(folder="Career")),
    ("coding", "Write a Python function is_palindrome(s) that ignores case and non-letters. Code only.", 220,
     lambda a: "def is_palindrome" in a and ("lower" in a or "casefold" in a)),
    ("bash", "Give one PowerShell command that lists all .pdf files under C:\\Docs recursively. Command only.",
     100, lambda a: "get-childitem" in a.lower() and "-recurse" in a.lower() and "pdf" in a.lower()),
    ("summarize", "Summarize in one sentence: The NPU is a processor specialised for neural-network math. "
     "It runs AI models faster and with far less power than the CPU, which keeps laptops cool and "
     "extends battery life during AI tasks.", 120, lambda a: ("power" in a.lower() or "battery" in a.lower() or "efficien" in a.lower())),
    ("meeting_actions", "From this meeting transcript, list every action item as JSON array of objects "
     '{"owner","task","due"}. Only JSON.\n' + MEETING, 300, lambda a: _strict_actions(a)),
    ("meeting_decisions", "From this transcript, what decisions were made? One bullet per decision.\n" + MEETING,
     200, lambda a: ("november 12" in a.lower() or "nov 12" in a.lower()) and ("20k" in a.lower() or "20,000" in a)),
    ("organizer_hard", "Choose ONE folder from [Finance, School, Research, Code, Career, Personal]. "
     "File 'notes.txt' first page: 'Ablation: 16a4w vs 16a8w perplexity table, hypothesis and method'. "
     'Reply with only JSON {"folder": "<name>"}', 100, json_with(folder="Research")),
]


def run(prompt, seq_len):
    with tempfile.TemporaryDirectory() as tmp:
        out, speed = Path(tmp) / "out.txt", Path(tmp) / "speed.txt"
        t0 = time.time()
        r = subprocess.run(
            [RUNNER, "--model_path", str(Path(MODEL_DIR) / PTE),
             "--tokenizer_path", str(Path(MODEL_DIR) / "tokenizer.json"), "--decoder_model_version",
             "phi_4_mini", "--eval_mode", EVAL_MODE, "--seq_len", str(seq_len), "--temperature", "0",
             "--prompt", prompt, "--output_path", str(out), "--performance_output_path", str(speed)],
            cwd=tmp, capture_output=True, text=True, timeout=600)
        wall = time.time() - t0
        text = out.read_text(encoding="utf-8", errors="replace") if out.exists() else ""
        log = r.stdout + r.stderr
    ans = text.split("<|assistant|>")[-1].replace("<|end|>", "").strip()
    gen = re.search(r"Generated (\d+) tokens:\s*([\d.]+).*?Rate:\s*([\d.]+)", log)
    pre = re.search(r"Prompt evaluation:\s*([\d.]+).*?Rate:\s*([\d.]+)", log)
    return r.returncode, ans, wall, gen, pre


def main():
    global MODEL_DIR, PTE, EVAL_MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--model_dir", default=MODEL_DIR)
    ap.add_argument("--pte", default=None, help="file in --model_dir (default: hybrid_llama_qnn.pte if present)")
    ap.add_argument("--save", help="write {case: {pass, answer}} JSON here")
    args = ap.parse_args()
    MODEL_DIR = args.model_dir
    PTE = args.pte or ("hybrid_llama_qnn.pte" if (Path(MODEL_DIR) / "hybrid_llama_qnn.pte").exists() else PTE)
    EVAL_MODE = "1" if PTE.startswith("hybrid") else "0"
    print(f"model: {Path(MODEL_DIR) / PTE}  (eval_mode {EVAL_MODE})")
    results, saved = [], {}
    for name, prompt, gen_budget, check in CASES:
        if args.only and args.only != name:
            continue
        seq_len = min(2048, int(len(prompt) / 3.5) + gen_budget + 64)
        code, ans, wall, gen, pre = run(prompt, seq_len)
        ok = code == 0 and check(ans)
        tps = float(gen.group(3)) if gen else 0.0
        results.append((name, ok, wall, tps))
        saved[name] = {"pass": ok, "answer": ans, "tok_per_s": tps}
        print(f"\n=== {name}: {'PASS' if ok else 'FAIL'}  exit={code}  wall={wall:.1f}s  "
              f"gen={gen.group(1) if gen else '?'} tok @ {tps:.1f} tok/s  "
              f"prompt={pre.group(2) if pre else '?'} tok/s")
        print(ans[:700])
    passed = sum(1 for r in results if r[1])
    print(f"\nSUMMARY: {passed}/{len(results)} passed; mean decode "
          f"{sum(r[3] for r in results) / max(1, len(results)):.1f} tok/s; total {sum(r[2] for r in results):.0f}s")
    if args.save:
        Path(args.save).write_text(json.dumps(saved, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
