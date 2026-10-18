"""LLM tier: Phi-4-mini-instruct on the Hexagon NPU via ExecuTorch's qnn_llama_runner.exe.

The model only CLASSIFIES: it picks one folder name from a fixed list and replies in JSON.
It never produces commands; the caller validates the answer against the allowed list.
"""
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional

from .config import Destination, LLMConfig
from .digest import FileDigest
from .route import Decision

CHARS_PER_TOKEN = 3.5  # conservative estimate, so the prompt never overruns the compiled context
_JSON = re.compile(r"\{[^{}]*\"folder\"[^{}]*\}", re.S)


def build_prompt(d: FileDigest, dests: Dict[str, Destination], cfg: LLMConfig) -> str:
    folders = "\n".join(f"- {n}: {x.description}" for n, x in dests.items())
    head = (
        "You organise files into folders. Choose exactly ONE folder from this list:\n"
        f"{folders}\n\n"
        "File to organise:\n"
    )
    tail = (
        "\n\nAnswer with only this JSON, nothing else:\n"
        '{"folder": "<one name from the list>", "confidence": <number 0 to 1>, "reason": "<max 12 words>"}'
    )
    if cfg.decoder_model_version.startswith("qwen3"):
        tail += " /no_think"  # Qwen3 otherwise spends the reply budget in <think>; Phi-4-mini has no such mode

    # Fit first+last page into what's left of the compiled context after the reply budget.
    prompt_tokens = cfg.max_seq_len - cfg.max_new_tokens - 64  # 64: chat-template overhead
    budget_chars = int(prompt_tokens * CHARS_PER_TOKEN) - len(head) - len(tail)
    return head + d.as_prompt(max(200, budget_chars)) + tail


def parse_answer(text: str, dests: Dict[str, Destination]) -> Optional[Decision]:
    for m in reversed(_JSON.findall(text)):  # the reply comes last; the prompt also contains a template
        try:
            obj = json.loads(m)
        except json.JSONDecodeError:
            continue
        folder = str(obj.get("folder", "")).strip()
        match = next((n for n in dests if n.lower() == folder.lower()), None)
        if match is None:
            continue  # template or invented folder: not a valid answer
        try:
            conf = float(obj.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        return Decision(match, max(0.0, min(1.0, conf)), "llm", str(obj.get("reason", ""))[:120])
    return None


class NpuLLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        for p in (cfg.runner, cfg.model, cfg.tokenizer):
            if not p.exists():
                raise FileNotFoundError(f"LLM file missing: {p}")

    def classify(self, d: FileDigest, dests: Dict[str, Destination]) -> Optional[Decision]:
        prompt = build_prompt(d, dests, self.cfg)
        seq_len = min(self.cfg.max_seq_len, int(len(prompt) / CHARS_PER_TOKEN) + self.cfg.max_new_tokens + 64)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.txt"
            cmd = [
                str(self.cfg.runner),
                "--model_path", str(self.cfg.model),
                "--tokenizer_path", str(self.cfg.tokenizer),
                "--decoder_model_version", self.cfg.decoder_model_version,
                "--eval_mode", "0",
                "--seq_len", str(seq_len),
                "--temperature", "0",
                "--prompt", prompt,
                "--output_path", str(out),
                "--performance_output_path", str(Path(tmp) / "speed.txt"),
            ]
            # cwd=tmp: the runner writes side files into its working directory.
            r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=300)
            if r.returncode != 0 or not out.exists():
                tail = (r.stderr or r.stdout)[-400:]
                raise RuntimeError(f"NPU runner failed (exit {r.returncode}): {tail}")
            text = out.read_text(encoding="utf-8", errors="replace")
        debug_dir = os.environ.get("NPU_ORGANIZER_DEBUG")
        if debug_dir:  # keep the raw runner output to inspect what the model said
            Path(debug_dir).mkdir(parents=True, exist_ok=True)
            (Path(debug_dir) / f"{d.name}.llm.txt").write_text(text, encoding="utf-8")
        # Only look at what the model generated, i.e. after the prompt text.
        return parse_answer(text.split(prompt[-60:])[-1], dests)
