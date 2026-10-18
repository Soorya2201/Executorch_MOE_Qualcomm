"""Loads config.yaml (next to the project root) with safe defaults."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict[str, Any] = {
    "workspace": "./workspace",
    "llm": {
        # heuristic | ollama | executorch_runner
        "backend": "ollama",
        "label": "",  # shown in the UI; empty = backend default
        "max_transcript_chars_per_chunk": 6000,
        "ollama": {"url": "http://127.0.0.1:11434", "model": "qwen2.5:3b-instruct", "timeout_s": 600,
                   "num_ctx": 8192},
        "executorch_runner": {
            # Filled in on the Snapdragon laptop after the model gate.
            # Placeholders: {prompt_file} {prompt} {output_file}
            "command": [],
            "cwd": "",
            "timeout_s": 600,
            "verified_on_npu": False,
        },
    },
    "asr": {
        # none | faster_whisper | executorch_runner
        "backend": "faster_whisper",
        "label": "",
        "faster_whisper": {"model": "base.en", "compute_type": "int8", "device": "cpu"},
        "executorch_runner": {
            # Placeholders: {audio_file} {output_file}
            "command": [],
            "cwd": "",
            "timeout_s": 600,
            "verified_on_npu": False,
        },
    },
    "audio": {"samplerate": 16000, "keep_audio_after_commit": False, "keep_transcript_after_commit": True, "max_minutes": 60,
              "save_transcripts_to": "tests/fixtures", "live_batch_s": 30, "live_notes_every_s": 120},
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else ROOT / "config.yaml"
    data: dict[str, Any] = {}
    if p.exists():
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cfg = _merge(DEFAULTS, data)
    ws = Path(cfg["workspace"])
    if not ws.is_absolute():
        ws = (ROOT / ws).resolve()
    cfg["workspace"] = str(ws)
    return cfg
