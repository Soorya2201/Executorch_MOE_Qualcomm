"""Speech-to-text adapters. All local.

transcribe(path) returns segments: {"start": s, "end": s, "text": str, "confidence": 0..1}
"""
from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path


class ASRError(RuntimeError):
    pass


class BaseASR:
    name = "base"
    available = True

    def __init__(self, label: str = ""):
        self._label = label

    @property
    def label(self) -> str:
        return self._label or self.default_label()

    def default_label(self) -> str:
        return self.name

    def transcribe(self, path: str) -> list[dict]:
        raise NotImplementedError


class NoASR(BaseASR):
    name = "none"
    available = False

    def default_label(self) -> str:
        return "No ASR (import or type a transcript)"

    def transcribe(self, path: str) -> list[dict]:
        raise ASRError("No speech recognizer configured (asr.backend: none). "
                       "Import a transcript .txt or set asr.backend in config.yaml.")


class FasterWhisperASR(BaseASR):
    name = "faster_whisper"

    def __init__(self, model: str = "base.en", compute_type: str = "int8", device: str = "cpu", label: str = ""):
        super().__init__(label)
        self.model_name, self.compute_type, self.device = model, compute_type, device
        self._model = None

    def default_label(self) -> str:
        return f"faster-whisper {self.model_name} ({self.device.upper()})"

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise ASRError("faster-whisper is not installed: pip install faster-whisper") from e
            # First load downloads the model once into the Hugging Face cache; after that it runs offline.
            self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
        return self._model

    def transcribe(self, path: str) -> list[dict]:
        model = self._load()
        segs, _info = model.transcribe(path, vad_filter=True, beam_size=1)
        out = []
        for s in segs:
            conf = math.exp(s.avg_logprob) if s.avg_logprob is not None else 0.0
            text = s.text.strip()
            if text:
                out.append({"start": float(s.start), "end": float(s.end), "text": text,
                            "confidence": round(min(1.0, conf), 3)})
        return out


class ExecuTorchRunnerASR(BaseASR):
    """Configured runner. Placeholders: {audio_file} {output_file}.
    The runner (or a small wrapper script you write on the Dell) must write JSON
    [{"start":..,"end":..,"text":..}] to {output_file}, or plain text to stdout.
    """
    name = "executorch_runner"

    def __init__(self, command: list[str], cwd: str = "", timeout_s: int = 600,
                 verified_on_npu: bool = False, label: str = ""):
        super().__init__(label)
        if not command:
            raise ASRError("asr.executorch_runner.command is empty in config.yaml.")
        self.command, self.cwd, self.timeout, self.verified = command, cwd or None, timeout_s, verified_on_npu

    def default_label(self) -> str:
        return "ExecuTorch ASR (NPU, verified)" if self.verified else "ExecuTorch ASR (NPU not yet verified)"

    def transcribe(self, path: str) -> list[dict]:
        with tempfile.TemporaryDirectory(prefix="mg_asr_") as td:
            of = Path(td) / "out.json"
            args = [a.replace("{audio_file}", str(path)).replace("{output_file}", str(of)) for a in self.command]
            try:
                p = subprocess.run(args, cwd=self.cwd, capture_output=True, text=True, timeout=self.timeout,
                                   encoding="utf-8", errors="replace")
            except (OSError, subprocess.TimeoutExpired) as e:
                raise ASRError(f"ASR runner failed: {e}") from e
            if p.returncode != 0:
                raise ASRError(f"ASR runner exited {p.returncode}: {p.stderr[-800:]}")
            if of.exists():
                data = json.loads(of.read_text(encoding="utf-8"))
                return [{"start": float(d.get("start", 0)), "end": float(d.get("end", 0)),
                         "text": str(d.get("text", "")).strip(), "confidence": float(d.get("confidence", 1.0))}
                        for d in data if str(d.get("text", "")).strip()]
            text = p.stdout.strip()
            return [{"start": 0.0, "end": 0.0, "text": text, "confidence": 1.0}] if text else []


def make_asr(cfg: dict) -> BaseASR:
    c = cfg["asr"]
    b = c.get("backend", "none")
    label = c.get("label", "")
    if b == "none":
        return NoASR(label)
    if b == "faster_whisper":
        f = c["faster_whisper"]
        return FasterWhisperASR(f.get("model", "base.en"), f.get("compute_type", "int8"), f.get("device", "cpu"), label)
    if b == "executorch_runner":
        e = c["executorch_runner"]
        return ExecuTorchRunnerASR(e.get("command") or [], e.get("cwd", ""), e.get("timeout_s", 600),
                                   e.get("verified_on_npu", False), label)
    raise ASRError(f"Unknown asr.backend {b!r}")
