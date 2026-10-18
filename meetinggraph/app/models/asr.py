"""Speech-to-text adapters. All local.

transcribe(path) returns segments: {"start": s, "end": s, "text": str, "confidence": 0..1}
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np


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


_TAG = re.compile(r"<\|[^|]*\|>")  # Whisper control tokens: <|en|><|transcribe|>...<|endoftext|>


class ExecuTorchRunnerASR(BaseASR):
    """Speech-to-text with an ExecuTorch Whisper model on the NPU. Two modes:

    - qnn_whisper_runner (set ``runner`` + ``model_path`` + ``tokenizer_path``): the WAV is cut into
      windows, each turned into a log-mel tensor in numpy, and ALL windows go through ONE runner call,
      so the model loads once per recording. Encoder and decoder run on the Hexagon NPU.
    - generic ``command`` with {audio_file} {output_file}: the command writes
      [{"start","end","text"}] JSON to {output_file}, or plain text to stdout.
    """
    name = "executorch_runner"

    def __init__(self, command: list[str], cwd: str = "", timeout_s: int = 600,
                 verified_on_npu: bool = False, label: str = "", runner: str = "", model_path: str = "",
                 tokenizer_path: str = "", window_s: float = 20.0, max_tokens: int = 128,
                 silence_rms: float = 0.003):
        super().__init__(label)
        self.runner, self.model_path, self.tokenizer_path = runner, model_path, tokenizer_path
        if runner:
            for p in (runner, model_path, tokenizer_path):
                if not Path(p).exists():
                    raise ASRError(f"ExecuTorch file not found: {p}")
        elif not command:
            raise ASRError("Set asr.executorch_runner.runner/model_path/tokenizer_path (or command) in config.yaml.")
        # window_s <= 30 (Whisper's input). Shorter windows keep each window's text within max_tokens,
        # the decoder length the model was exported with.
        self.window_s, self.max_tokens, self.silence_rms = min(float(window_s), 30.0), max_tokens, silence_rms
        self.command, self.cwd, self.timeout, self.verified = command, cwd or None, timeout_s, verified_on_npu

    def default_label(self) -> str:
        what = Path(self.model_path).parent.name if self.runner else "ExecuTorch ASR"
        return f"{what} on the NPU (verified)" if self.verified else f"{what} (NPU not yet verified)"

    def _qnn_transcribe(self, path: str) -> list[dict]:
        from .whisper_features import SAMPLE_RATE, log_mel, read_wav, to_16k

        try:
            audio, rate = read_wav(path)
        except (OSError, ValueError) as e:
            raise ASRError(f"Cannot read audio {path}: {e}") from e
        audio = to_16k(audio, rate)
        step = int(self.window_s * SAMPLE_RATE)
        windows = []  # (index, start_s, end_s) of non-silent windows
        with tempfile.TemporaryDirectory(prefix="mg_asr_") as td:
            tdp = Path(td)
            (tdp / "outputs").mkdir()  # the runner does not create it and silently writes nothing
            lines = []
            for i, s in enumerate(range(0, len(audio), step)):
                chunk = audio[s:s + step]
                if len(chunk) == 0 or float(np.sqrt(np.mean(chunk ** 2))) < self.silence_rms:
                    continue  # silent window: Whisper tends to hallucinate text on silence
                k = len(windows)
                log_mel(chunk).tofile(tdp / f"in_{k}.raw")
                lines.append(f"in_{k}.raw")
                windows.append((k, s / SAMPLE_RATE, min(len(audio), s + step) / SAMPLE_RATE))
            if not windows:
                return []
            # LF line endings: the runner does not strip '\r'.
            (tdp / "input_list.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
            args = [self.runner, "--model_path", self.model_path, "--tokenizer_json_path", self.tokenizer_path,
                    "--input_list_path", "input_list.txt", "--seq_len", str(self.max_tokens),
                    "--output_folder_path", "outputs"]
            try:
                p = subprocess.run(args, cwd=td, capture_output=True, text=True, timeout=self.timeout,
                                   encoding="utf-8", errors="replace")
            except (OSError, subprocess.TimeoutExpired) as e:
                raise ASRError(f"NPU Whisper runner failed: {e}") from e
            if p.returncode != 0:
                raise ASRError(f"NPU Whisper runner exited {p.returncode}: {(p.stderr or p.stdout)[-800:]}")
            segs = []
            for k, start, end in windows:
                f = tdp / "outputs" / f"output_{k}.txt"
                text = _TAG.sub("", f.read_text(encoding="utf-8", errors="replace")).strip() if f.exists() else ""
                if text:
                    segs.append({"start": round(start, 2), "end": round(end, 2), "text": text, "confidence": 1.0})
            return segs

    def transcribe(self, path: str) -> list[dict]:
        if self.runner:
            return self._qnn_transcribe(path)
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
                                   e.get("verified_on_npu", False), label, e.get("runner", ""),
                                   e.get("model_path", ""), e.get("tokenizer_path", ""),
                                   e.get("window_s", 20.0), e.get("max_tokens", 128), e.get("silence_rms", 0.003))
    raise ASRError(f"Unknown asr.backend {b!r}")
