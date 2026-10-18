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

    def transcribe_audio(self, audio) -> list[dict]:
        """Transcribe 16 kHz mono float32 samples (used for live 30-second batches)."""
        import wave
        import numpy as np
        with tempfile.TemporaryDirectory(prefix="mg_live_") as td:
            p = Path(td) / "batch.wav"
            with wave.open(str(p), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
            return self.transcribe(str(p))


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
        if not self.available:
            return "Speech recognition not installed (pip install -r requirements-asr.txt)"
        return f"Whisper {self.model_name} via faster-whisper ({self.device.upper()})"

    @property
    def available(self) -> bool:
        import importlib.util
        return importlib.util.find_spec("faster_whisper") is not None

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
        return self.transcribe_audio(load_audio_16k(path))

    def transcribe_audio(self, audio) -> list[dict]:
        model = self._load()
        if audio.size == 0:
            return []
        segs, _info = model.transcribe(audio, vad_filter=True, beam_size=1)
        out = []
        for s in segs:
            conf = math.exp(s.avg_logprob) if s.avg_logprob is not None else 0.0
            text = s.text.strip()
            if text:
                out.append({"start": float(s.start), "end": float(s.end), "text": text,
                            "confidence": round(min(1.0, conf), 3)})
        return out


def _resample(x, rate: int, target: int = 16000):
    import numpy as np
    if rate == target or x.size == 0:
        return x.astype(np.float32)
    n = int(round(x.size * target / rate))
    return np.interp(np.linspace(0, x.size - 1, n), np.arange(x.size), x).astype(np.float32)


def _read_wav(path: str):
    """Plain WAV reader (our own recordings): no PyAV needed."""
    import wave
    import numpy as np
    with wave.open(str(path), "rb") as w:
        ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ASRError(f"Unsupported WAV sample width: {width * 8} bit")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return _resample(x, rate)


def _read_with_av(path: str):
    """Other formats (mp3, m4a, ...) through PyAV. Works with old and new PyAV versions
    (new PyAV removed the 'metadata_errors' argument that faster-whisper's own decoder passes)."""
    import numpy as np
    try:
        import av
    except ImportError as e:
        raise ASRError("Can't read this audio format (PyAV missing). Use a .wav file.") from e
    chunks = []
    with av.open(str(path), mode="r") as container:
        stream = container.streams.audio[0]
        res = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
        for frame in container.decode(stream):
            for f in (res.resample(frame) or []):
                chunks.append(f.to_ndarray().reshape(-1))
        for f in (res.resample(None) or []):
            chunks.append(f.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return (np.concatenate(chunks).astype(np.float32) / 32768.0)


def load_audio_16k(path: str):
    """Return mono float32 audio at 16 kHz."""
    if str(path).lower().endswith(".wav"):
        try:
            return _read_wav(path)
        except (ASRError, EOFError, Exception) as e:  # odd WAV (e.g. float/extensible): let PyAV try
            try:
                return _read_with_av(path)
            except Exception:
                raise ASRError(f"Could not read {Path(path).name}: {e}") from e
    return _read_with_av(path)


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
