"""Two-track recorder: microphone (you) and speaker loopback (remote participants).

Uses the `soundcard` library (WASAPI on Windows). Each track is written to its own
16 kHz mono 16-bit WAV in the session's staging folder. Both tracks are stamped
against the same monotonic clock so they can be aligned before transcription.
"""
from __future__ import annotations

import sys
import threading
import time
import wave
from pathlib import Path

# IMPORTANT (Windows): soundcard initialises COM as multithreaded on whatever thread
# imports or calls it. Qt needs the GUI thread single-threaded (OLE) for file dialogs,
# clipboard and drag-and-drop. So soundcard is imported and used ONLY on worker
# threads, never on the GUI thread.
_sc_mod = None
_np_mod = None
_import_error = ""
_import_lock = threading.Lock()


def _com_init() -> None:
    """Initialise COM (multithreaded) on the current worker thread. Call only AFTER _libs()."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.ole32.CoInitializeEx(None, 0)   # COINIT_MULTITHREADED, this thread only
        except Exception:
            pass


def _libs():
    """Import numpy + soundcard lazily (call only from a worker thread)."""
    global _sc_mod, _np_mod, _import_error
    with _import_lock:
        if _sc_mod is None and not _import_error:
            try:
                import numpy
                import soundcard
                _np_mod, _sc_mod = numpy, soundcard
            except Exception as e:  # ImportError, or no audio backend
                _import_error = str(e)
    if _import_error:
        raise RuntimeError(_import_error)
    return _np_mod, _sc_mod


def _on_audio_thread(fn, timeout: float = 15.0):
    """Run fn() on a short-lived worker thread with its own COM apartment; return or raise its result."""
    box: dict = {}

    def run():
        try:
            _libs()        # import first: soundcard initialises COM itself and rejects a pre-initialised thread
            _com_init()    # no-op if soundcard just did it; needed when it was imported on another thread
            box["ok"] = fn()
        except Exception as e:
            box["err"] = e

    t = threading.Thread(target=run, daemon=True, name="audio-query")
    t.start()
    t.join(timeout)
    if "err" in box:
        raise box["err"]
    if "ok" not in box:
        raise RuntimeError("Audio device query timed out")
    return box["ok"]

CAPTURE_RATE = 48000      # record at the device's native-ish rate...
OUT_RATE = 16000          # ...and decimate to 16 kHz for ASR
BLOCK = 4800              # 0.1 s at 48 kHz
SILENCE_RMS = 0.003
SILENCE_WARN_S = 15


def list_devices() -> dict:
    def query():
        _np, sc = _libs()
        mics = [{"id": m.id, "name": m.name} for m in sc.all_microphones(include_loopback=False)]
        spk = [{"id": s.id, "name": s.name} for s in sc.all_speakers()]
        dm, ds = sc.default_microphone(), sc.default_speaker()
        return {"mics": mics, "speakers": spk, "default_mic": dm.id if dm else None,
                "default_speaker": ds.id if ds else None, "error": ""}
    try:
        return _on_audio_thread(query)
    except Exception as e:
        return {"mics": [], "speakers": [], "error": str(e)}


class _Track(threading.Thread):
    def __init__(self, name: str, device_id: str, loopback: bool, path: Path, t0: float, stop: threading.Event,
                 paused: threading.Event | None = None):
        super().__init__(daemon=True, name=f"rec-{name}")
        self.track, self.device_id, self.loopback = name, device_id, loopback
        self.path, self.t0, self.stop_evt = path, t0, stop
        self.paused = paused or threading.Event()
        self.opened = threading.Event()
        self.level = 0.0
        self.offset: float | None = None
        self.error = ""
        self.frames = 0
        self.silent_since: float | None = None
        self._live: list = []                 # 16 kHz float32 blocks not yet taken by the live transcriber
        self._live_lock = threading.Lock()

    def drain(self):
        """Return (and forget) the 16 kHz mono float32 audio recorded since the last call."""
        np, _ = _libs()
        with self._live_lock:
            blocks, self._live = self._live, []
        return np.concatenate(blocks).astype(np.float32) if blocks else np.zeros(0, dtype=np.float32)

    def run(self):
        try:
            np, sc = _libs()   # import before any COM init (see _on_audio_thread)
            _com_init()
            # On Windows a speaker can be opened as a loopback "microphone".
            device = sc.get_microphone(self.device_id, include_loopback=self.loopback)
            with wave.open(str(self.path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(OUT_RATE)
                with device.recorder(samplerate=CAPTURE_RATE, channels=1, blocksize=BLOCK) as rec:
                    self.opened.set()
                    while not self.stop_evt.is_set():
                        data = rec.record(numframes=BLOCK)
                        if self.paused.is_set():       # keep draining the device, but write nothing
                            self.level = 0.0
                            self.silent_since = None
                            continue
                        if self.offset is None:
                            self.offset = time.monotonic() - self.t0
                        mono = data.mean(axis=1) if data.ndim > 1 else data
                        n = len(mono) - len(mono) % 3
                        down = mono[:n].reshape(-1, 3).mean(axis=1)        # crude low-pass + decimate
                        rms = float(np.sqrt(np.mean(down ** 2))) if len(down) else 0.0
                        self.level = min(1.0, rms * 8)
                        now = time.monotonic()
                        if rms < SILENCE_RMS:
                            self.silent_since = self.silent_since or now
                        else:
                            self.silent_since = None
                        pcm = (np.clip(down, -1, 1) * 32767).astype("<i2").tobytes()
                        w.writeframes(pcm)
                        self.frames += len(down)
                        with self._live_lock:
                            self._live.append(np.clip(down, -1, 1).astype(np.float32))
        except Exception as e:
            self.error = f"{self.track}: {e}"
        finally:
            self.opened.set()

    def silent_for(self) -> float:
        return time.monotonic() - self.silent_since if self.silent_since else 0.0


class DualRecorder:
    def __init__(self, mic_id: str | None, speaker_id: str | None, mic_path: Path, loop_path: Path):
        self.mic_id, self.speaker_id = mic_id, speaker_id
        self.mic_path, self.loop_path = mic_path, loop_path
        self._stop = threading.Event()
        self._paused = threading.Event()
        self.tracks: dict[str, _Track] = {}
        self.started_at = 0.0
        self._paused_total = 0.0
        self._paused_at: float | None = None

    def start(self) -> list[str]:
        """Start both tracks. Returns warnings (e.g. loopback unavailable)."""
        warnings = []
        t0 = time.monotonic()
        self.started_at = t0
        if self.mic_id is not None:
            self.tracks["mic"] = _Track("mic", self.mic_id, False, self.mic_path, t0, self._stop, self._paused)
        if self.speaker_id is not None:
            self.tracks["loopback"] = _Track("loopback", self.speaker_id, True, self.loop_path, t0, self._stop, self._paused)
        if not self.tracks:
            raise RuntimeError("Choose a microphone and/or an output device first.")
        for t in self.tracks.values():
            t.start()
        for t in self.tracks.values():
            t.opened.wait(8)
        for k, t in list(self.tracks.items()):
            if t.error:
                if k == "loopback":
                    warnings.append(f"System-audio loopback unavailable ({t.error}). Remote voices will not be "
                                    "captured; use audio import if you need them.")
                else:
                    warnings.append(f"Microphone unavailable: {t.error}")
                del self.tracks[k]
        if not self.tracks:
            raise RuntimeError("No audio device could be opened. " + " ".join(warnings))
        return warnings

    def levels(self) -> dict[str, float]:
        return {k: t.level for k, t in self.tracks.items()}

    def health(self) -> list[str]:
        out = []
        for k, t in self.tracks.items():
            if t.error:
                out.append(f"{k} stopped: {t.error}")
            elif t.silent_for() > SILENCE_WARN_S:
                out.append(f"{k}: silent for {int(t.silent_for())} s (muted or wrong device?)")
        return out

    @property
    def is_paused(self) -> bool:
        return self._paused.is_set()

    def pause(self) -> None:
        if not self._paused.is_set():
            self._paused_at = time.monotonic()
            self._paused.set()

    def resume(self) -> None:
        if self._paused.is_set():
            self._paused_total += time.monotonic() - (self._paused_at or time.monotonic())
            self._paused_at = None
            self._paused.clear()

    def elapsed(self) -> float:
        """Recorded time, not counting pauses."""
        if not self.started_at:
            return 0.0
        paused = self._paused_total + (time.monotonic() - self._paused_at if self._paused_at else 0.0)
        return time.monotonic() - self.started_at - paused

    def stop(self) -> tuple[dict[str, float], list[str]]:
        """Stop and return (offsets in seconds per track, health notes)."""
        self._stop.set()
        for t in self.tracks.values():
            t.join(timeout=3)
        offsets, notes = {}, []
        for k, t in self.tracks.items():
            offsets[k] = round(t.offset or 0.0, 3)
            if t.error:
                notes.append(t.error)
            if t.frames == 0:
                notes.append(f"{k}: no audio frames captured")
        return offsets, notes
