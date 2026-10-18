"""Live transcript + live notes while a meeting is being recorded.

Every `batch_s` seconds (default 30) the new audio of each track is taken from the recorder, cut at the
quietest moment near the end (so words are not split between batches), transcribed, and APPENDED to the
transcript, like a log. A second worker turns every `notes_every_s` seconds (default 120) of new transcript
into notes (summary part + features) while recording continues. When the meeting stops, only the last few
seconds of audio and the last notes part are left to do.

The full WAV files are still written, so if anything goes wrong the old "transcribe everything after Stop"
path is used instead.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback

import numpy as np

from . import extract as ex
from .transcript import merge_tracks, to_text

RATE = 16000


def quiet_cut(x: np.ndarray, search_s: float = 3.0, win_s: float = 0.2) -> int:
    """Where to end this batch: the quietest moment in the last few seconds (a pause between words).
    Leftover audio after it is carried into the next batch. No clear pause -> take everything."""
    n = len(x)
    win = int(win_s * RATE)
    lo = max(n - int(search_s * RATE), (2 * n) // 3)
    if n - lo < 2 * win:
        return n
    energies = [(float(np.mean(x[i:i + win] ** 2)), i) for i in range(lo, n - win + 1, win // 2)]
    quiet, i = min(energies, key=lambda e: (e[0], -e[1]))
    loud = float(np.median([e for e, _ in energies]))
    if loud > 0 and quiet > 0.25 * loud:
        return n
    return i + win // 2


class LiveSession:
    def __init__(self, sm, sid: str, recorder, batch_s: float = 30.0, notes_every_s: float = 120.0,
                 existing_titles: list[str] | None = None, threads: bool = True):
        self.sm, self.sid, self.rec = sm, sid, recorder
        self.batch_s, self.notes_every_s = batch_s, notes_every_s
        self.existing = list(existing_titles or [])
        self.segments: list[dict] = []
        self.parts: list[dict] = []
        self.failed_ranges: list[tuple[int, int]] = []
        self.error = ""                         # a fatal live error: fall back to the after-Stop path
        self._buf: dict[str, np.ndarray] = {}
        self._base: dict[str, float | None] = {}
        self._notes_from = 0                     # first segment index not yet sent for notes
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.stats = {"lines": 0, "audio_s": 0.0, "notes_done": 0, "notes_waiting": 0, "last": "",
                      "busy": "", "batches": 0}
        self._threads = []
        if threads:
            for target, name in ((self._asr_loop, "live-asr"), (self._notes_loop, "live-notes")):
                t = threading.Thread(target=target, daemon=True, name=name)
                t.start()
                self._threads.append(t)

    # ---------- transcription ----------
    def _asr_loop(self):
        while not self._stop.wait(self.batch_s):
            try:
                self.step()
            except Exception as e:  # keep the WAVs; the after-Stop path takes over
                self.error = f"Live transcript stopped: {e}"
                self.stats["busy"] = "live transcript stopped (will finish after Stop)"
                traceback.print_exc()
                return

    def step(self, final: bool = False) -> int:
        """Transcribe the audio recorded since the last step; append the lines. Returns lines added."""
        with self._lock:
            per_track = {}
            for key, track in list(self.rec.tracks.items()):
                new = track.drain()
                if self._base.get(key) is None:
                    if not new.size:
                        continue
                    self._base[key] = float(track.offset or 0.0)
                buf = np.concatenate([self._buf.get(key, np.zeros(0, np.float32)), new])
                cut = len(buf) if final else quiet_cut(buf)
                piece, self._buf[key] = buf[:cut], buf[cut:]
                base = self._base[key]
                self._base[key] = base + cut / RATE
                if piece.size < RATE // 2 or float(np.sqrt(np.mean(piece ** 2))) < 0.002:
                    continue                    # under half a second, or silence: nothing to transcribe
                self.stats["busy"] = "writing transcript"
                segs = self.sm.asr.transcribe_audio(piece)
                per_track[key] = [dict(s, start=s["start"] + base, end=s["end"] + base) for s in segs]
            self.stats["busy"] = ""
            merged = merge_tracks(per_track.get("mic", []), per_track.get("loopback", []))
            self.stats["batches"] += 1
            self.stats["audio_s"] = max([b for b in self._base.values() if b is not None] or [0.0])
            if not merged:
                self._maybe_queue_notes(final)
                return 0
            merged.sort(key=lambda s: s["start"])
            self.segments += merged
            self.sm.append_live_transcript(self.sid, to_text(merged))
            self.stats["lines"] = len(self.segments)
            self.stats["last"] = merged[-1]["text"]
            self._maybe_queue_notes(final)
            self.sm.live_progress(self.sid, self.stats)
            return len(merged)

    # ---------- notes ----------
    def _maybe_queue_notes(self, final: bool):
        if self._notes_from >= len(self.segments):
            return
        span = self.segments[-1]["end"] - self.segments[self._notes_from]["start"]
        if final or (self.notes_every_s > 0 and span >= self.notes_every_s):
            self._q.put((self._notes_from, len(self.segments)))
            self._notes_from = len(self.segments)
            self.stats["notes_waiting"] = self._q.qsize()

    def _notes_loop(self):
        while True:
            job = self._q.get()
            if job is None:
                return
            self.notes_for(*job)

    def notes_for(self, a: int, b: int) -> None:
        self.stats["notes_waiting"] = self._q.qsize()
        titles = self.existing + [f["title"] for p in self.parts for f in p["feature_mentions"]]
        try:
            parts = ex.extract_part(self.sm.llm, self.segments[a:b], titles,
                                    self.sm.cfg["llm"].get("max_transcript_chars_per_chunk", 6000))
            self.parts += parts
            self.stats["notes_done"] += 1
            self.stats["notes_waiting"] = self._q.qsize()
            self.sm.live_progress(self.sid, self.stats)
        except Exception as e:  # retried once when the meeting stops
            self.failed_ranges.append((a, b))
            self.stats["last_notes_error"] = str(e)

    # ---------- stop ----------
    def finish(self) -> tuple[list[dict], list[dict]]:
        """Call after the recorder has stopped: transcribe what is left, finish the notes, return
        (all transcript segments, notes parts)."""
        self._stop.set()
        if self._threads:
            self._threads[0].join(timeout=600)
        if not self.error:
            self.step(final=True)
        self._q.put(None)
        if len(self._threads) > 1:
            self._threads[1].join()
        else:
            while True:                          # no threads (tests): run queued notes here
                job = self._q.get()
                if job is None:
                    break
                self.notes_for(*job)
        for a, b in self.failed_ranges:
            parts = ex.extract_part(self.sm.llm, self.segments[a:b],
                                    self.existing + [f["title"] for p in self.parts for f in p["feature_mentions"]],
                                    self.sm.cfg["llm"].get("max_transcript_chars_per_chunk", 6000))
            self.parts += parts
        self.failed_ranges = []
        return self.segments, self.parts

    def abort(self) -> None:
        self._stop.set()
        self._q.put(None)
