"""Live transcript while recording: 30-second batches appended like a log, notes made along the way."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.config import load_config
from app.models.asr import BaseASR
from app.models.llm import HeuristicLLM
from app.pipeline.live import RATE, quiet_cut
from app.pipeline.session import SessionManager
from app.workspace.paths import Layout
from app.workspace.store import Store


def tone(seconds, loud=0.3):
    t = np.arange(int(seconds * RATE)) / RATE
    return (np.sin(2 * np.pi * 220 * t) * loud).astype(np.float32)


class FakeTrack:
    def __init__(self, offset=0.0):
        self.offset, self.pending = offset, []

    def drain(self):
        out = np.concatenate(self.pending) if self.pending else np.zeros(0, np.float32)
        self.pending = []
        return out


class FakeRecorder:
    def __init__(self, keys=("mic", "loopback")):
        self.tracks = {k: FakeTrack() for k in keys}


class ScriptASR(BaseASR):
    """Says the next scripted sentence for each batch it hears (one segment spanning the batch)."""
    name = "script"
    available = True

    def __init__(self, lines):
        super().__init__("script")
        self.lines, self.calls = list(lines), 0

    def transcribe_audio(self, audio):
        self.calls += 1
        text = self.lines.pop(0) if self.lines else "..."
        return [{"start": 0.0, "end": round(len(audio) / RATE, 2), "text": text, "confidence": 1.0}]


LECTURE = [
    "New feature: Offline sync — edit without network.",
    "Offline sync depends on User login.",
    "New feature: User login — sign in with email.",
    "Decision: ship the beta in May.",
]


@pytest.fixture()
def env(tmp_path):
    cfg = load_config()
    cfg["workspace"] = str(tmp_path / "ws")
    cfg["audio"]["save_transcripts_to"] = str(tmp_path / "fixtures")
    cfg["audio"]["live_notes_every_s"] = 3
    layout = Layout(cfg["workspace"])
    store = Store(layout)
    store.init_workspace()
    asr = ScriptASR(LECTURE)
    sm = SessionManager(cfg, layout, store, asr, HeuristicLLM())
    return sm, tmp_path


def test_quiet_cut_splits_in_the_pause():
    x = np.concatenate([tone(5), np.zeros(int(0.4 * RATE), np.float32), tone(1.5)])
    cut = quiet_cut(x)
    assert 5 * RATE <= cut <= 5.4 * RATE                  # inside the silence, not mid-word


def test_transcript_grows_batch_by_batch_and_notes_are_made_while_recording(env):
    sm, tmp = env
    s = sm.create("recording", None, "Atlas", "Live test")
    rec = FakeRecorder(keys=("loopback",))
    live = sm.start_live(s["id"], rec, threads=False)
    saved = Path(sm.load(s["id"])["saved_transcript"])
    assert saved.exists() and saved.read_text().startswith("# Recorded meeting")

    rec.tracks["loopback"].pending = [tone(2)]
    assert live.step() == 1
    first = sm.transcript(s["id"])
    assert "Offline sync" in first and saved.read_text().count("\n") == 2       # header + 1 line, like a log

    rec.tracks["loopback"].pending = [tone(2)]
    live.step()
    lines = sm.transcript(s["id"]).splitlines()
    assert len(lines) == 2 and lines[0] == first.strip()                       # appended, never rewritten
    assert "[00:00:02" in lines[1]                                             # times keep counting
    assert live._q.qsize() == 1                                                 # 4 s of text >= 3 s: notes queued

    rec.tracks["loopback"].pending = [tone(2)]
    live.step()
    rec.tracks["loopback"].pending = [tone(1.2)]                               # last bit, after Stop
    sm.finish_recording(s["id"], {"loopback": 0.0}, [])
    out = sm.finish_live(s["id"])
    assert out["stage"] == "ready", out.get("error")
    assert out["timings"]["live_parts"] == 2                                   # notes came in 2 parts
    titles = {c["title"] for c in sm.proposal(s["id"])["changes"]}
    assert {"Offline sync", "User login"} <= titles
    assert len(sm.transcript(s["id"]).splitlines()) == 4
    assert "ship the beta" in saved.read_text()
    assert json.loads((sm._dir(s["id"]) / "session.json").read_text())["live_stats"]["lines"] == 4


def test_mic_and_computer_sound_are_both_used(env):
    sm, _ = env
    s = sm.create("recording", None, "Atlas", "Two tracks")
    rec = FakeRecorder()
    rec.tracks["mic"].offset = 3.0
    live = sm.start_live(s["id"], rec, threads=False)
    rec.tracks["mic"].pending = [tone(2)]
    rec.tracks["loopback"].pending = [tone(2)]
    assert live.step() == 2
    text = sm.transcript(s["id"])
    assert "(you)" in text and "(remote)" in text
    assert "[00:00:00-00:00:02] (remote)" in text and "[00:00:03-00:00:05] (you)" in text  # mic offset kept


def test_silence_is_not_sent_to_speech_recognition(env):
    sm, _ = env
    s = sm.create("recording", None, "Atlas", "Quiet")
    rec = FakeRecorder(keys=("mic",))
    live = sm.start_live(s["id"], rec, threads=False)
    rec.tracks["mic"].pending = [np.zeros(RATE * 5, np.float32)]
    assert live.step() == 0 and sm.asr.calls == 0


def test_discarding_removes_live_transcript_copy(env):
    sm, _ = env
    s = sm.create("recording", None, "Atlas", "Oops")
    live = sm.start_live(s["id"], FakeRecorder(keys=("mic",)), threads=False)
    saved = Path(sm.load(s["id"])["saved_transcript"])
    assert saved.exists()
    sm.stop_live(s["id"])
    sm.discard(s["id"])
    assert not saved.exists() and s["id"] not in sm.live


def test_live_failure_falls_back_to_full_recording(env, monkeypatch):
    sm, _ = env
    s = sm.create("recording", None, "Atlas", "Broken")
    live = sm.start_live(s["id"], FakeRecorder(keys=("mic",)), threads=False)
    live.error = "Live transcript stopped: boom"
    called = {}
    monkeypatch.setattr(sm, "transcribe", lambda sid: called.setdefault("t", sm._set(sid, stage="failed",
                                                                                    error="no audio")))
    out = sm.finish_live(s["id"])
    assert called and out["stage"] == "failed"


def test_threads_run_and_finish(env):
    """Real worker threads with a tiny batch time."""
    sm, _ = env
    sm.cfg["audio"]["live_batch_s"] = 0.05
    s = sm.create("recording", None, "Atlas", "Threads")
    rec = FakeRecorder(keys=("loopback",))
    live = sm.start_live(s["id"], rec)
    rec.tracks["loopback"].pending = [tone(2)]
    import time
    for _ in range(100):
        if live.stats["lines"]:
            break
        time.sleep(0.02)
    assert live.stats["lines"] == 1
    rec.tracks["loopback"].pending = [tone(2)]
    out = sm.finish_live(s["id"])
    assert out["stage"] == "ready", out.get("error")
    assert len(sm.transcript(s["id"]).splitlines()) == 2
