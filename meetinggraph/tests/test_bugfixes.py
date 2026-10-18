"""Regression tests for the six fixes of 2026-10-18 (no NPU, no microphone: fakes stand in for both)."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import wave
from pathlib import Path

import pytest

from app.config import load_config
from app.models import asr as asr_mod
from app.models.asr import NoASR
from app.models.llm import HeuristicLLM, _RunnerServer, _skeleton
from app.models.npu import NPU_LOCK
from app.pipeline import extract as ex, prompts
from app.pipeline.session import SessionManager
from app.workspace.paths import Layout
from app.workspace.store import Store

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture()
def sm(tmp_path):
    cfg = load_config()
    cfg["workspace"] = str(tmp_path / "ws")
    layout = Layout(cfg["workspace"])
    store = Store(layout)
    store.init_workspace()
    return SessionManager(cfg, layout, store, NoASR(), HeuristicLLM())


def _wav(path: Path, seconds: float):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * int(16000 * seconds))


# ---- 1: a recording interrupted before any audio is marked failed, a real one is kept ----
def test_crashed_empty_recording_is_marked_failed(sm):
    p = sm.create_project("P")
    s = sm.create("recording", p["id"], "P", "Crashed at start")
    mic, loop = sm.recording_paths(s["id"])
    mic.write_bytes(b"")                       # what the crash left: an empty mic file ...
    _wav(loop, 0.2)                            # ... and 0.2 s of system sound
    notes = sm.recover()
    after = sm.load(s["id"])
    assert after["stage"] == "failed" and "before any audio" in after["error"]
    assert any("no usable audio" in n for n in notes)


def test_interrupted_recording_with_audio_is_kept(sm):
    p = sm.create_project("P")
    s = sm.create("recording", p["id"], "P", "Long meeting")
    mic, _ = sm.recording_paths(s["id"])
    _wav(mic, 3.0)
    sm.recover()
    after = sm.load(s["id"])
    assert after["stage"] == "recorded" and after["audio"] == {"mic": "mic.wav"}


# ---- 3: the JSON shape shows what goes where; a placeholder copied back counts as empty ----
def _strings(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from _strings(v)
    elif isinstance(x, str):
        yield x


def test_skeleton_has_no_empty_values():
    for schema in (prompts.NOTES_SCHEMA, prompts.FEATURES_SCHEMA, prompts.FEATURES_COMPACT_SCHEMA):
        vals = list(_strings(_skeleton(schema)))
        assert vals and all(v for v in vals), vals
    sk = _skeleton(prompts.FEATURES_SCHEMA)["feature_mentions"][0]
    assert sk["action"] == "add|change|archive" and sk["evidence"][0]["quote"] == "<exact words from the transcript>"


def test_placeholders_copied_back_are_dropped():
    d = ex._normalize_draft({
        "summary": "<2-4 sentences>",
        "decisions": [{"text": "Ship on Nov 12", "quote": "<exact words from the transcript>"},
                      {"text": "<one sentence>", "quote": "x"}],
        "action_items": [{"text": "Arun writes the copy", "owner": "<name, or empty>", "quote": "Arun will"}],
        "feature_mentions": [{"title": "<1-4 word name>"}, {"title": "Offline sync", "description": "<1-2 sentences>"}],
    })
    assert d["summary"] == ""
    assert [x["text"] for x in d["decisions"]] == ["Ship on Nov 12"] and d["decisions"][0]["quote"] == ""
    assert d["action_items"][0]["owner"] == ""
    assert [f["title"] for f in d["feature_mentions"]] == ["Offline sync"]
    assert d["feature_mentions"][0]["description"] == ""


# ---- 2: prompt + a complete answer always fit a small context window ----
class SmallCtxLLM:
    heuristic = False
    max_seq_len = 2048

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, max_tokens=1024, json_mode=False, schema=None):
        # exactly what the NPU adapter sends: the prompt plus the JSON-shape instruction
        full = prompt + "\n\nReply with ONLY one JSON object in exactly this shape:\n" + \
            json.dumps(_skeleton(schema), separators=(",", ":"))
        self.prompts.append((full, max_tokens))
        if "feature_mentions" in json.dumps(schema):
            return '{"feature_mentions": []}'
        return '{"summary": "s", "decisions": [], "action_items": [], "open_questions": []}'


def test_requests_fit_the_context_window():
    llm = SmallCtxLLM()
    long_meeting = (FIX / "atlas_03_natural_design_review.txt").read_text(encoding="utf-8") * 6
    from app.pipeline.transcript import from_text
    ex.extract_part(llm, from_text(long_meeting), ["Offline sync", "User login"], max_chars=2500)
    assert len(llm.prompts) >= 4                                    # long meeting -> several chunks
    for full, budget in llm.prompts:
        prompt_tokens = int(len(full) / ex.CHARS_PER_TOKEN) + 64     # same estimate as the NPU adapter
        assert prompt_tokens + budget <= 2048, (prompt_tokens, budget)
    assert any("at most 5 items" in f for f, _ in llm.prompts)      # compact prompt for small contexts


def test_large_context_models_keep_the_full_prompt():
    chars, template, schema, notes_out, feat_out = ex.plan_requests(HeuristicLLM(), 6000, "- (none yet)")
    assert (chars, template, notes_out, feat_out) == (6000, prompts.FEATURES, 1200, 3500)


# ---- 4: one NPU job at a time ----
def test_speech_waits_for_the_notes_model():
    a = asr_mod.ExecuTorchRunnerASR.__new__(asr_mod.ExecuTorchRunnerASR)
    a.runner = "fake"
    ran = []
    a._qnn_transcribe = lambda path: ran.append(time.monotonic()) or []
    with NPU_LOCK:                                 # the notes model is using the NPU
        t = threading.Thread(target=a.transcribe, args=("x.wav",))
        t0 = time.monotonic()
        t.start()
        time.sleep(0.3)
        assert not ran                             # speech is queued, not running alongside
    t.join(5)
    assert ran and ran[0] - t0 >= 0.3


# ---- 5: warm-up loads the model ahead of the first request ----
FAKE = r'''
import sys
from pathlib import Path
print("READY", flush=True)
for line in sys.stdin:
    line = line.rstrip("\r\n")
    if line == "QUIT":
        break
    pf, n, out = line.split("\t")
    Path(out).write_text("<|assistant|>ok<|end|>", encoding="utf-8")
    print("OK\t1\t1\t1\t1", flush=True)
'''


def test_warm_up_loads_before_the_first_request(tmp_path):
    fake = tmp_path / "fake_runner.py"
    fake.write_text(FAKE, encoding="utf-8")
    s = _RunnerServer([sys.executable, str(fake)], load_timeout=30, idle_s=0)
    s.warm_up()
    deadline = time.monotonic() + 10
    while (s.proc is None or s.proc.poll() is not None) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert s.proc is not None and s.proc.poll() is None        # loaded with no request yet
    pid = s.proc.pid
    assert "ok" in s.generate("hi", 16, 10) and s.proc.pid == pid
    s.stop()


# ---- 6: pressing Record does not block the window while devices open ----
def test_record_opens_devices_off_the_ui_thread(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))
    from PySide6.QtWidgets import QApplication
    from app.ui import recorder_pill as rp
    app = QApplication.instance() or QApplication([])

    class SlowRecorder:
        is_paused = False

        def __init__(self, *a):
            self.tracks = {}

        def start(self):
            time.sleep(0.8)                              # a slow (e.g. Bluetooth) device
            return []

        def stop(self):
            return {}, []

        def levels(self):
            return {"mic": 0.0}

        def elapsed(self):
            return 0.0

        def health(self):
            return []

    monkeypatch.setattr(rp.recorder, "list_devices", lambda: {"default_mic": "m", "default_speaker": "s"})
    monkeypatch.setattr(rp.recorder, "DualRecorder", SlowRecorder)

    class SM:
        def create(self, *a):
            return {"id": "s1"}

        def recording_paths(self, sid):
            return tmp_path / "mic.wav", tmp_path / "loop.wav"

        def start_live(self, sid, rec):
            return None

        def discard(self, sid):
            pass

    class Ctx:
        cfg = {"audio": {"max_minutes": 60}}
        sm = SM()
        llm = HeuristicLLM()

    pill = rp.RecorderPill(Ctx())
    pill.begin("p", "P", "Meeting")
    t0 = time.monotonic()
    pill._on_record()
    assert time.monotonic() - t0 < 0.3                   # returned at once: the UI stays responsive
    assert pill.state.text() == "Starting…"              # shown in full, not clipped
    deadline = time.monotonic() + 10
    while pill.rec is None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.02)
    assert pill.rec is not None and pill.sid == "s1" and pill.timer.isActive()
    for text in ("Starting…", "Listening…", "Recording", "Paused", "128 lines · 12 notes", "999 lines · 99 notes"):
        pill._status(text)
        assert pill.state.text() == text, (text, pill.state.text())     # every status fits uncut
    pill.timer.stop()
    pill.anim.stop()
    pill.rec = None
    pill.hide()


# ---- features the transcript does not back never reach Review ----
def test_features_not_in_the_transcript_are_removed():
    from app.pipeline.transcript import from_text
    segs = from_text("Ana: Did you see the Knicks game last night?\nBen: Yes, and then we watched a movie.")
    draft = {"summary": "s", "decisions": [], "action_items": [], "open_questions": [],
             "feature_mentions": [
                 {"title": "Mitochondria", "evidence": [{"quote": "the powerhouse of the cell", "start": 0, "end": 0}]},
                 {"title": "Knicks game", "evidence": []},                                   # its name is said
                 {"title": "Movie night", "evidence": [{"quote": "we watched a movie", "start": 0, "end": 0}]},
             ]}
    ex.check_evidence(draft, segs)
    ex.drop_ungrounded_features(draft, segs)
    assert [f["title"] for f in draft["feature_mentions"]] == ["Knicks game", "Movie night"]
    assert draft["dropped_features"] == ["Mitochondria"]
