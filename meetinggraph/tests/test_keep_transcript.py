"""Saved meetings keep their full transcript in the project (audio.keep_transcript_after_commit)."""
from pathlib import Path

from app.config import load_config
from app.models.asr import NoASR
from app.models.llm import HeuristicLLM
from app.pipeline.session import SessionManager
from app.workspace.paths import Layout
from app.workspace.store import Store

FIX = Path(__file__).parent / "fixtures"


def _save_one(tmp_path, keep: bool):
    cfg = load_config()
    cfg["workspace"] = str(tmp_path / "ws")
    cfg["audio"]["keep_transcript_after_commit"] = keep
    layout = Layout(cfg["workspace"])
    store = Store(layout)
    store.init_workspace()
    sm = SessionManager(cfg, layout, store, NoASR(), HeuristicLLM())
    p = sm.create_project("Atlas")
    text = (FIX / "atlas_01_kickoff.txt").read_text(encoding="utf-8")
    s = sm.import_text(text, p["id"], p["name"], "Kickoff")
    sm.extract(s["id"])
    prop = sm.proposal(s["id"])
    for c in prop["changes"]:
        if c.get("needs_confirmation"):
            c["match_confirmed"] = True
    res = sm.approve(s["id"], prop)
    assert res["ok"], res
    return sm, res, s["id"], text


def test_transcript_kept_after_save(tmp_path):
    sm, res, sid, text = _save_one(tmp_path, keep=True)
    kept = sm.transcript_path(res["project_id"], res["meeting_id"])
    assert kept.exists() and kept.read_text(encoding="utf-8").strip()
    assert not sm._dir(sid).exists()          # the temporary session folder is still cleaned up


def test_transcript_deleted_when_turned_off(tmp_path):
    sm, res, sid, _ = _save_one(tmp_path, keep=False)
    assert not sm.transcript_path(res["project_id"], res["meeting_id"]).exists()
    assert not sm._dir(sid).exists()
