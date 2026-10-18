"""End-to-end tests of the approval contract with the rule-based backend (no models needed).

Run:  python -m pytest -q
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from app.config import load_config
from app.models.asr import NoASR
from app.models.llm import HeuristicLLM
from app.pipeline.query import IDK, answer
from app.pipeline.session import SessionManager
from app.workspace.commit import build_commit
from app.workspace.graph import build_graph
from app.workspace.paths import Layout, PathError, check_id, confine
from app.workspace.store import Store
from app.workspace.transaction import Transactions, TransactionError

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture()
def env(tmp_path):
    cfg = load_config()
    cfg["workspace"] = str(tmp_path / "ws")
    layout = Layout(cfg["workspace"])
    store = Store(layout)
    store.init_workspace()
    sm = SessionManager(cfg, layout, store, NoASR(), HeuristicLLM())
    return cfg, layout, store, sm


def run_meeting(sm, fixture, project_id, project_name, title, confirm=True):
    s = sm.import_text((FIX / fixture).read_text(encoding="utf-8"), project_id, project_name, title)
    s = sm.extract(s["id"])
    assert s["stage"] == "ready", s.get("error")
    prop = sm.proposal(s["id"])
    if confirm:
        for c in prop["changes"]:
            if c.get("needs_confirmation"):
                c["match_confirmed"] = True
    return s, prop


def tree_hash(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(d.rglob("*")):
        if p.is_file() and ".index" not in p.name:
            h.update(str(p.relative_to(d)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def two_meetings(sm, store):
    s1, p1 = run_meeting(sm, "atlas_01_kickoff.txt", None, "Atlas", "Kickoff")
    r1 = sm.approve(s1["id"], p1)
    assert r1["ok"], r1
    pid = r1["project_id"]
    s2, p2 = run_meeting(sm, "atlas_02_sprint_sync.txt", pid, "Atlas", "Sprint sync")
    r2 = sm.approve(s2["id"], p2)
    assert r2["ok"], r2
    return pid


def test_first_meeting_creates_project(env):
    cfg, layout, store, sm = env
    s, prop = run_meeting(sm, "atlas_01_kickoff.txt", None, "Atlas", "Kickoff")
    titles = sorted(c["title"] for c in prop["changes"])
    assert titles == ["Offline sync", "Search", "User login"]
    sync = next(c for c in prop["changes"] if c["title"] == "Offline sync")
    assert sync["depends_on"] == ["User login"]
    assert set(sync["subtasks"]) == {"local change queue", "storage format decision"}
    r = sm.approve(s["id"], prop)
    assert r["ok"] and r["cleanup_ok"], r
    pid = r["project_id"]
    pdir = layout.project_dir(pid)
    assert (layout.root / "AGENTS.md").exists()
    for rel in ("project.json", "MASTER_LOG.md", "meetings", "features"):
        assert (pdir / rel).exists(), rel
    assert store.project(pid)["revision"] == 1
    assert len(store.features(pid)) == 3
    assert not layout.session_dir(s["id"]).exists(), "staging must be removed after verified commit"
    assert not (pdir / ".txn").exists() or not any((pdir / ".txn").iterdir())


def test_second_meeting_changes_feature_and_dependency(env):
    cfg, layout, store, sm = env
    pid = two_meetings(sm, store)
    feats = {f["title"]: f for f in store.features(pid)}
    sync, login, export, search = feats["Offline sync"], feats["User login"], feats["Export to PDF"], feats["Search"]
    assert sync["revision"] == 2
    assert "last-write-wins" in sync["description"]
    assert "conflict rules" in [s["text"] for s in sync["subtasks"]]
    assert "local store" in sync["impact"]
    assert export["depends_on"] == [sync["id"]]
    assert login["status"] == "done"
    assert search["status"] == "archived"           # tombstone, not deleted
    assert len(sync["source_meetings"]) == 2
    log = store.master_log(pid)
    assert "Revision 2" in log and "ARCHIVE" in log and "CHANGE" in log
    g = build_graph(store, pid)
    types = [e["type"] for e in g["edges"]]
    assert "meeting_relates_to_meeting" in types
    assert types.count("feature_depends_on_feature") == 2


def test_match_requires_confirmation(env):
    cfg, layout, store, sm = env
    s1, p1 = run_meeting(sm, "atlas_01_kickoff.txt", None, "Atlas", "Kickoff")
    pid = sm.approve(s1["id"], p1)["project_id"]
    s2, p2 = run_meeting(sm, "atlas_02_sprint_sync.txt", pid, "Atlas", "Sprint sync", confirm=False)
    signin = next(c for c in p2["changes"] if "sign-in" in " ".join(c["flags"]).lower())
    assert signin["feature_id"] == next(f["id"] for f in store.features(pid) if f["title"] == "User login")
    r = sm.approve(s2["id"], p2)
    assert not r["ok"] and any("same as your existing" in e for e in r["errors"])
    assert store.project(pid)["revision"] == 1


def test_reject_leaves_project_unchanged_and_keeps_transcript(env):
    cfg, layout, store, sm = env
    s1, p1 = run_meeting(sm, "atlas_01_kickoff.txt", None, "Atlas", "Kickoff")
    pid = sm.approve(s1["id"], p1)["project_id"]
    before = tree_hash(layout.project_dir(pid))
    s2, _ = run_meeting(sm, "atlas_02_sprint_sync.txt", pid, "Atlas", "Sprint sync")
    sm.reject(s2["id"])
    assert tree_hash(layout.project_dir(pid)) == before
    assert sm.load(s2["id"])["stage"] == "transcribed"
    assert sm.transcript(s2["id"]).strip()
    sm.discard(s2["id"])
    assert not layout.session_dir(s2["id"]).exists()


def test_cycle_and_dangling_dependency_blocked(env):
    cfg, layout, store, sm = env
    pid = two_meetings(sm, store)
    login = next(f for f in store.features(pid) if f["title"] == "User login")
    prop = {"project_id": pid, "base_revision": 2, "meeting": {"title": "Bad"},
            "changes": [{"change_id": "c1", "include": True, "action": "change", "feature_id": login["id"],
                         "title": "User login", "depends_on": ["Export to PDF"], "subtasks": []}]}
    plan = build_commit(store, copy.deepcopy(prop))
    assert any("loop" in e.lower() for e in plan.errors)
    prop["changes"][0]["depends_on"] = ["Teleportation"]
    plan = build_commit(store, prop)
    assert any("not a feature" in e for e in plan.errors)


def test_archiving_depended_on_feature_needs_policy(env):
    cfg, layout, store, sm = env
    pid = two_meetings(sm, store)
    sync = next(f for f in store.features(pid) if f["title"] == "Offline sync")
    prop = {"project_id": pid, "base_revision": 2, "meeting": {"title": "Cut sync"},
            "changes": [{"change_id": "c1", "include": True, "action": "archive", "feature_id": sync["id"],
                         "title": "Offline sync"}]}
    plan = build_commit(store, copy.deepcopy(prop))
    assert any("depend on it" in e for e in plan.errors)
    prop["archive_policy"] = "mark_blocked"
    plan = build_commit(store, prop)
    assert plan.ok, plan.errors
    export_rel = next(r for r, c in plan.files.items() if "Export to PDF" in c and r.startswith("features/"))
    assert "status: blocked" in plan.files[export_rel]


def test_stale_proposal_rejected(env):
    cfg, layout, store, sm = env
    s1, p1 = run_meeting(sm, "atlas_01_kickoff.txt", None, "Atlas", "Kickoff")
    pid = sm.approve(s1["id"], p1)["project_id"]
    sa, pa = run_meeting(sm, "atlas_02_sprint_sync.txt", pid, "Atlas", "Sprint A")
    sb, pb = run_meeting(sm, "atlas_02_sprint_sync.txt", pid, "Atlas", "Sprint B")
    assert sm.approve(sa["id"], pa)["ok"]
    r = sm.approve(sb["id"], pb)
    assert not r["ok"] and "changed since" in r["errors"][0]


def test_transcript_deleted_after_commit(env):
    cfg, layout, store, sm = env
    two_meetings(sm, store)
    phrase = "Quick sprint sync on Atlas"   # appears only in the transcript
    for p in layout.root.rglob("*"):
        if p.is_file():
            assert phrase not in p.read_text(encoding="utf-8", errors="ignore"), p
    assert not any(layout.staging_dir().iterdir())


def test_query_cites_and_is_project_scoped(env):
    cfg, layout, store, sm = env
    atlas = two_meetings(sm, store)
    s, p = run_meeting(sm, "website_01_planning.txt", None, "Website", "Planning")
    assert sm.approve(s["id"], p)["ok"]
    llm = HeuristicLLM()
    a = answer(store, llm, atlas, "What changed in offline sync conflict resolution?")
    assert a["citations"], a
    kinds = {c["kind"] for c in a["citations"]}
    assert "feature" in kinds or "meeting" in kinds
    b = answer(store, llm, atlas, "When does the pricing page launch?")
    assert b["answer"] == IDK
    assert all("Pricing" not in c["title"] for c in b["citations"])


def test_crash_recovery_rolls_back_unlogged_commit(env):
    cfg, layout, store, sm = env
    s1, p1 = run_meeting(sm, "atlas_01_kickoff.txt", None, "Atlas", "Kickoff")
    pid = sm.approve(s1["id"], p1)["project_id"]
    before = tree_hash(layout.project_dir(pid))
    s2, p2 = run_meeting(sm, "atlas_02_sprint_sync.txt", pid, "Atlas", "Sprint sync")
    plan = build_commit(store, p2)
    assert plan.ok
    tx = Transactions(layout)
    for phase in ("prepared", "applied"):
        with pytest.raises(TransactionError):
            tx.commit(pid, plan.files, plan.log_entry, plan.revision, _crash_after=phase)
        notes = sm.recover()
        assert any("Rolled back" in n for n in notes)
        assert tree_hash(layout.project_dir(pid)) == before
    # the session is still approvable afterwards
    assert sm.approve(s2["id"], p2)["ok"]


def test_crash_after_log_is_finished(env, monkeypatch):
    cfg, layout, store, sm = env
    s1, p1 = run_meeting(sm, "atlas_01_kickoff.txt", None, "Atlas", "Kickoff")
    # Simulate: commit fully logged, app died before verify/cleanup.
    monkeypatch.setattr(SessionManager, "_verify", lambda self, plan, txn: (_ for _ in ()).throw(SystemExit))
    with pytest.raises(SystemExit):
        sm.approve(s1["id"], p1)
    monkeypatch.undo()
    assert layout.session_dir(s1["id"]).exists()
    notes = sm.recover()
    assert any("already saved" in n for n in notes)
    assert not layout.session_dir(s1["id"]).exists()


def test_path_confinement():
    with pytest.raises(PathError):
        check_id("../../etc")
    with pytest.raises(PathError):
        confine(Path("/tmp/ws"), "..", "evil")


def test_explicit_project_then_first_meeting(env):
    cfg, layout, store, sm = env
    p = sm.create_project("Atlas")
    assert store.project(p["id"])["revision"] == 0
    assert sm.create_project("atlas")["id"] == p["id"]          # no duplicate by name
    s, prop = run_meeting(sm, "atlas_01_kickoff.txt", p["id"], "Atlas", "Kickoff")
    r = sm.approve(s["id"], prop)
    assert r["ok"] and r["project_id"] == p["id"]
    assert store.project(p["id"])["revision"] == 1
    assert "project created" in store.master_log(p["id"])
