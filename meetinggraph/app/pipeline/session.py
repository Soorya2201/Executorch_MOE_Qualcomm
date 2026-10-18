"""Processing sessions in workspace/.staging/<session-id>/ and the approve/commit/cleanup contract.

Stages: recording -> recorded -> transcribing -> transcribed -> summarizing -> ready
        -> saving -> (complete: staging removed) | failed
Rejecting a proposal returns the session to 'transcribed' (transcript kept until Discard).
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from ..models.asr import BaseASR
from ..models.llm import BaseLLM
from ..workspace.commit import CommitPlan, build_commit
from ..workspace.index import ProjectIndex
from ..workspace.paths import Layout, check_id, clean_name, confine, new_id
from ..workspace.store import Store, now_iso
from ..workspace.transaction import Transactions, TransactionError
from . import extract as ex
from .match import build_proposal
from .transcript import from_text, merge_tracks, to_text

ACTIVE_STAGES = {"transcribing", "summarizing", "saving"}


class SessionError(RuntimeError):
    pass


class SessionManager:
    def __init__(self, cfg: dict, layout: Layout, store: Store, asr: BaseASR, llm: BaseLLM):
        self.cfg, self.layout, self.store, self.asr, self.llm = cfg, layout, store, asr, llm
        self.txns = Transactions(layout)
        self._lock = threading.RLock()

    # ---------- basic IO ----------
    def _dir(self, sid: str) -> Path:
        return self.layout.session_dir(sid)

    def _file(self, sid: str, name: str) -> Path:
        return confine(self._dir(sid), name)

    def load(self, sid: str) -> dict[str, Any]:
        return json.loads(self._file(sid, "session.json").read_text(encoding="utf-8"))

    def save(self, s: dict[str, Any]) -> None:
        with self._lock:
            p = self._file(s["id"], "session.json")
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(s, indent=2), encoding="utf-8")
            os.replace(tmp, p)

    def _set(self, sid: str, **kw) -> dict:
        with self._lock:
            s = self.load(sid)
            s.update(kw)
            s["updated_at"] = now_iso()
            self.save(s)
            return s

    def list(self) -> list[dict]:
        out = []
        d = self.layout.staging_dir()
        if not d.exists():
            return out
        for p in d.iterdir():
            f = p / "session.json"
            if f.exists():
                try:
                    out.append(json.loads(f.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError):
                    continue
        return sorted(out, key=lambda s: s.get("created_at", ""), reverse=True)

    # ---------- projects ----------
    def create_project(self, name: str) -> dict:
        """Create an empty project now (the user asked for it explicitly). Logged like any change."""
        name = clean_name(name)
        for p in self.store.list_projects():
            if p["name"].strip().lower() == name.lower():
                return p
        pid = new_id()
        project = {"schema_version": 1, "id": pid, "name": name, "created_at": now_iso(), "revision": 0,
                   "features": {}, "meetings": []}
        entry = f"## Revision 0 · {now_iso()} · project created\n- Project \"{name}\" created by user"
        txn = self.txns.commit(pid, {"project.json": json.dumps(project, indent=2) + "\n"}, entry, 0)
        self.txns.finish(pid, txn)
        return project

    # ---------- create ----------
    def create(self, source: str, project_id: str | None, project_name: str, title: str) -> dict:
        sid = new_id()
        self._dir(sid).mkdir(parents=True, exist_ok=False)
        if project_id:
            check_id(project_id)
        s = {"id": sid, "created_at": now_iso(), "updated_at": now_iso(), "source": source,
             "stage": "recording" if source == "recording" else "recorded",
             "project_id": project_id, "project_name": clean_name(project_name or "New project"),
             "title": clean_name(title or "Meeting"), "date": now_iso()[:10],
             "audio": {}, "offsets": {}, "error": "", "note": "", "timings": {},
             "labels": {"asr": self.asr.label, "llm": self.llm.label}}
        self.save(s)
        return s

    def import_text(self, text: str, project_id: str | None, project_name: str, title: str) -> dict:
        s = self.create("text_import", project_id, project_name, title)
        segs = from_text(text)
        self._file(s["id"], "transcript.txt").write_text(to_text(segs), encoding="utf-8")
        return self._set(s["id"], stage="transcribed")

    def import_audio(self, src: str, project_id: str | None, project_name: str, title: str) -> dict:
        s = self.create("audio_import", project_id, project_name, title)
        ext = Path(src).suffix.lower() or ".wav"
        dst = self._file(s["id"], f"import{ext}")
        shutil.copy2(src, dst)
        return self._set(s["id"], audio={"import": dst.name})

    def recording_paths(self, sid: str) -> tuple[Path, Path]:
        return self._file(sid, "mic.wav"), self._file(sid, "loopback.wav")

    def finish_recording(self, sid: str, offsets: dict[str, float], health: list[str]) -> dict:
        audio = {}
        mic, loop = self.recording_paths(sid)
        if mic.exists():
            audio["mic"] = mic.name
        if loop.exists():
            audio["loopback"] = loop.name
        note = "; ".join(health)
        return self._set(sid, stage="recorded", audio=audio, offsets=offsets, note=note)

    # ---------- transcript ----------
    def transcript(self, sid: str) -> str:
        p = self._file(sid, "transcript.txt")
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def save_transcript(self, sid: str, text: str) -> dict:
        self._file(sid, "transcript.txt").write_text(text, encoding="utf-8")
        s = self.load(sid)
        if s["stage"] in ("recorded", "failed", "ready", "transcribed"):
            # an edited transcript invalidates any existing proposal
            self._file(sid, "proposal.json").unlink(missing_ok=True)
            return self._set(sid, stage="transcribed", error="")
        return s

    def transcribe(self, sid: str) -> dict:
        s = self._set(sid, stage="transcribing", error="")
        t0 = time.perf_counter()
        try:
            audio = s.get("audio") or {}
            if "import" in audio:
                segs = self.asr.transcribe(str(self._file(sid, audio["import"])))
            else:
                tracks = {}
                for key in ("mic", "loopback"):
                    if key in audio:
                        off = float((s.get("offsets") or {}).get(key, 0.0))
                        tracks[key] = [dict(x, start=x["start"] + off, end=x["end"] + off)
                                       for x in self.asr.transcribe(str(self._file(sid, audio[key])))]
                if not tracks:
                    raise SessionError("No audio was recorded. Check the selected devices, or import audio/text.")
                segs = merge_tracks(tracks.get("mic", []), tracks.get("loopback", []))
            if not segs:
                raise SessionError("No speech was recognized. Check levels, or type the transcript manually.")
            self._file(sid, "transcript.txt").write_text(to_text(segs), encoding="utf-8")
            timings = dict(s.get("timings") or {}, transcribe_s=round(time.perf_counter() - t0, 2))
            return self._set(sid, stage="transcribed", timings=timings, labels=dict(s["labels"], asr=self.asr.label))
        except Exception as e:  # surfaced in the Inbox with Retry
            return self._set(sid, stage="failed", error=f"Transcription failed: {e}")

    # ---------- extraction ----------
    def extract(self, sid: str) -> dict:
        s = self._set(sid, stage="summarizing", error="")
        t0 = time.perf_counter()
        try:
            segs = from_text(self.transcript(sid))
            pid = s.get("project_id")
            existing = [f["title"] for f in self.store.features(pid)
                        if f.get("status") != "archived"] if pid and self.store.project(pid) else []
            draft = ex.extract_draft(self.llm, segs, s["title"], existing,
                                     self.cfg["llm"].get("max_transcript_chars_per_chunk", 6000))
            self._file(sid, "draft.json").write_text(json.dumps(draft, indent=2), encoding="utf-8")
            proposal = build_proposal(self.store, pid if pid and self.store.project(pid) else None,
                                      s["project_name"], draft, sid, s["date"], segments=segs)
            self._file(sid, "proposal.json").write_text(json.dumps(proposal, indent=2), encoding="utf-8")
            timings = dict(s.get("timings") or {}, extract_s=round(time.perf_counter() - t0, 2))
            return self._set(sid, stage="ready", timings=timings, labels=dict(s["labels"], llm=self.llm.label))
        except ex.ExtractionError as e:
            note = ""
            if e.raw:
                out = self._file(sid, "model_output.txt")
                out.write_text(e.raw, encoding="utf-8")
                note = f" The AI's raw reply was saved to {out} (send it to your team to debug)."
            return self._set(sid, stage="failed", error=f"{e}{note}")
        except Exception as e:
            return self._set(sid, stage="failed", error=f"Extraction failed: {e}")

    def rematch(self, sid: str) -> dict:
        """Rebuild cards against the project's current state (discards card edits)."""
        s = self.load(sid)
        draft = json.loads(self._file(sid, "draft.json").read_text(encoding="utf-8"))
        pid = s.get("project_id")
        proposal = build_proposal(self.store, pid if pid and self.store.project(pid) else None,
                                  s["project_name"], draft, sid, s["date"], segments=from_text(self.transcript(sid)))
        self._file(sid, "proposal.json").write_text(json.dumps(proposal, indent=2), encoding="utf-8")
        return self._set(sid, stage="ready", error="")

    def proposal(self, sid: str) -> dict | None:
        p = self._file(sid, "proposal.json")
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def save_proposal(self, sid: str, proposal: dict) -> None:
        self._file(sid, "proposal.json").write_text(json.dumps(proposal, indent=2), encoding="utf-8")

    def set_project(self, sid: str, project_id: str | None, project_name: str) -> dict:
        return self._set(sid, project_id=project_id, project_name=clean_name(project_name))

    # ---------- review ----------
    def preview(self, sid: str, proposal: dict) -> CommitPlan:
        return build_commit(self.store, json.loads(json.dumps(proposal)))

    def reject(self, sid: str) -> dict:
        self._file(sid, "proposal.json").unlink(missing_ok=True)
        return self._set(sid, stage="transcribed", note="Proposal rejected; project unchanged. "
                                                        "Transcript kept until you discard the session.")

    def discard(self, sid: str) -> None:
        d = self._dir(sid)
        shutil.rmtree(d, ignore_errors=True)
        if d.exists():
            raise SessionError(f"Could not delete {d}. Close any program using these files and retry.")

    def approve(self, sid: str, proposal: dict) -> dict:
        """Validate -> transactional commit -> verify -> rebuild index -> delete transcript/audio."""
        s = self._set(sid, stage="saving", error="")
        self.save_proposal(sid, proposal)
        t0 = time.perf_counter()
        plan = build_commit(self.store, json.loads(json.dumps(proposal)))
        if not plan.ok:
            self._set(sid, stage="ready", error="Cannot save: " + " | ".join(plan.errors))
            return {"ok": False, "errors": plan.errors}
        try:
            txn_id = self.txns.commit(plan.project_id, plan.files, plan.log_entry, plan.revision)
        except (TransactionError, OSError) as e:
            self._set(sid, stage="ready", error=f"Save failed, project unchanged or restored: {e}")
            self.txns.recover_all()
            return {"ok": False, "errors": [str(e)]}
        self._set(sid, commit={"project_id": plan.project_id, "txn_id": txn_id, "revision": plan.revision,
                               "meeting_id": plan.meeting_id}, project_id=plan.project_id)
        problems = self._verify(plan, txn_id)
        if problems:
            self._set(sid, stage="failed", error="Commit verification failed; transcript kept: " + "; ".join(problems))
            return {"ok": False, "errors": problems}
        self.txns.finish(plan.project_id, txn_id)
        ProjectIndex(self.store, plan.project_id).rebuild()
        timings = dict(s.get("timings") or {}, commit_s=round(time.perf_counter() - t0, 2))
        self._set(sid, timings=timings)
        cleanup = self.cleanup(sid)
        return {"ok": True, "project_id": plan.project_id, "meeting_id": plan.meeting_id,
                "revision": plan.revision, "cleanup_ok": cleanup["ok"], "cleanup_error": cleanup.get("error", ""),
                "warnings": plan.warnings}

    def _verify(self, plan: CommitPlan, txn_id: str) -> list[str]:
        problems = []
        for rel, content in plan.files.items():
            p = self.layout.project_file(plan.project_id, rel)
            try:
                if p.read_text(encoding="utf-8") != content:
                    problems.append(f"{rel} content differs")
            except OSError as e:
                problems.append(f"{rel} unreadable: {e}")
        if not self.store.log_has_revision(plan.project_id, plan.revision, txn_id):
            problems.append("MASTER_LOG.md lacks the commit marker")
        return problems

    def cleanup(self, sid: str) -> dict:
        """Delete transcript, proposal and temporary audio for a verified commit."""
        s = self.load(sid)
        keep_audio = bool(self.cfg["audio"].get("keep_audio_after_commit"))
        try:
            if keep_audio and s.get("audio") and s.get("commit"):
                c = s["commit"]
                adir = self.layout.project_file(c["project_id"], "audio")
                adir.mkdir(exist_ok=True)
                for key, name in s["audio"].items():
                    src = self._file(sid, name)
                    if src.exists():
                        shutil.move(str(src), adir / f"{c['meeting_id']}_{key}{src.suffix}")
            d = self._dir(sid)
            shutil.rmtree(d)
            if d.exists():
                raise OSError(f"{d} still exists")
            return {"ok": True}
        except OSError as e:
            try:
                self._set(sid, stage="cleanup_failed",
                          error=f"Saved, but temporary transcript/audio could not be deleted: {e}. Use Retry cleanup.")
            except OSError:
                pass
            return {"ok": False, "error": str(e)}

    # ---------- startup ----------
    def recover(self) -> list[str]:
        notes = self.txns.recover_all()
        for s in self.list():
            sid, stage = s["id"], s.get("stage")
            c = s.get("commit")
            if c and self.store.log_has_revision(c["project_id"], c["revision"], c["txn_id"]):
                r = self.cleanup(sid)
                notes.append(f"Session '{s['title']}' was already saved; cleanup {'done' if r['ok'] else 'failed'}.")
                continue
            if stage == "recording":
                self._set(sid, stage="recorded", note="App closed while recording; partial audio kept.")
            elif stage in ACTIVE_STAGES:
                back = "ready" if stage == "saving" and self._file(sid, "proposal.json").exists() else "failed"
                self._set(sid, stage=back, error="Interrupted when the app closed. Retry from here.")
                notes.append(f"Session '{s['title']}' was interrupted at '{stage}'.")
        return notes
