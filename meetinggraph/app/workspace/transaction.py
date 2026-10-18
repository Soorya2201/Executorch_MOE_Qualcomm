"""Single-writer file transactions with a manifest, backups, atomic replace and startup recovery.

Commit protocol (per project):
  1. take the project lock
  2. write every new file into .txn/<id>/new/ (flushed + fsynced) and back up the
     current versions into .txn/<id>/old/; manifest state = "prepared"
  3. os.replace each new file onto its target; manifest state = "applied"
  4. append the MASTER_LOG.md entry carrying "<!-- txn:<id> rev:<n> -->"  <- commit marker
  5. caller verifies the files reopen; then finish() deletes .txn/<id>/

Recovery (on startup): for each leftover manifest, if the log already carries its
marker the commit is finished (remaining new files applied); otherwise every target
is restored from old/ (or removed if it did not exist before).
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from .paths import Layout, check_id, confine, new_id

LOCK_STALE_SECONDS = 600


class TransactionError(RuntimeError):
    pass


def _fsync_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def _write_manifest(path: Path, manifest: dict) -> None:
    tmp = path.with_suffix(".tmp")
    _fsync_write(tmp, json.dumps(manifest, indent=2))
    os.replace(tmp, path)


class ProjectLock:
    def __init__(self, project_dir: Path):
        self.path = project_dir / ".lock"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()} {time.time()}".encode())
                os.close(fd)
                return self
            except FileExistsError:
                age = time.time() - self.path.stat().st_mtime
                if age > LOCK_STALE_SECONDS:
                    self.path.unlink(missing_ok=True)
                    continue
                raise TransactionError("Another save is in progress for this project. Try again shortly.")
        raise TransactionError("Could not take the project lock.")

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


class Transactions:
    def __init__(self, layout: Layout):
        self.layout = layout

    def _txn_root(self, project_id: str) -> Path:
        return self.layout.project_file(project_id, ".txn")

    def commit(self, project_id: str, files: dict[str, str], log_entry: str, revision: int,
               _crash_after: str | None = None) -> str:
        """Apply files atomically and append the log entry. Returns the txn id.

        `_crash_after` is for tests: raise right after the named phase
        ("prepared" | "applied") to simulate a crash.
        """
        check_id(project_id)
        pdir = self.layout.project_dir(project_id)
        pdir.mkdir(parents=True, exist_ok=True)
        txn_id = new_id()
        marker = f"<!-- txn:{txn_id} rev:{revision} -->"
        with ProjectLock(pdir):
            troot = confine(self._txn_root(project_id), txn_id)
            (troot / "new").mkdir(parents=True)
            (troot / "old").mkdir(parents=True)
            entries = []
            for i, (rel, content) in enumerate(sorted(files.items())):
                target = confine(pdir, rel)
                staged = troot / "new" / f"{i}.part"
                _fsync_write(staged, content)
                existed = target.exists()
                if existed:
                    shutil.copy2(target, troot / "old" / f"{i}.bak")
                entries.append({"rel": rel, "slot": i, "existed": existed})
            manifest = {"txn_id": txn_id, "revision": revision, "marker": marker,
                        "files": entries, "state": "prepared"}
            mpath = troot / "manifest.json"
            _write_manifest(mpath, manifest)
            if _crash_after == "prepared":
                raise TransactionError("simulated crash after prepare")

            for e in entries:
                target = confine(pdir, e["rel"])
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(troot / "new" / f"{e['slot']}.part", target)
            manifest["state"] = "applied"
            _write_manifest(mpath, manifest)
            if _crash_after == "applied":
                raise TransactionError("simulated crash after apply")

            log = pdir / "MASTER_LOG.md"
            with open(log, "a", encoding="utf-8", newline="\n") as f:
                if f.tell() == 0:
                    f.write("# Master log\n\nAppend-only audit trail. Each revision entry is a commit marker.\n")
                f.write("\n" + log_entry.rstrip() + "\n" + marker + "\n")
                f.flush()
                os.fsync(f.fileno())
            manifest["state"] = "logged"
            _write_manifest(mpath, manifest)
        return txn_id

    def finish(self, project_id: str, txn_id: str) -> None:
        """Drop backups once the caller has verified the commit."""
        troot = confine(self._txn_root(project_id), check_id(txn_id))
        shutil.rmtree(troot, ignore_errors=True)

    def recover_all(self) -> list[str]:
        """Finish or roll back interrupted transactions. Returns human-readable notes."""
        notes = []
        pdir_root = self.layout.projects_dir()
        if not pdir_root.exists():
            return notes
        for pdir in pdir_root.iterdir():
            troot = pdir / ".txn"
            if not troot.is_dir():
                continue
            (pdir / ".lock").unlink(missing_ok=True)
            log = pdir / "MASTER_LOG.md"
            log_text = log.read_text(encoding="utf-8") if log.exists() else ""
            for tdir in troot.iterdir():
                mpath = tdir / "manifest.json"
                if not mpath.exists():
                    shutil.rmtree(tdir, ignore_errors=True)
                    continue
                m = json.loads(mpath.read_text(encoding="utf-8"))
                if m["marker"] in log_text:
                    for e in m["files"]:
                        part = tdir / "new" / f"{e['slot']}.part"
                        if part.exists():
                            os.replace(part, confine(pdir, e["rel"]))
                    notes.append(f"Finished logged commit rev {m['revision']} in project {pdir.name}.")
                else:
                    for e in m["files"]:
                        target = confine(pdir, e["rel"])
                        bak = tdir / "old" / f"{e['slot']}.bak"
                        if e["existed"] and bak.exists():
                            os.replace(bak, target)
                        elif not e["existed"]:
                            target.unlink(missing_ok=True)
                    notes.append(f"Rolled back unlogged commit rev {m['revision']} in project {pdir.name}.")
                shutil.rmtree(tdir, ignore_errors=True)
        return notes
