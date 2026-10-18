"""Turn an approved proposal into a validated set of file writes (a CommitPlan).

The proposal's change cards describe the *desired final state* of each feature.
Nothing here writes files; Transactions.commit does that with the plan's output.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .paths import check_id, clean_name, new_id
from .store import SCHEMA_VERSION, STATUSES, Store, now_iso, render_feature, render_meeting


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


@dataclass
class CommitPlan:
    project_id: str
    revision: int
    meeting_id: str
    files: dict[str, str] = field(default_factory=dict)
    log_entry: str = ""
    preview: list[tuple[str, str, str]] = field(default_factory=list)  # (action, file, detail)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _subtasks(raw: list, existing: list[dict] | None) -> list[dict]:
    existing = existing or []
    by_text = {norm(s.get("text", "")): s for s in existing}
    out, used = [], set()
    n = 1 + max([int(str(s.get("id", "st-0")).split("-")[-1] or 0) for s in existing if
                 str(s.get("id", "")).startswith("st-") and str(s.get("id")).split("-")[-1].isdigit()] or [0])
    for item in raw or []:
        if isinstance(item, dict):
            text, status = item.get("text", ""), item.get("status", "open")
        else:
            text, status = str(item), "open"
        text = text.strip()
        if text.startswith("[x]") or text.startswith("[X]"):
            text, status = text[3:].strip(), "done"
        elif text.startswith("[ ]"):
            text = text[3:].strip()
        if not text or norm(text) in used:
            continue
        used.add(norm(text))
        prev = by_text.get(norm(text))
        if prev:
            out.append({"id": prev["id"], "text": text, "status": status or prev.get("status", "open")})
        else:
            out.append({"id": f"st-{n}", "text": text, "status": status if status in ("open", "done") else "open"})
            n += 1
    return out


def _find_cycle(graph: dict[str, list[str]]) -> list[str] | None:
    WHITE, GREY, BLACK = 0, 1, 2
    color = {k: WHITE for k in graph}
    stack: list[str] = []

    def visit(u: str) -> list[str] | None:
        color[u] = GREY
        stack.append(u)
        for v in graph.get(u, []):
            if color.get(v, WHITE) == GREY:
                return stack[stack.index(v):] + [v]
            if color.get(v, WHITE) == WHITE and v in graph:
                c = visit(v)
                if c:
                    return c
        stack.pop()
        color[u] = BLACK
        return None

    for k in list(graph):
        if color[k] == WHITE:
            c = visit(k)
            if c:
                return c
    return None


def build_commit(store: Store, proposal: dict[str, Any]) -> CommitPlan:
    pid = proposal.get("project_id")
    is_new_project = not pid
    project = None
    if pid:
        check_id(pid)
        project = store.project(pid)
        if project is None:
            is_new_project = True
    if is_new_project:
        pid = pid or new_id()
        project = {"schema_version": SCHEMA_VERSION, "id": pid,
                   "name": clean_name(proposal.get("project_name") or "Untitled project"),
                   "created_at": now_iso(), "revision": 0, "features": {}, "meetings": []}

    revision = int(project.get("revision", 0)) + 1
    meeting_id = new_id()
    plan = CommitPlan(project_id=pid, revision=revision, meeting_id=meeting_id)

    base = proposal.get("base_revision")
    if not is_new_project and base is not None and int(base) != int(project.get("revision", 0)):
        plan.errors.append(
            f"The project changed since this proposal was made (revision {base} → {project.get('revision')}). "
            "Use 'Re-match features' and review again.")
        return plan

    existing = {f["id"]: f for f in store.features(pid)} if not is_new_project else {}
    changes = [c for c in proposal.get("changes", []) if c.get("include", True)]

    # ---- assign ids and gather final states ----
    final: dict[str, dict] = {}      # feature_id -> final meta (no history)
    touched: dict[str, str] = {}     # feature_id -> action
    change_for: dict[str, dict] = {}
    for c in changes:
        action = c.get("action")
        if action not in ("add", "change", "archive"):
            plan.errors.append(f"Card '{c.get('title')}': unknown action {action!r}.")
            continue
        if action == "add":
            fid = new_id()
            c["_assigned_id"] = fid
        else:
            fid = c.get("feature_id")
            if not fid or fid not in existing:
                plan.errors.append(f"Card '{c.get('title')}': choose which existing feature to {action}.")
                continue
            if fid in touched:
                plan.errors.append(f"Two cards change the same feature '{existing[fid]['title']}'. Merge or remove one.")
                continue
            if c.get("needs_confirmation") and not c.get("match_confirmed"):
                plan.errors.append(f"Answer the question on the change '{c.get('title')}': is it the same "
                                   f"as your existing feature '{existing[fid]['title']}'?")
                continue
        touched[fid] = action
        change_for[fid] = c

    if plan.errors:
        return plan

    # title -> id map for dependency resolution (current + new adds)
    title_map: dict[str, str] = {}
    for fid, f in existing.items():
        title_map[norm(f["title"])] = fid
    for fid, c in change_for.items():
        if c.get("title"):
            title_map[norm(c["title"])] = fid

    def resolve(ref: str, owner_title: str) -> str | None:
        ref = (ref or "").strip()
        if not ref:
            return None
        if ref in existing or ref in change_for:
            return ref
        if ref.startswith("new:"):
            for fid, c in change_for.items():
                if c.get("change_id") == ref[4:]:
                    return fid
        hit = title_map.get(norm(ref))
        if hit:
            return hit
        plan.errors.append(f"'{owner_title}' depends on '{ref}', which is not a feature in this project or proposal.")
        return None

    ts = now_iso()
    for fid, c in change_for.items():
        action = touched[fid]
        prev = existing.get(fid)
        if action == "archive":
            meta = {k: v for k, v in prev.items() if not k.startswith("_")}
            meta["status"] = "archived"
        else:
            title = clean_name(c.get("title") or (prev or {}).get("title") or "")
            status = c.get("status") or (prev or {}).get("status") or "proposed"
            if status not in STATUSES:
                plan.errors.append(f"'{title}': invalid status {status!r}.")
                continue
            deps = []
            for r in c.get("depends_on") or []:
                rid = resolve(r, title)
                if rid and rid not in deps:
                    deps.append(rid)
            meta = {
                "id": fid, "title": title, "status": status,
                "description": (c.get("description") or "").strip() or (prev or {}).get("description") or "unknown",
                "subtasks": _subtasks(c.get("subtasks"), (prev or {}).get("subtasks")),
                "depends_on": deps,
                "impact": (c.get("impact") or "").strip() or (prev or {}).get("impact") or "unknown",
                "source_meetings": list((prev or {}).get("source_meetings") or []),
                "created_at": (prev or {}).get("created_at") or ts,
                "history": list((prev or {}).get("history") or []),
                "revision": int((prev or {}).get("revision", 0)),
            }
        meta["updated_at"] = ts
        meta["revision"] = int(meta.get("revision", 0)) + 1
        if meeting_id not in meta.setdefault("source_meetings", []):
            meta["source_meetings"].append(meeting_id)
        final[fid] = meta

    if plan.errors:
        return plan

    # ---- archive policy for dependents ----
    all_now = {fid: {k: v for k, v in f.items() if not k.startswith("_")} for fid, f in existing.items()}
    all_now.update(final)
    archived = {fid for fid, f in all_now.items() if f.get("status") == "archived"}
    policy = proposal.get("archive_policy")
    for fid, f in list(all_now.items()):
        if f.get("status") == "archived":
            continue
        bad = [d for d in f.get("depends_on") or [] if d in archived]
        if not bad:
            continue
        names = ", ".join(all_now[d]["title"] for d in bad)
        if policy in ("remove_dependency", "mark_blocked"):
            f = dict(f)
            f["depends_on"] = [d for d in f["depends_on"] if d not in archived]
            note = f"removed dependency on archived {names}"
            if policy == "mark_blocked":
                f["status"] = "blocked"
                note += "; marked blocked"
            if fid not in final:
                f["revision"] = int(f.get("revision", 0)) + 1
                f["updated_at"] = ts
                f.setdefault("history", [])
            f["_auto_note"] = note
            final[fid] = f
            all_now[fid] = f
            touched.setdefault(fid, "change")
        else:
            plan.errors.append(
                f"'{f['title']}' depends on archived {names}. On the archive change, choose what happens "
                "to features that depend on it.")
    if plan.errors:
        return plan

    # ---- graph checks ----
    for fid, f in all_now.items():
        deps = f.get("depends_on") or []
        if fid in deps:
            plan.errors.append(f"'{f['title']}' cannot depend on itself.")
        for d in deps:
            if d not in all_now:
                plan.errors.append(f"'{f['title']}' depends on a missing feature ({d}).")
    graph = {fid: [d for d in f.get("depends_on") or [] if d in all_now]
             for fid, f in all_now.items() if f.get("status") != "archived"}
    cyc = _find_cycle(graph)
    if cyc:
        plan.errors.append("These features would need each other in a loop: "
                           + " → ".join(all_now[x]["title"] for x in cyc)
                           + ". Click “Edit details” on one of them and remove the other from “Needs first”.")
    if plan.errors:
        return plan

    # ---- render files ----
    m = proposal.get("meeting") or {}
    titles = {fid: f["title"] for fid, f in all_now.items()}
    meeting_title = clean_name(m.get("title") or "Meeting")
    meeting_titles = {mm["id"]: mm.get("title", "") for mm in store.meetings(pid)} if not is_new_project else {}
    meeting_titles[meeting_id] = meeting_title

    log_lines = [f"## Revision {revision} · {ts} · meeting {meeting_id} \"{meeting_title}\""]
    edited = sum(1 for c in changes if c.get("user_edited"))
    log_lines.append(f"- Approved by user; {len(changes)} change(s), {edited} edited before approval")
    change_lines = []

    def dep_names(ids):
        return ", ".join(titles.get(i, i) for i in ids) or "none"

    def short(v, n=60):
        v = str(v)
        return v if len(v) <= n else v[: n - 1] + "…"

    for fid, f in final.items():
        action = touched.get(fid, "change")
        prev = existing.get(fid)
        rel = f"features/{fid}.md"
        diffs: list[str] = []
        if action == "change" and prev:
            for k in ("title", "status", "description", "impact"):
                old_v, new_v = str(prev.get(k) or ""), str(f.get(k) or "")
                if old_v == new_v:
                    continue
                if old_v and old_v not in ("unknown",) and new_v.startswith(old_v.rstrip(".")):
                    diffs.append(f"{k} appended: {short(new_v[len(old_v.rstrip('.')):].lstrip('. '))!r}")
                else:
                    diffs.append(f"{k}: {short(old_v)!r} → {short(new_v)!r}")
            old_d, new_d = set(prev.get("depends_on") or []), set(f.get("depends_on") or [])
            if old_d != new_d:
                diffs.append(f"depends_on +[{dep_names(sorted(new_d - old_d))}] -[{dep_names(sorted(old_d - new_d))}]")
            old_s = {norm(s['text']) for s in prev.get("subtasks") or []}
            added = [s["text"] for s in f.get("subtasks") or [] if norm(s["text"]) not in old_s]
            if added:
                diffs.append("subtasks +" + "; ".join(added))
        what = {"add": "created", "archive": "archived (tombstone)"}.get(action) \
            or f.get("_auto_note") or ("; ".join(diffs) or "reviewed, no field changes")
        f.pop("_auto_note", None)
        f.setdefault("history", []).append(f"r{f['revision']} · {ts[:10]} · [[{meeting_id}|{meeting_title}]]: {what}")
        meta = {k: f[k] for k in ["id", "title", "status", "description", "subtasks", "depends_on",
                                  "impact", "source_meetings", "created_at", "updated_at", "revision", "history"]
                if k in f}
        plan.files[rel] = render_feature(meta, titles, meeting_titles, meta.get("history", []))

        if action == "add":
            detail = f"ADD {fid} {f['title']}: depends_on {dep_names(f['depends_on'])}"
            plan.preview.append(("create", rel, f"New feature '{f['title']}'"))
        elif action == "archive":
            detail = f"ARCHIVE {fid} {f['title']} (tombstone)"
            plan.preview.append(("update", rel, f"Archive '{f['title']}'"))
        else:
            detail = f"CHANGE {fid} {f['title']}: {what}"
            plan.preview.append(("update", rel, f"Change '{f['title']}': {what}"))
        log_lines.append("- " + detail)
        change_lines.append(f"- {action.upper()} [[{fid}|{f['title']}]]")

    # meeting file
    mmeta = {
        "id": meeting_id, "title": meeting_title, "date": m.get("date") or ts[:10],
        "participants": m.get("participants") or [],
        "committed_at": ts, "revision": revision,
        "feature_refs": [{"feature_id": fid, "action": touched.get(fid, "change")} for fid in final],
        "transcript": "temporary; deleted after verified commit",
    }
    rel_m = f"meetings/{meeting_id}.md"
    plan.files[rel_m] = render_meeting(mmeta, m.get("summary", ""), m.get("decisions") or [],
                                       m.get("action_items") or [], m.get("open_questions") or [], change_lines)
    plan.preview.append(("create", rel_m, f"Meeting summary '{meeting_title}'"))

    # project.json
    project = dict(project)
    project["revision"] = revision
    project["updated_at"] = ts
    feats = dict(project.get("features") or {})
    for fid, f in final.items():
        feats[fid] = {"title": f["title"], "status": f["status"]}
    project["features"] = feats
    project["meetings"] = list(project.get("meetings") or []) + [meeting_id]
    plan.files["project.json"] = json.dumps(project, indent=2) + "\n"
    plan.preview.append(("create" if is_new_project else "update", "project.json", f"Revision {revision}"))
    plan.preview.append(("append", "MASTER_LOG.md", f"Revision {revision} entry"))

    plan.log_entry = "\n".join(log_lines)
    for c in changes:
        for w in c.get("flags") or []:
            plan.warnings.append(f"'{c.get('title')}': {w}")
    return plan
