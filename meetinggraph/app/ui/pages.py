"""Readable HTML pages for a feature and a meeting, built from the saved files."""
from __future__ import annotations

import re

from ..workspace.store import RELATION_LABEL
from .common import STATUS_COLOR, esc

CSS = """
<style>
body { font-family: 'Segoe UI'; font-size: 10.5pt; color: #1f2328; }
h1 { font-size: 16pt; margin: 0 0 4px 0; }
h3 { font-size: 11pt; margin: 14px 0 4px 0; color: #1f2328; }
.muted { color: #57606a; }
.pill { color: white; padding: 2px 8px; border-radius: 8px; font-weight: 600; }
a { color: #1f6feb; text-decoration: none; }
ul { margin-top: 2px; }
</style>
"""

WIKI = re.compile(r"\[\[([0-9a-f-]{36})\|([^\]]+)\]\]")


def _links(text: str, kind_of) -> str:
    return WIKI.sub(lambda m: f"<a href='open:{kind_of(m.group(1))}:{m.group(1)}'>{esc(m.group(2))}</a>", esc(text))


def _ul(items: list[str], empty: str) -> str:
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>" if items else f"<p class='muted'>{empty}</p>"


def feature_html(f: dict, features: list[dict], meetings: list[dict]) -> str:
    titles = {x["id"]: x["title"] for x in features}
    mt = {m["id"]: (m.get("title", ""), str(m.get("date", ""))) for m in meetings}
    status = f.get("status", "")
    color = STATUS_COLOR.get(status, "#8c959f")
    out = [CSS, f"<h1>{esc(f['title'])}</h1>",
           f"<p><span class='pill' style='background:{color}'>&nbsp;{esc(status)}&nbsp;</span>"
           f" <span class='muted'>· version {f.get('revision', 1)} · updated {esc(str(f.get('updated_at', ''))[:10])}"
           "</span></p>"]
    if status == "archived":
        out.append("<p class='muted'><i>Archived: kept for history, no longer part of the plan.</i></p>")
    out.append("<h3>What it is</h3>")
    out.append(f"<p>{esc(f.get('description') or 'Not described yet.')}</p>")
    if f.get("key_points"):
        out.append("<h3>Key points</h3>" + _ul([esc(k) for k in f["key_points"]], ""))
    subs = [("☑ " if s.get("status") == "done" else "☐ ") + esc(s.get("text", "")) for s in f.get("subtasks") or []]
    out.append("<h3>Subtasks</h3>" + _ul(subs, "None recorded."))
    rels = f.get("relations") or []

    def why(r):
        return (f" — <span class='muted'>why:</span> {esc(r['why'])}" if r.get("why")
                else " — <span class='muted'>no reason was given</span>")

    def link(fid, fallback=""):
        return f"<a href='open:feature:{fid}'>{esc(titles[fid])}</a>" if fid in titles else esc(fallback or fid)

    deps = []
    for d in f.get("depends_on") or []:
        r = next((r for r in rels if r.get("type") == "needs" and r.get("to") == d), {})
        deps.append(link(d) + why(r))
    out.append("<h3>Needs first</h3>" + _ul(deps, "Nothing — it doesn't depend on other features."))
    rev = []
    for x in features:
        if f["id"] in (x.get("depends_on") or []) and x.get("status") != "archived":
            r = next((r for r in x.get("relations") or [] if r.get("type") == "needs" and r.get("to") == f["id"]), {})
            rev.append(link(x["id"]) + why(r))
    out.append("<h3>Needed by</h3>" + _ul(rev, "No other feature depends on this one."))
    conn = [f"{esc(RELATION_LABEL.get(r.get('type'), 'is related to'))} {link(r.get('to'), r.get('to_title'))}{why(r)}"
            for r in rels if r.get("type") != "needs"]
    conn += [f"{link(x['id'])} {esc(RELATION_LABEL.get(r.get('type'), 'is related to'))} this{why(r)}"
             for x in features if x["id"] != f["id"]
             for r in x.get("relations") or [] if r.get("type") != "needs" and r.get("to") == f["id"]]
    if conn:
        out.append("<h3>Other connections</h3>" + _ul(conn, ""))
    out.append(f"<h3>Impact</h3><p>{esc(f.get('impact') or 'Not stated.')}</p>")
    src = [f"<a href='open:meeting:{m}'>{esc(mt.get(m, (m, ''))[0])}</a> <span class='muted'>"
           f"{esc(mt.get(m, ('', ''))[1])}</span>" for m in f.get("source_meetings") or []]
    out.append("<h3>Discussed in</h3>" + _ul(src, "—"))
    said = [f"<span class='muted'>[{esc(e.get('at', ''))}] {esc(mt.get(e.get('meeting'), ('', ''))[0])}</span> "
            f"{esc(e.get('text', ''))}" for e in f.get("excerpts") or []]
    if said:
        out.append("<h3>What was said</h3>" + _ul(said, ""))
    hist = [_links(h, lambda i: "meeting") for h in f.get("history") or []]
    out.append("<h3>History</h3>" + _ul(hist, "—"))
    return "".join(out)


def meeting_html(m: dict, features: list[dict], meetings: list[dict], transcript_saved: bool = False) -> str:
    titles = {x["id"]: x["title"] for x in features}
    body = m.get("_body", "")

    def section(name: str) -> list[str]:
        mm = re.search(rf"^## {re.escape(name)}\n(.*?)(?=^## |\Z)", body, re.S | re.M)
        if not mm:
            return []
        return [ln[2:].strip() for ln in mm.group(1).splitlines() if ln.startswith("- ") and ln[2:].strip() != "none"]

    summary = re.search(r"^## Summary\n(.*?)(?=^## |\Z)", body, re.S | re.M)
    words = {"add": "Added", "change": "Updated", "archive": "Archived"}
    changes = [f"{words.get(r.get('action'), 'Changed')} <a href='open:feature:{r['feature_id']}'>"
               f"{esc(titles.get(r['feature_id'], '?'))}</a>" for r in m.get("feature_refs") or []]
    mine = {r["feature_id"] for r in m.get("feature_refs") or []}
    related = []
    for x in meetings:
        if x["id"] == m["id"]:
            continue
        shared = mine & {r["feature_id"] for r in x.get("feature_refs") or []}
        if shared:
            related.append(f"<a href='open:meeting:{x['id']}'>{esc(x.get('title', ''))}</a> "
                           f"<span class='muted'>{esc(str(x.get('date', '')))} · both changed "
                           f"{esc(', '.join(sorted(titles.get(s, '?') for s in shared)))}</span>")
    clean = lambda s: esc(re.sub(r"\s*_\(at [^)]*\)_", "", s))  # noqa: E731
    return "".join([
        CSS, f"<h1>{esc(m.get('title', ''))}</h1>",
        f"<p class='muted'>{esc(str(m.get('date', '')))} · saved as project version {m.get('revision', '')} · "
        + (f"<a href='open:transcript:{m['id']}'>Open the full transcript</a>" if transcript_saved
           else "the transcript was deleted after saving") + "</p>",
        "<h3>Summary</h3>", f"<p>{esc(summary.group(1).strip()) if summary else '—'}</p>",
        "<h3>Decisions</h3>", _ul([clean(x) for x in section("Decisions")], "None."),
        "<h3>Action items</h3>", _ul([clean(x) for x in section("Action items")], "None."),
        "<h3>Open questions</h3>", _ul([clean(x) for x in section("Open questions")], "None."),
        "<h3>Changes saved to the project</h3>", _ul(changes, "None."),
        "<h3>Related meetings</h3>", _ul(related, "None yet."),
    ])
