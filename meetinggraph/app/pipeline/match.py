"""Draft + current project state -> proposal (one editable card per feature change).

Cards hold the *final desired state* of a feature. Matching to existing features is
by stable ID once confirmed; title similarity only produces candidates for the user.
"""
from __future__ import annotations

import difflib
import json
import re

from ..workspace.store import Store
from .extract import merge_relations
from .transcript import excerpts_for, norm

SYNONYMS = [
    (r"\b(sign in|signin|sign on|log in|logon|authentication|auth)\b", "login"),
    (r"\b(sync|syncing|synchronization|synchronisation)\b", "sync"),
    (r"\b(pdf export|export pdf)\b", "export pdf"),
]


def canon(title: str) -> str:
    t = norm(title)
    for pat, rep in SYNONYMS:
        t = re.sub(pat, rep, t)
    t = re.sub(r"\b(feature|flow|the|user|users)\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def similarity(a: str, b: str) -> float:
    ca, cb = canon(a), canon(b)
    if not ca or not cb:
        return 0.0
    if ca == cb:
        return 1.0
    ta, tb = set(ca.split()), set(cb.split())
    jac = len(ta & tb) / len(ta | tb)
    seq = difflib.SequenceMatcher(None, ca, cb).ratio()
    return round(max(jac, seq * 0.9), 3)


def candidates(title: str, existing: list[dict], k: int = 3) -> list[dict]:
    scored = [{"feature_id": f["id"], "title": f["title"], "status": f.get("status", ""),
               "score": similarity(title, f["title"])} for f in existing]
    scored = [s for s in scored if s["score"] >= 0.35]
    return sorted(scored, key=lambda s: s["score"], reverse=True)[:k]


def _card_from_existing(card: dict, f: dict, titles: dict[str, str]) -> dict:
    """Merge a mention onto an existing feature so the card shows the final state."""
    subs = [("[x] " if s.get("status") == "done" else "") + s["text"] for s in f.get("subtasks") or []]
    seen = {norm(s.replace("[x] ", "")) for s in subs}
    subs += [s for s in card["subtasks"] if norm(s) not in seen]
    deps = [titles.get(d, d) for d in f.get("depends_on") or []]
    seen = {norm(d) for d in deps}
    deps += [d for d in card["depends_on"] if norm(d) not in seen]
    old_desc, new_desc = (f.get("description") or "").strip(), (card["description"] or "").strip()
    if old_desc in ("", "unknown"):
        desc = new_desc
    elif new_desc and norm(new_desc) not in norm(old_desc):
        desc = f"{old_desc.rstrip('.')}. Update: {new_desc}"
    else:
        desc = old_desc
    old_rels = [{"to": titles.get(r.get("to"), r.get("to_title") or r.get("to", "")), "type": r.get("type", "related"),
                 "why": r.get("why", "")} for r in f.get("relations") or []]
    old_kp = list(f.get("key_points") or [])
    kp = old_kp + [k for k in card.get("key_points") or [] if norm(k) not in {norm(x) for x in old_kp}]
    card.update({
        "key_points": kp,
        "relations": merge_relations(old_rels, card.get("relations") or []),
        "feature_id": f["id"],
        "title": f["title"],
        "description": desc,
        "status": card["status"] or f.get("status", "active"),
        "subtasks": subs, "depends_on": deps,
        "impact": card["impact"] or f.get("impact", ""),
    })
    return card


FORWARD_CUES = ("depends on", "depend on", "depending on", "needs", "need", "requires", "require",
                "blocked by", "waits for", "wait for", "after", "once", "can t happen before",
                "cannot happen before", "can not happen before", "not before")


def _pos(title: str, text_canon: str) -> int:
    t = canon(title)
    if not t:
        return -1
    m = re.search(r"\b" + re.escape(t) + r"\b", text_canon)
    return m.start() if m else -1


def _direction_score(a: str, b: str, segments: list[dict]) -> int:
    """+2 if the transcript says "a needs b", -2 if it says "b needs a", 0 if it doesn't say."""
    score = 0
    for seg in segments or []:
        t = canon(seg.get("text", ""))
        pa, pb = _pos(a, t), _pos(b, t)
        if pa < 0 or pb < 0 or pa == pb:
            continue
        lo, hi = sorted((pa, pb))
        between = t[lo:hi]
        if any(re.search(r"\b" + c + r"\b", between) for c in FORWARD_CUES):
            score = max(score, 2) if pa < pb else min(score, -2) if score <= 0 else score
    return score


def _fix_dependencies(cards: list[dict], existing: list[dict], segments: list[dict]) -> None:
    """Remove model-suggested dependencies that the transcript states the other way round, and break
    loops by dropping the least-supported suggested link. Saved (existing) links are never removed."""
    by_key = {}
    for f in existing:
        by_key[canon(f["title"])] = f["title"]
    for c in cards:
        if c.get("include", True):
            by_key.setdefault(canon(c["title"]), c["title"])
    saved = {(canon(f["title"]), canon(d_title)) for f in existing
             for d_title in [x["title"] for x in existing if x["id"] in (f.get("depends_on") or [])]}

    def note(c, msg):
        if msg not in c["flags"]:
            c["flags"].append(msg)

    edges = {}   # (from_key, to_key) -> (card, dep_text, score)
    for c in cards:
        if not c.get("include", True) or c["action"] == "archive":
            continue
        keep = []
        for d in c.get("depends_on") or []:
            a, b = canon(c["title"]), canon(d)
            if a == b:
                note(c, f"Removed “needs {d} first”: a feature can't depend on itself.")
                continue
            sc = 5 if (a, b) in saved else _direction_score(c["title"], d, segments)
            if sc < 0:
                note(c, f"Removed “needs {d} first”: the meeting said it the other way round "
                        f"({d} needs {c['title']}).")
                continue
            keep.append(d)
            edges[(a, b)] = (c, d, sc)
        c["depends_on"] = keep
    # break loops
    for _ in range(20):
        graph: dict[str, list[str]] = {}
        for (a, b) in edges:
            graph.setdefault(a, []).append(b)
        cycle = _cycle(graph)
        if not cycle:
            return
        pairs = [(cycle[i], cycle[i + 1]) for i in range(len(cycle) - 1)]
        weakest = min(pairs, key=lambda e: edges[e][2] if e in edges else 99)
        if weakest not in edges or edges[weakest][2] >= 5:
            return       # only saved links left in the loop; the save check will explain it
        c, d, _sc = edges.pop(weakest)
        c["depends_on"] = [x for x in c["depends_on"] if x != d]
        loop = " → ".join(by_key.get(k, k) for k in cycle)
        note(c, f"Removed “needs {d} first”: it would create a loop ({loop}). Add it back under "
                "“Edit details” only if it is really needed.")


def sync_relations(card: dict) -> None:
    """'Needs first' (depends_on) is the source of truth for dependencies: drop 'needs' connections that are
    no longer in it and add an entry (without a reason yet) for each dependency that has none."""
    deps = {norm(d): d for d in card.get("depends_on") or []}
    rels = [r for r in card.get("relations") or [] if r.get("type") != "needs" or norm(r.get("to", "")) in deps]
    have = {norm(r["to"]) for r in rels if r.get("type") == "needs"}
    rels += [{"to": d, "type": "needs", "why": ""} for k, d in deps.items() if k not in have]
    card["relations"] = rels


def _cycle(graph: dict[str, list[str]]) -> list[str] | None:
    seen, stack = {}, []

    def visit(u):
        seen[u] = 1
        stack.append(u)
        for v in graph.get(u, []):
            if seen.get(v) == 1:
                return stack[stack.index(v):] + [v]
            if not seen.get(v):
                r = visit(v)
                if r:
                    return r
        stack.pop()
        seen[u] = 2
        return None

    for k in list(graph):
        if not seen.get(k):
            r = visit(k)
            if r:
                return r
    return None


def build_proposal(store: Store, project_id: str | None, project_name: str, draft: dict, session_id: str,
                   date: str, participants: list[str] | None = None, segments: list[dict] | None = None) -> dict:
    existing = store.features(project_id) if project_id else []
    project = store.project(project_id) if project_id else None
    titles = {f["id"]: f["title"] for f in existing}
    active = [f for f in existing if f.get("status") != "archived"]
    by_id = {f["id"]: f for f in existing}

    cards = []
    for i, m in enumerate(draft.get("feature_mentions", []), 1):
        supported = [e for e in m.get("evidence", []) if e.get("supported")]
        card = {
            "change_id": f"c{i}", "include": True, "action": m["action"], "feature_id": None,
            "needs_confirmation": False, "match_confirmed": False,
            "title": m["title"], "description": m.get("description", ""), "status": m.get("status", ""),
            "subtasks": list(m.get("subtasks", [])), "depends_on": list(m.get("depends_on", [])),
            "impact": m.get("impact", ""), "uncertainty": m.get("uncertainty", ""),
            "evidence": m.get("evidence", []), "candidates": [], "flags": [], "user_edited": False,
            "said_as": m["title"], "mention": json.loads(json.dumps(m)),
            "key_points": list(m.get("key_points", [])), "relations": [dict(r) for r in m.get("relations", [])],
            "excerpts": excerpts_for(m, segments or [], [r["to"] for r in m.get("relations", [])]),
        }
        pool = active if m["action"] != "archive" else active
        cands = candidates(m["title"], pool)
        card["candidates"] = cands
        if m["action"] == "add":
            if cands and cands[0]["score"] >= 0.6:
                card["flags"].append(f"Possible duplicate of existing '{cands[0]['title']}'. "
                                     "If it is the same feature, pick it under 'Existing feature'.")
            card["status"] = card["status"] or "proposed"
        else:
            if cands:
                best = by_id[cands[0]["feature_id"]]
                _card_from_existing(card, best, titles)
                card["needs_confirmation"] = True
                # Exact same name: pre-answer "yes" (still shown to the user before saving).
                card["match_confirmed"] = norm(best["title"]) == norm(m["title"])
                if norm(best["title"]) != norm(m["title"]):
                    card["flags"].append(f"Meeting said '{m['title']}'; matched to existing '{best['title']}'. Confirm.")
            elif m["action"] == "archive":
                card["include"] = False
                card["flags"].append(f"Asked to archive '{m['title']}', but no such feature exists in this project.")
            else:
                card["action"] = "add"
                card["status"] = card["status"] or "proposed"
                card["flags"].append(f"No existing feature matches '{m['title']}'; proposed as a new feature.")
        if not supported:
            card["include"] = False
            card["flags"].append("No supporting transcript evidence found; excluded until you check it.")
        if card["uncertainty"]:
            card["flags"].append(f"Uncertain: {card['uncertainty']}")
        cards.append(card)

    _fix_dependencies(cards, existing, segments or [])
    for c in cards:
        sync_relations(c)
    meeting = {
        "title": draft.get("title") or "Meeting", "date": date, "participants": participants or [],
        "summary": draft.get("summary", ""),
        "decisions": [_item(x) for x in draft.get("decisions", [])],
        "action_items": [dict(_item(x), owner=x.get("owner", "")) for x in draft.get("action_items", [])],
        "open_questions": [_item(x) for x in draft.get("open_questions", [])],
    }
    return {
        "version": 1, "session_id": session_id, "project_id": project_id,
        "project_name": (project or {}).get("name") or project_name,
        "base_revision": (project or {}).get("revision", 0), "archive_policy": None,
        "meeting": meeting, "changes": cards,
    }


def _item(x: dict) -> dict:
    from .transcript import fmt_ts
    o = {"text": x["text"], "supported": x.get("supported", False)}
    if x.get("supported"):
        o["evidence"] = {"start": fmt_ts(x.get("start", 0)), "end": fmt_ts(x.get("end", 0))}
    return o
