"""Graph of one project, derived only from approved files. Every edge carries its reason."""
from __future__ import annotations

from itertools import combinations
from typing import Any

from .store import Store


def build_graph(store: Store, project_id: str) -> dict[str, Any]:
    feats = store.features(project_id)
    meets = store.meetings(project_id)
    titles = {f["id"]: f["title"] for f in feats}
    mtitles = {m["id"]: m.get("title", "") for m in meets}
    nodes = [{"id": f["id"], "type": "feature", "label": f["title"], "status": f.get("status", "")} for f in feats]
    nodes += [{"id": m["id"], "type": "meeting", "label": m.get("title", ""), "date": str(m.get("date", ""))}
              for m in meets]
    edges = []
    for f in feats:
        for d in f.get("depends_on") or []:
            if d in titles:
                edges.append({"src": f["id"], "dst": d, "type": "feature_depends_on_feature",
                              "reason": f"'{f['title']}' lists '{titles[d]}' in depends_on ({f['_path']})"})
    touched_by: dict[str, set[str]] = {}
    for m in meets:
        for ref in m.get("feature_refs") or []:
            fid = ref.get("feature_id")
            if fid in titles:
                edges.append({"src": m["id"], "dst": fid, "type": "meeting_updates_feature",
                              "reason": f"Meeting '{m.get('title', '')}' ({m.get('date', '')}) approved "
                                        f"{str(ref.get('action', 'change')).upper()} of '{titles[fid]}'"})
                touched_by.setdefault(m["id"], set()).add(fid)
    for a, b in combinations(sorted(touched_by, key=lambda x: str(next((m.get('date') for m in meets if m['id'] == x), ''))), 2):
        shared = touched_by[a] & touched_by[b]
        if shared:
            names = ", ".join(sorted(titles[s] for s in shared))
            edges.append({"src": a, "dst": b, "type": "meeting_relates_to_meeting",
                          "reason": f"'{mtitles[a]}' and '{mtitles[b]}' both changed: {names}"})
    return {"nodes": nodes, "edges": edges}
