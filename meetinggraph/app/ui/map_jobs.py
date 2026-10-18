"""Background work for the Map: read graph data from the saved files and (only when needed) compute the layout.
Runs in a QThread started at LowestPriority; results come back through a signal. The UI thread only creates
items and sets positions."""
from __future__ import annotations

import traceback

from PySide6.QtCore import QThread, Signal

from ..workspace.graph import build_graph
from . import tokens as tk
from .graph_layout import (cache_path, compute_layout, graph_signature, load_cache, place_near, save_cache,
                           seed_xy, topic_groups)


def visible_part(graph: dict, show_meetings: bool) -> tuple[list[dict], list[dict]]:
    nodes = [n for n in graph["nodes"] if n["type"] == "feature" or show_meetings]
    ids = {n["id"] for n in nodes}
    edges = [e for e in graph["edges"] if e["src"] in ids and e["dst"] in ids
             and (show_meetings or not e["type"].startswith("meeting_"))]
    return nodes, edges


def run_job(req: dict) -> dict:
    """Pure function (testable without Qt threads). req keys:
    token, mode ('load' | 'rearrange'), allow_layout, show_meetings, store, project_id, workspace_root, graph,
    spacing, spread."""
    graph = req.get("graph")
    if graph is None:
        store, pid = req["store"], req["project_id"]
        graph = build_graph(store, pid)
        feats = {f["id"]: f for f in store.features(pid)}
        for n in graph["nodes"]:                    # display-only extras for colors and the selection card
            if n["type"] == "feature":
                f = feats.get(n["id"], {})
                n["revision"] = f.get("revision", 1)
                desc = str(f.get("description") or "")
                n["summary"] = "" if desc == "unknown" else desc
    nodes, edges = visible_part(graph, req.get("show_meetings", False))
    path = cache_path(req["workspace_root"], req["project_id"]) if req.get("project_id") and \
        req.get("workspace_root") else None
    cache = load_cache(path) if path else {}
    positions = dict(cache.get("positions") or {})
    pinned = set(cache.get("pinned") or [])
    ids = [n["id"] for n in nodes]
    missing = [i for i in ids if i not in positions]
    rearrange = req.get("mode") == "rearrange"
    laid_out = pending = False
    link = tk.g("link.distance") * float(req.get("spacing", 1.0))
    if (rearrange or missing) and ids:
        if req.get("allow_layout", True):
            if rearrange:
                fixed = {i: positions[i] for i in pinned if i in positions and i in ids}
                # forget hidden nodes' old spots (e.g. meetings) so they're re-placed around the new layout
                positions = {k: v for k, v in positions.items() if k in pinned}
            else:
                fixed = {i: positions[i] for i in ids if i in positions}
            new = compute_layout(nodes, edges, fixed, link, tk.g("meeting.link.distance"),
                                 repel=float(req.get("spread", 1.0)))
            positions.update(new)
            laid_out = True
        else:
            # AI busy: provisional spots now (not saved); the real layout runs when it's free
            for i in missing:
                nb = [positions[e["dst"] if e["src"] == i else e["src"]] for e in edges
                      if i in (e["src"], e["dst"]) and (e["dst"] if e["src"] == i else e["src"]) in positions]
                positions[i] = place_near(i, nb, link, link * 3) if nb else seed_xy(i, link * 3)
            pending = True
    if laid_out and path:
        save_cache(path, {"positions": positions, "pinned": sorted(pinned),
                          "settings": cache.get("settings") or {}, "sig": graph_signature(nodes, edges)})
    return {"token": req.get("token"), "graph": graph, "nodes": nodes, "edges": edges,
            "positions": {i: positions[i] for i in ids}, "laid_out": laid_out, "pending": pending,
            "pinned": sorted(pinned), "settings": cache.get("settings") or {},
            "groups": topic_groups(graph["nodes"], [e for e in graph["edges"] if not e["type"].startswith("meeting_")]),
            "project_id": req.get("project_id")}


class MapJob(QThread):
    done = Signal(object)

    def __init__(self, req: dict, parent=None):
        super().__init__(parent)
        self.req = req

    def run(self):
        try:
            res = run_job(self.req)
        except Exception as e:                      # never a traceback in the UI: the Map just keeps what it has
            traceback.print_exc()
            res = {"token": self.req.get("token"), "error": str(e)}
        self.done.emit(res)


def save_settings_async(workspace_root, project_id: str, settings: dict, pinned=None, positions_update=None):
    """Write Map settings / pins / moved positions into the project's cache file off the UI thread."""
    from .common import Job
    from PySide6.QtCore import QThreadPool

    def work():
        path = cache_path(workspace_root, project_id)
        data = load_cache(path)
        data.setdefault("positions", {})
        if positions_update:
            data["positions"].update(positions_update)
        if pinned is not None:
            data["pinned"] = sorted(pinned)
        data["settings"] = dict(data.get("settings") or {}, **settings)
        save_cache(path, data)
    QThreadPool.globalInstance().start(Job(work))
