"""Map layout engine: computed rarely, cached per project, incremental, deterministic, waits for the AI."""
from __future__ import annotations

import json
import random
import time

import numpy as np

from app.ui.graph_layout import cache_path, compute_layout, load_cache
from app.ui.map_jobs import run_job


def gen(n, seed=1, meetings=0):
    r = random.Random(seed)
    nodes = [{"id": f"f{i}", "type": "feature", "label": f"Feature {i}", "status": "proposed"} for i in range(n)]
    edges = [{"src": f"f{i}", "dst": f"f{r.randrange(i)}", "type": "feature_relates_to_feature", "verb": "uses"}
             for i in range(1, n) if r.random() < 0.9]
    for j in range(meetings):
        nodes.append({"id": f"m{j}", "type": "meeting", "label": f"Meeting {j}"})
    edges += [{"src": f"m{i % meetings}", "dst": f"f{i}", "type": "meeting_updates_feature"}
              for i in range(n)] if meetings else []
    return {"nodes": nodes, "edges": edges}


def req(tmp, g, **kw):
    """A request that uses the cache in tmp but takes the graph directly (no store needed)."""
    return dict({"token": 1, "mode": "load", "allow_layout": True, "show_meetings": False,
                 "project_id": "11111111-1111-4111-8111-111111111111", "workspace_root": tmp, "graph": g}, **kw)


def test_300_nodes_under_two_seconds_and_no_overlap():
    g = gen(300)
    t = time.perf_counter()
    pos = compute_layout(g["nodes"], g["edges"])
    assert time.perf_counter() - t < 2.0
    P = np.array(list(pos.values()))
    d = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
    np.fill_diagonal(d, 1e9)
    assert d.min() >= 4 + 4 + 8 - 0.5                    # smallest nodes never overlap


def test_deterministic():
    g = gen(40)
    assert compute_layout(g["nodes"], g["edges"]) == compute_layout(g["nodes"], g["edges"])


def test_first_open_lays_out_and_caches_then_reopen_does_no_layout(tmp_path):
    g = gen(30)
    a = run_job(req(tmp_path, g))
    assert a["laid_out"] and cache_path(tmp_path, req(tmp_path, g)["project_id"]).exists()
    b = run_job(req(tmp_path, g))
    assert not b["laid_out"] and b["positions"] == a["positions"]     # just loads saved positions


def test_new_node_does_not_reshuffle_existing(tmp_path):
    g = gen(30)
    a = run_job(req(tmp_path, g))
    g2 = json.loads(json.dumps(g))
    g2["nodes"].append({"id": "new", "type": "feature", "label": "New one"})
    g2["edges"].append({"src": "new", "dst": "f7", "type": "feature_relates_to_feature"})
    b = run_job(req(tmp_path, g2))
    assert b["laid_out"]
    assert all(b["positions"][k] == v for k, v in a["positions"].items())
    nx, ny = b["positions"]["new"]
    fx, fy = b["positions"]["f7"]
    assert ((nx - fx) ** 2 + (ny - fy) ** 2) ** 0.5 < 400                # placed near what it connects to


def test_waits_while_ai_busy_and_does_not_save_provisional_spots(tmp_path):
    g = gen(20)
    r = run_job(req(tmp_path, g, allow_layout=False))
    assert r["pending"] and not r["laid_out"] and len(r["positions"]) == 20
    assert not cache_path(tmp_path, req(tmp_path, g)["project_id"]).exists()


def test_corrupted_cache_is_ignored(tmp_path):
    g = gen(10)
    p = cache_path(tmp_path, req(tmp_path, g)["project_id"])
    p.parent.mkdir(parents=True)
    p.write_text("{not json", encoding="utf-8")
    assert load_cache(p) == {}
    assert run_job(req(tmp_path, g))["laid_out"]


def test_rearrange_keeps_pinned(tmp_path):
    g = gen(25)
    a = run_job(req(tmp_path, g))
    p = cache_path(tmp_path, req(tmp_path, g)["project_id"])
    data = json.loads(p.read_text())
    data["positions"]["f3"] = [999.0, -999.0]
    data["pinned"] = ["f3"]
    p.write_text(json.dumps(data))
    b = run_job(req(tmp_path, g, mode="rearrange"))
    assert b["laid_out"] and tuple(b["positions"]["f3"]) == (999.0, -999.0)


def test_meetings_laid_out_only_when_shown_and_on_the_outside(tmp_path):
    g = gen(40, meetings=4)
    a = run_job(req(tmp_path, g))
    assert not any(k.startswith("m") for k in a["positions"])
    b = run_job(req(tmp_path, g, show_meetings=True))
    assert b["laid_out"]
    assert all(b["positions"][k] == v for k, v in a["positions"].items())      # features didn't move
    F = np.array([v for k, v in b["positions"].items() if k.startswith("f")])
    M = np.array([v for k, v in b["positions"].items() if k.startswith("m")])
    c = F.mean(0)
    assert np.hypot(*(M - c).T).min() > np.hypot(*(F - c).T).max()
