"""Map layout: a bounded force-directed pass (numpy), run rarely and off the UI thread, plus the per-project
position cache. Pure functions; no Qt widgets here (safe to call from a worker thread).

- repulsion between all pairs, springs on links (features ~link_distance; meetings longer and weaker),
  gentle pull to the centre, collision so nodes never overlap (r1 + r2 + 8 px, plus room for labels)
- at most 200 iterations, stops early when nodes barely move, hard cap of 2 seconds
- deterministic: start positions come from a hash of the node id, so the same data gives the same layout
- incremental: existing / pinned nodes stay fixed; new nodes start next to the nodes they connect to
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

LABEL_ROW = 18.0          # px: label height + gap, kept free under each node
LABEL_CHAR = 6.6          # px: average width of a 12 px label character (for side-by-side label room)
MAX_LABEL = 24
REPEL_SCALE = 0.25        # tuned so links come out near link_distance (the 'Spread' slider multiplies it)


# ------------------------------------------------------------------ helpers
def _h(nid: str, salt: str = "") -> float:
    return int.from_bytes(hashlib.md5((salt + nid).encode()).digest()[:4], "little") / 2 ** 32


def seed_xy(nid: str, spread: float) -> tuple[float, float]:
    a, r = _h(nid, "a") * 2 * math.pi, math.sqrt(_h(nid, "r")) * spread
    return r * math.cos(a), r * math.sin(a)


def radius_for(degree: int, kind: str) -> float:
    if kind == "meeting":
        return 5.0
    return max(4.0, min(14.0, 4.0 + 2.2 * math.sqrt(degree)))


def label_w(label: str) -> float:
    return min(len(label or ""), MAX_LABEL) * LABEL_CHAR + 4


def graph_signature(nodes: list[dict], edges: list[dict]) -> str:
    ids = sorted(n["id"] for n in nodes)
    es = sorted(f"{e['src']}>{e['dst']}:{e['type']}" for e in edges)
    return hashlib.sha1(json.dumps([ids, es]).encode()).hexdigest()


def place_near(nid: str, neighbours: list[tuple[float, float]], link_distance: float,
               spread: float) -> tuple[float, float]:
    """Cheap provisional spot for a new node: next to its neighbours, or a hashed spot near the centre."""
    if neighbours:
        cx = sum(p[0] for p in neighbours) / len(neighbours)
        cy = sum(p[1] for p in neighbours) / len(neighbours)
        a = _h(nid, "n") * 2 * math.pi
        return cx + math.cos(a) * link_distance * 0.7, cy + math.sin(a) * link_distance * 0.7
    return seed_xy(nid, spread)


# ------------------------------------------------------------------ the layout
def compute_layout(nodes: list[dict], edges: list[dict], fixed: dict[str, tuple[float, float]] | None = None,
                   link_distance: float = 90.0, meeting_distance: float = 160.0, repel: float = 1.0,
                   max_iter: int = 200, time_cap: float = 2.0) -> dict[str, tuple[float, float]]:
    """Features by force layout; meetings (if any) afterwards on an outer ring, each at the angle of the
    features it touches, so they sit on the edge instead of pulling the middle together."""
    fixed = fixed or {}
    feats = [nd for nd in nodes if nd["type"] != "meeting"]
    meets = [nd for nd in nodes if nd["type"] == "meeting"]
    fids = {nd["id"] for nd in feats}
    out = _force_layout(feats, [e for e in edges if e["src"] in fids and e["dst"] in fids], fixed,
                        link_distance, meeting_distance, repel, max_iter, time_cap)
    if meets:
        out.update(_ring_meetings(meets, edges, out, fixed, meeting_distance))
    return out


def _ring_meetings(meets, edges, fpos, fixed, meeting_distance):
    if fpos:
        P = np.array(list(fpos.values()))
        c = P.mean(axis=0)
        r_out = float(np.max(np.hypot(*(P - c).T))) + meeting_distance * 0.6
    else:
        c, r_out = np.zeros(2), meeting_distance
    res, used = {}, []
    for m in sorted(meets, key=lambda x: x["id"]):
        if m["id"] in fixed:
            res[m["id"]] = tuple(fixed[m["id"]])
            continue
        touched = [fpos[e["dst"] if e["src"] == m["id"] else e["src"]] for e in edges
                   if m["id"] in (e["src"], e["dst"]) and (e["dst"] if e["src"] == m["id"] else e["src"]) in fpos]
        if touched:
            v = np.array(touched).mean(axis=0) - c
            ang = math.atan2(v[1], v[0]) if np.hypot(*v) > 1 else _h(m["id"], "m") * 2 * math.pi
        else:
            ang = _h(m["id"], "m") * 2 * math.pi
        # keep meetings apart along the ring (about 40 px of arc between them)
        step = 40.0 / r_out
        for _ in range(64):
            if all(abs((ang - a + math.pi) % (2 * math.pi) - math.pi) >= step for a in used):
                break
            ang += step
        used.append(ang)
        res[m["id"]] = (round(float(c[0] + r_out * math.cos(ang)), 2), round(float(c[1] + r_out * math.sin(ang)), 2))
    return res


def _force_layout(nodes: list[dict], edges: list[dict], fixed: dict[str, tuple[float, float]] | None = None,
                   link_distance: float = 90.0, meeting_distance: float = 160.0, repel: float = 1.0,
                   max_iter: int = 200, time_cap: float = 2.0) -> dict[str, tuple[float, float]]:
    """Positions for `nodes`. Nodes in `fixed` don't move; the others are placed around them."""
    t0 = time.perf_counter()
    fixed = fixed or {}
    ids = [n["id"] for n in nodes]
    n = len(ids)
    if n == 0:
        return {}
    idx = {nid: i for i, nid in enumerate(ids)}
    kind = [nd["type"] for nd in nodes]
    is_meet = np.array([k == "meeting" for k in kind])

    # links (unique pairs), degree from feature-feature links only
    pairs: dict[tuple[int, int], str] = {}
    for e in edges:
        a, b = idx.get(e["src"]), idx.get(e["dst"])
        if a is None or b is None or a == b:
            continue
        key = (min(a, b), max(a, b))
        if key not in pairs or not e["type"].startswith("meeting_"):
            pairs[key] = "meeting" if e["type"].startswith("meeting_") else "feature"
    deg = np.zeros(n)
    for (a, b), k in pairs.items():
        if k == "feature":
            deg[a] += 1
            deg[b] += 1
    rad = np.array([radius_for(int(deg[i]), kind[i]) for i in range(n)])
    lw = np.array([label_w(nd.get("label", "")) for nd in nodes])
    if pairs:
        E = np.array(list(pairs.keys()))
        meet_link = np.array([v == "meeting" for v in pairs.values()])
        L = np.where(meet_link, meeting_distance, link_distance)
        S = np.where(meet_link, 0.3, 1.0)
    else:
        E = np.zeros((0, 2), dtype=int)
        L = S = np.zeros(0)

    # start positions: fixed ones as given; new ones next to fixed neighbours, else hashed seeds
    spread = link_distance * max(1.0, math.sqrt(n)) * 0.6
    pos = np.zeros((n, 2))
    movable = np.ones(n, dtype=bool)
    for nid, p in fixed.items():
        if nid in idx:
            pos[idx[nid]] = p
            movable[idx[nid]] = False
    nbrs: dict[int, list[int]] = {}
    for a, b in E:
        nbrs.setdefault(int(a), []).append(int(b))
        nbrs.setdefault(int(b), []).append(int(a))
    for i in range(n):
        if movable[i]:
            anchors = [tuple(pos[j]) for j in nbrs.get(i, []) if not movable[j]]
            pos[i] = place_near(ids[i], anchors, link_distance, spread) if anchors else seed_xy(ids[i], spread)
    if not movable.any():
        return {nid: (float(pos[i][0]), float(pos[i][1])) for i, nid in enumerate(ids)}
    centre = pos[~movable].mean(axis=0) if (~movable).any() else np.zeros(2)

    k = link_distance
    cutoff = 4 * k
    temp = k * 0.6
    eye = np.eye(n, dtype=bool)
    links_per_node = np.bincount(E.ravel(), minlength=n) if len(E) else np.zeros(n)
    pull = np.where(is_meet, 0.012, 0.060) * np.where(links_per_node == 0, 3.0, 1.0)   # orphans pulled harder
    for it in range(max_iter):
        sq = (pos * pos).sum(1)
        d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (pos @ pos.T), 1e-6)
        # repulsion k^2/d along (p_i - p_j); ignored far away so separate clusters don't drift off.
        # sum_j f_ij (p_i - p_j) = p_i * sum_j f_ij - (F @ P)_i   (a matrix product instead of an N*N*2 array)
        f = np.where((d2 < cutoff * cutoff) & ~eye, REPEL_SCALE * repel * k * k / d2, 0.0)
        disp = pos * f.sum(1)[:, None] - f @ pos
        # springs (Fruchterman–Reingold attraction d^2/L), weaker and longer for meetings
        if len(E):
            d = pos[E[:, 1]] - pos[E[:, 0]]
            dl = np.sqrt((d ** 2).sum(-1)) + 1e-6
            att = (d * (S * dl / L)[:, None])
            np.add.at(disp, E[:, 0], att)
            np.add.at(disp, E[:, 1], -att)
        # gentle pull to the centre (orphans and small clusters stay close)
        disp -= (pos - centre) * pull[:, None]
        mag = np.sqrt((disp ** 2).sum(-1)) + 1e-9
        step = disp / mag[:, None] * np.minimum(mag, temp)[:, None]
        step[~movable] = 0
        pos += step
        if it % 3 == 2:
            _collide(pos, rad, lw, movable)
        temp = max(0.2, temp * 0.95)
        if np.abs(step[movable]).sum(-1).mean() < 0.5 or time.perf_counter() - t0 > time_cap:
            break
    for _ in range(30):                        # make sure nothing overlaps at the end
        if not _collide(pos, rad, lw, movable):
            break
    if fixed and (~movable).any():
        pass                                   # keep the frame of the existing nodes: no re-centring
    else:
        pos -= pos.mean(axis=0)
    return {nid: (round(float(pos[i][0]), 2), round(float(pos[i][1]), 2)) for i, nid in enumerate(ids)}


def _collide(pos: np.ndarray, rad: np.ndarray, lw: np.ndarray, movable: np.ndarray) -> bool:
    """Separate overlapping footprints. Each node's footprint is its circle plus its label underneath:
    a box [x - hw, x + hw] x [y - top, y + bottom] with hw = max(r, label/2). Overlapping pairs are pushed
    apart along the axis that needs the smaller move (minimum 8 px between circles). Returns True if moved."""
    n = len(pos)
    if n < 2:
        return False
    pad = 4.0
    hw = np.maximum(rad, lw / 2) + pad
    top = rad + pad
    bot = rad + LABEL_ROW + pad
    dx = pos[:, None, 0] - pos[None, :, 0]            # x_i - x_j
    dy = pos[:, None, 1] - pos[None, :, 1]
    ox = hw[:, None] + hw[None, :] - np.abs(dx)
    # vertical overlap depends on who is above: if i is above j, i's label meets j's top
    oy = np.where(dy < 0, bot[:, None] + top[None, :] + dy, bot[None, :] + top[:, None] - dy)
    hit = np.triu((ox > 0.01) & (oy > 0.01), 1)
    if not hit.any():
        return False
    i, j = np.nonzero(hit)
    along_x = ox[i, j] < oy[i, j]
    sx = np.where(dx[i, j] >= 0, 1.0, -1.0)
    sy = np.where(dy[i, j] >= 0, 1.0, -1.0)
    mx = np.where(along_x, ox[i, j] * sx, 0.0) / 2
    my = np.where(along_x, 0.0, oy[i, j] * sy) / 2
    mi, mj = movable[i].astype(float), movable[j].astype(float)
    # if one side is fixed, the movable one takes the whole move
    wi = np.where(mj == 0, 2.0, 1.0) * mi
    wj = np.where(mi == 0, 2.0, 1.0) * mj
    np.add.at(pos[:, 0], i, mx * wi)
    np.add.at(pos[:, 1], i, my * wi)
    np.add.at(pos[:, 0], j, -mx * wj)
    np.add.at(pos[:, 1], j, -my * wj)
    return True


# ------------------------------------------------------------------ topic groups (for colour)
def topic_groups(nodes: list[dict], edges: list[dict]) -> dict[str, int]:
    """Group features that are linked together (deterministic label propagation on feature-feature links).
    Returns feature id -> group number, biggest group = 0. Unlinked features get -1 (drawn neutral)."""
    feats = sorted(n["id"] for n in nodes if n["type"] == "feature")
    fset = set(feats)
    adj: dict[str, list[str]] = {f: [] for f in feats}
    for e in edges:
        a, b = e["src"], e["dst"]
        if a in fset and b in fset and a != b:
            adj[a].append(b)
            adj[b].append(a)
    label = {f: f for f in feats}
    order = sorted(feats, key=lambda f: (-len(adj[f]), f))
    for _ in range(20):
        changed = False
        for f in order:
            if not adj[f]:
                continue
            counts: dict[str, int] = {}
            for m in adj[f]:
                counts[label[m]] = counts.get(label[m], 0) + 1
            best = max(counts.values())
            tied = [lab for lab, c in counts.items() if c == best]
            if label[f] in tied:
                continue                                          # already in a top group: stay (stable)
            size = {lab: 0 for lab in tied}
            for x in feats:
                if label[x] in size:
                    size[label[x]] += 1
            new = min(tied, key=lambda lab: (-size[lab], lab))    # ties: join the bigger group
            if new != label[f]:
                label[f] = new
                changed = True
        if not changed:
            break
    sizes: dict[str, int] = {}
    for f in feats:
        if adj[f]:
            sizes[label[f]] = sizes.get(label[f], 0) + 1
    rank = {lab: i for i, lab in enumerate(sorted(sizes, key=lambda x: (-sizes[x], x)))}
    return {f: (rank[label[f]] if adj[f] else -1) for f in feats}


# ------------------------------------------------------------------ cache (one JSON per project)
def cache_path(workspace_root: Path, project_id: str) -> Path:
    return Path(workspace_root) / ".ui" / "map" / f"{project_id}.json"


def load_cache(path: Path) -> dict:
    """Saved positions/pins/settings, or {} if missing or unreadable (a bad file is simply ignored)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != 1:
            return {}
        pos = data.get("positions") or {}
        data["positions"] = {k: (float(v[0]), float(v[1])) for k, v in pos.items()
                             if isinstance(v, (list, tuple)) and len(v) == 2 and all(map(math.isfinite, v))}
        data["pinned"] = [p for p in data.get("pinned") or [] if isinstance(p, str)]
        data["settings"] = data.get("settings") if isinstance(data.get("settings"), dict) else {}
        return data
    except (OSError, ValueError, TypeError):
        return {}


def save_cache(path: Path, data: dict) -> None:
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        out = dict(data, version=1)
        out["positions"] = {k: [round(v[0], 2), round(v[1], 2)] for k, v in (data.get("positions") or {}).items()}
        tmp.write_text(json.dumps(out), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass                                    # a cache: losing it only means one more layout later
