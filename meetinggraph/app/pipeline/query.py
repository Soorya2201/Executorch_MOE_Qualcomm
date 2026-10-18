"""Project-scoped, cited Q&A over approved files only.

Relationship questions ("why does A need B?", "how is A related to B?") get special help: the features named
in the question are found, the connection between them is looked up in the saved files (directly, or as a
short chain through other features), and the model is given those reasons, the features' key points and the
transcript lines saved with them.
"""
from __future__ import annotations

import re
from collections import deque

from ..models.llm import BaseLLM, LLMError
from ..workspace.index import ProjectIndex, tokens
from ..workspace.store import RELATION_LABEL, Store
from . import prompts

IDK = "I don't know from this project."

MIN_SCORE = 0.8
MAX_DOCS = 6
WIKILINK = re.compile(r"\[\[[^|\]]+\|([^\]]+)\]\]")


def norm_eq_key(a: str) -> str:
    return re.sub(r"\W+", "", a.lower())


def norm_eq(a: str, b: str) -> bool:
    return re.sub(r"\W+", "", a.lower()) == re.sub(r"\W+", "", b.lower())


def _stem(t: str) -> str:
    """Crude singular: ribosomes -> ribosome, bodies -> body, glass stays glass."""
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    return t


def _words(text: str) -> list[str]:
    return [_stem(t) for t in tokens(text)]


def mentioned_features(question: str, feats: list[dict]) -> list[dict]:
    """Features the question names: all words of the title, or a word that only that title has."""
    q = set(_words(question))
    words = {f["id"]: set(_words(f["title"])) for f in feats}
    count: dict[str, int] = {}
    for ws in words.values():
        for w in ws:
            count[w] = count.get(w, 0) + 1
    scored = []
    for f in feats:
        ws = words[f["id"]]
        if not ws:
            continue
        hit = ws & q
        full = hit == ws
        unique = any(count[w] == 1 and len(w) > 3 for w in hit)
        if full or unique:
            scored.append((2 if full else 1, len(hit), f))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    out = [f for *_, f in scored]
    # a full-title match wins over a feature that only shares one word with it
    full_ids = {f["id"] for s, _, f in scored if s == 2}
    if full_ids:
        out = [f for f in out if f["id"] in full_ids or not (words[f["id"]] <= set().union(
            *(words[i] for i in full_ids)))]
    return out[:4]


def _links(feats: list[dict]) -> list[dict]:
    """All feature->feature connections with their reasons: {src, dst, type, why}."""
    ids = {f["id"] for f in feats}
    out = []
    for f in feats:
        rels = f.get("relations") or []
        for d in f.get("depends_on") or []:
            if d in ids:
                why = next((r.get("why") for r in rels if r.get("type") == "needs" and r.get("to") == d), "")
                out.append({"src": f["id"], "dst": d, "type": "needs", "why": why or ""})
        for r in rels:
            if r.get("type") != "needs" and r.get("to") in ids and r.get("to") != f["id"]:
                out.append({"src": f["id"], "dst": r["to"], "type": r.get("type", "related"),
                            "why": r.get("why") or ""})
    return out


def _path(a: str, b: str, links: list[dict], max_hops: int = 4) -> list[dict] | None:
    """Shortest chain of connections between two features (either direction)."""
    adj: dict[str, list[tuple[str, dict]]] = {}
    for l in links:
        adj.setdefault(l["src"], []).append((l["dst"], l))
        adj.setdefault(l["dst"], []).append((l["src"], l))
    prev: dict[str, tuple[str, dict] | None] = {a: None}
    dq = deque([(a, 0)])
    while dq:
        u, d = dq.popleft()
        if u == b:
            chain = []
            while prev[u] is not None:
                p, l = prev[u]
                chain.append(l)
                u = p
            return list(reversed(chain))
        if d >= max_hops:
            continue
        for v, l in adj.get(u, []):
            if v not in prev:
                prev[v] = (u, l)
                dq.append((v, d + 1))
    return None


def _link_text(l: dict, titles: dict[str, str]) -> str:
    verb = RELATION_LABEL.get(l["type"], "is related to")
    why = l.get("why") or "(no reason was given in the meeting)"
    return f"{titles.get(l['src'], '?')} {verb} {titles.get(l['dst'], '?')}. Why: {why}"


def _feature_text(f: dict, titles: dict[str, str], limit: int = 2200) -> str:
    """Compact, model-friendly version of a feature file."""
    out = [f"What it is: {f.get('description') or 'unknown'}"]
    if f.get("status"):
        out.append(f"Status: {f['status']}")
    for k in f.get("key_points") or []:
        out.append(f"Key point: {k}")
    for l in _links([f] + [{"id": i, "title": t} for i, t in titles.items() if i != f["id"]]):
        if l["src"] == f["id"]:
            out.append("Connection: " + _link_text(l, titles))
    if f.get("impact") and f["impact"] != "unknown":
        out.append(f"Impact: {f['impact']}")
    for e in f.get("excerpts") or []:
        out.append(f"What was said [{e.get('at', '')}]: {e.get('text', '')}")
    text, size = [], 0
    for ln in out:
        if size + len(ln) > limit:
            break
        text.append(ln)
        size += len(ln) + 1
    return "\n".join(text)


def _excerpt(text: str, q: list[str], limit: int = 1500) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("---")]
    hits = [ln for ln in lines if any(t in ln.lower() for t in q)]
    chosen = (hits + [ln for ln in lines if ln not in hits])
    out, size = [], 0
    for ln in chosen:
        if size + len(ln) > limit:
            break
        out.append(WIKILINK.sub(r"\1", ln))
        size += len(ln)
    return "\n".join(out)


def answer(store: Store, llm: BaseLLM, project_id: str, question: str, k: int = 5) -> dict:
    """Returns {"answer": str, "citations": [{"n", "doc_id", "kind", "title", "path"}]}."""
    feats = store.features(project_id)
    by_id = {f["id"]: f for f in feats}
    titles = {f["id"]: f["title"] for f in feats}
    named = mentioned_features(question, feats)
    links = _links(feats)

    # connections between the features the question names (direct, or through other features)
    chains: list[list[dict]] = []
    for i in range(len(named)):
        for j in range(i + 1, len(named)):
            p = _path(named[i]["id"], named[j]["id"], links)
            if p:
                chains.append(p)
    if len(named) == 1:      # "why does X ...": its own connections are the useful context
        chains += [[l] for l in links if named[0]["id"] in (l["src"], l["dst"])]

    # documents to show: named features, features on the chains, then search hits
    docs: list[dict] = []

    def add_feature(fid):
        if fid in by_id and not any(d["doc_id"] == fid for d in docs):
            f = by_id[fid]
            docs.append({"doc_id": fid, "kind": "feature", "title": f["title"], "path": f["_path"],
                         "body": _feature_text(f, titles), "raw": f.get("_body", "")})

    for f in named:
        add_feature(f["id"])
    for ch in chains:
        for l in ch:
            add_feature(l["src"])
            add_feature(l["dst"])
    q = tokens(question)
    hits = [h for h in ProjectIndex(store, project_id).search(question, k=k) if h["score"] >= MIN_SCORE]
    for h in hits:
        if len(docs) >= MAX_DOCS:
            break
        if h["kind"] == "feature":
            add_feature(h["doc_id"])
        elif not any(d["doc_id"] == h["doc_id"] for d in docs):
            docs.append({"doc_id": h["doc_id"], "kind": h["kind"], "title": h["title"], "path": h["path"],
                         "body": _excerpt(h["text"], q), "raw": h["text"]})
    docs = docs[:MAX_DOCS]
    if not docs:
        return {"answer": IDK, "citations": []}
    cites = [{"n": i + 1, "doc_id": d["doc_id"], "kind": d["kind"], "title": d["title"], "path": d["path"]}
             for i, d in enumerate(docs)]
    num = {c["doc_id"]: c["n"] for c in cites}

    # each connection is cited to the feature file it is written in (its source)
    seen, conn_lines = set(), []
    for ch in chains:
        for l in ch:
            key = (l["src"], l["dst"], l["type"])
            if key in seen:
                continue
            seen.add(key)
            conn_lines.append(f"- {_link_text(l, titles)} [{num.get(l['src'], num.get(l['dst'], 1))}]")

    if llm.heuristic:
        return _extractive(question, q, docs, cites, conn_lines)

    ex = ""
    if conn_lines:
        ex += "Connections between the things asked about (from the saved feature files):\n" + \
              "\n".join(conn_lines) + "\n\n"
    ex += "\n\n".join(f"[{c['n']}] {d['kind']}: {d['title']}\n{d['body']}" for c, d in zip(cites, docs))
    try:
        text = llm.generate(prompts.ANSWER.format(excerpts=ex, question=question), max_tokens=450).strip()
    except LLMError as e:
        return {"answer": f"Local model error: {e}", "citations": []}
    if not text or text.lower().startswith(IDK.lower().rstrip(".")):
        return {"answer": IDK, "citations": []}
    used = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    cited = [c for c in cites if c["n"] in used]
    if not cited:
        text += " (Model gave no citation; sources it was shown are listed.)"
        cited = cites
    return {"answer": text, "citations": cited}


def _extractive(question: str, q: list[str], docs: list[dict], cites: list[dict], conn_lines: list[str]) -> dict:
    """No language model: return the saved connection reasons and the most relevant saved lines, cited."""
    parts, used, seen = [], set(), set()
    for ln in conn_lines:
        parts.append(ln[2:])
        used |= {int(n) for n in re.findall(r"\[(\d+)\]", ln)}
    for c, d in zip(cites, docs):
        best = []
        for ln in (d["raw"] or d["body"]).splitlines():
            t = ln.strip().lstrip("-* ").strip()
            if not t or t.startswith(("#", "---", "r1 ", "r2 ", "r3 ", "r4 ", "r5 ", "ADD ", "CHANGE ", "ARCHIVE ")) \
                    or re.match(r"r\d+ ·", t) or norm_eq(t, d["title"]) or t.startswith("[[") and t.endswith("]]") \
                    or re.match(r"^[a-z_]+:\s", t):
                continue
            t = WIKILINK.sub(r"\1", t)
            key = norm_eq_key(t)
            if key in seen:
                continue
            score = sum(1 for tok in q if tok in t.lower())
            if score:
                seen.add(key)
                best.append((score, t))
        best.sort(key=lambda x: -x[0])
        for _, t in best[:2]:
            parts.append(f"{d['title']}: {t} [{c['n']}]")
            used.add(c["n"])
        if len(parts) >= 5:
            break
    if not parts:
        return {"answer": IDK, "citations": []}
    return {"answer": "\n".join(parts), "citations": [c for c in cites if c["n"] in used]}
