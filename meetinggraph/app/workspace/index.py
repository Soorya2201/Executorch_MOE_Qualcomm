"""Disposable per-project SQLite index (search text + graph edges), rebuilt from approved files.

It never stores transcripts: only approved feature and meeting files are indexed.
"""
from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from .graph import build_graph
from .store import Store

TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("a an the and or of to in on for is are was were be been it this that with as at by from we our "
           "what which who how when why do does did can could should would will".split())


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall((text or "").lower()) if t not in STOP]


class ProjectIndex:
    def __init__(self, store: Store, project_id: str):
        self.store = store
        self.project_id = project_id
        self.path: Path = store.layout.project_file(project_id, ".index.sqlite")

    def rebuild(self) -> None:
        docs = self._collect_docs()
        g = build_graph(self.store, self.project_id)
        tmp = self.path.with_suffix(".tmp")
        tmp.unlink(missing_ok=True)
        con = sqlite3.connect(tmp)
        try:
            con.execute("CREATE TABLE docs (doc_id TEXT PRIMARY KEY, kind TEXT, title TEXT, path TEXT, text TEXT)")
            con.execute("CREATE TABLE edges (src TEXT, dst TEXT, type TEXT, reason TEXT)")
            con.executemany("INSERT INTO docs VALUES (?,?,?,?,?)",
                            [(d["doc_id"], d["kind"], d["title"], d["path"], d["text"]) for d in docs])
            con.executemany("INSERT INTO edges VALUES (?,?,?,?)",
                            [(e["src"], e["dst"], e["type"], e["reason"]) for e in g["edges"]])
            con.commit()
        finally:
            con.close()
        tmp.replace(self.path)

    def _collect_docs(self) -> list[dict[str, Any]]:
        docs = []
        for f in self.store.features(self.project_id):
            subs = " ".join(s.get("text", "") for s in f.get("subtasks") or [])
            text = f"{f['title']}\n{f.get('description', '')}\n{subs}\n{f.get('impact', '')}\n{f.get('_body', '')}"
            docs.append({"doc_id": f["id"], "kind": "feature", "title": f["title"], "path": f["_path"], "text": text})
        for m in self.store.meetings(self.project_id):
            docs.append({"doc_id": m["id"], "kind": "meeting", "title": m.get("title", ""),
                         "path": m["_path"], "text": f"{m.get('title', '')}\n{m.get('_body', '')}"})
        return docs

    def search(self, query: str, k: int = 6) -> list[dict[str, Any]]:
        """BM25 over the indexed docs (small corpus, so scored in Python)."""
        if not self.path.exists():
            self.rebuild()
        con = sqlite3.connect(self.path)
        try:
            rows = con.execute("SELECT doc_id, kind, title, path, text FROM docs").fetchall()
        finally:
            con.close()
        q = tokens(query)
        if not q or not rows:
            return []
        docs = [{"doc_id": r[0], "kind": r[1], "title": r[2], "path": r[3], "text": r[4],
                 "_tf": Counter(tokens(r[4]) + tokens(r[2]) * 3)} for r in rows]
        n = len(docs)
        avgdl = sum(sum(d["_tf"].values()) for d in docs) / n or 1
        df = Counter(t for d in docs for t in set(d["_tf"]))
        k1, b = 1.5, 0.75
        for d in docs:
            dl = sum(d["_tf"].values()) or 1
            s = 0.0
            for t in q:
                if t not in d["_tf"]:
                    continue
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                tf = d["_tf"][t]
                s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
            d["score"] = s
        hits = [d for d in docs if d["score"] > 0]
        hits.sort(key=lambda d: d["score"], reverse=True)
        for d in hits:
            d.pop("_tf", None)
        return hits[:k]

    def edges(self) -> list[tuple[str, str, str, str]]:
        if not self.path.exists():
            self.rebuild()
        con = sqlite3.connect(self.path)
        try:
            return con.execute("SELECT src, dst, type, reason FROM edges").fetchall()
        finally:
            con.close()
