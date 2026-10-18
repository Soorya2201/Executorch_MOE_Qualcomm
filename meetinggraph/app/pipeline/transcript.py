"""Transcript segments <-> the editable text format shown to the user.

Line format:  [00:01:05-00:01:12] (remote) text      low confidence lines start with "?"
Lines without a timestamp (e.g. a pasted transcript) get synthetic 5-second slots.
"""
from __future__ import annotations

import difflib
import re

LINE = re.compile(r"^\s*(\?)?\s*\[(\d+):(\d{2}):(\d{2})\s*-\s*(\d+):(\d{2}):(\d{2})\]\s*(?:\((\w+)\))?\s*(.*)$")
LOW_CONF = 0.45


def fmt_ts(sec: float) -> str:
    sec = max(0, int(round(sec)))
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def parse_ts(s: str) -> float:
    parts = [int(p) for p in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def to_text(segments: list[dict]) -> str:
    out = []
    for s in segments:
        q = "? " if s.get("confidence", 1.0) < LOW_CONF else ""
        sp = f" ({s['speaker']})" if s.get("speaker") else ""
        out.append(f"{q}[{fmt_ts(s['start'])}-{fmt_ts(s['end'])}]{sp} {s['text']}")
    return "\n".join(out) + ("\n" if out else "")


def from_text(text: str) -> list[dict]:
    segs, t = [], 0.0
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = LINE.match(line)
        if m:
            q, h1, m1, s1, h2, m2, s2, speaker, body = m.groups()
            start = int(h1) * 3600 + int(m1) * 60 + int(s1)
            end = int(h2) * 3600 + int(m2) * 60 + int(s2)
            if body.strip():
                segs.append({"start": float(start), "end": float(max(end, start)), "speaker": speaker or "",
                             "text": body.strip(), "confidence": 0.3 if q else 1.0})
            t = max(t, end)
        else:
            segs.append({"start": t, "end": t + 5, "speaker": "", "text": line, "confidence": 1.0})
            t += 5
    return segs


def merge_tracks(mic: list[dict], remote: list[dict]) -> list[dict]:
    """Interleave two aligned tracks; drop mic segments that echo a remote segment."""
    kept = []
    for s in mic:
        echo = False
        for r in remote:
            overlap = min(s["end"], r["end"]) - max(s["start"], r["start"])
            if overlap > -1.0 and difflib.SequenceMatcher(None, s["text"].lower(), r["text"].lower()).ratio() > 0.8:
                echo = True
                break
        if not echo:
            kept.append(dict(s, speaker="you"))
    merged = kept + [dict(r, speaker="remote") for r in remote]
    return sorted(merged, key=lambda x: (x["start"], x["end"]))


def chunk_lines(text: str, max_chars: int) -> list[str]:
    chunks, cur = [], []
    size = 0
    for line in text.splitlines(keepends=True):
        if size + len(line) > max_chars and cur:
            chunks.append("".join(cur))
            # one line of overlap keeps context across the boundary
            cur, size = [cur[-1]], len(cur[-1])
        cur.append(line)
        size += len(line)
    if cur:
        chunks.append("".join(cur))
    return chunks


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def quote_supported(quote: str, segments: list[dict]) -> dict | None:
    """Return the segment that supports the quote (normalized containment or >=0.75 token overlap)."""
    q = norm(quote)
    if not q:
        return None
    qt = set(q.split())
    best, best_score = None, 0.0
    for s in segments:
        st = norm(s["text"])
        if q in st or (st and st in q and len(st) > 20):
            return s
        toks = set(st.split())
        if qt and toks:
            score = len(qt & toks) / len(qt)
            if score > best_score:
                best, best_score = s, score
    return best if best_score >= 0.75 else None
