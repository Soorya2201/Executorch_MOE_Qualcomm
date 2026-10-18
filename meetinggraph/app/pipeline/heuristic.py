"""Rule-based extractor: lets the full app run with no model installed.

It only understands explicit phrasing ("New feature: X.", "X depends on Y.", "Subtasks for X: a; b.",
"Decision: ...", "Action item: ...", "Open question: ...", "We are dropping X.", "Change to X: ...").
The fixture meetings in tests/fixtures are written that way. Real meetings need an LLM backend.
"""
from __future__ import annotations

import re

SENT = re.compile(r"(?<=[.!?])\s+")

R_NEW = re.compile(r"^(?:new feature|we(?:'ll| will) (?:build|add)|let'?s add)\s*:?\s*(.+)$", re.I)
R_DEP = re.compile(r"^(.+?)\s+(?:depends on|needs|requires|is blocked by)\s+(.+)$", re.I)
R_SUB = re.compile(r"^subtasks?\s+for\s+(.+?):\s*(.+)$", re.I)
R_ARCH = re.compile(r"^(?:we are |we're |let'?s |we will |we'll )?(?:drop|dropping|archive|archiving|remove|removing|cut|cutting)"
                    r"\s+(?:the\s+)?(.+?)(?:\s+feature)?(?:\s+for now|\s+from (?:the )?scope)?[.!]?$", re.I)
R_CHG = re.compile(r"^(?:change to|update to|update on|changes? for)\s+(.+?):\s*(.+)$", re.I)
R_IMP = re.compile(r"^impact(?:\s+of|\s+on)?\s+(.+?):\s*(.+)$", re.I)
R_STATUS = re.compile(r"^(.+?)\s+is\s+(?:now\s+)?(blocked|done|complete|finished|active|in progress)\b", re.I)
R_DEC = re.compile(r"^(?:decision:\s*|we decided (?:to |that )?)(.+)$", re.I)
R_ACT = re.compile(r"^action(?:\s+item)?:\s*(.+)$", re.I)
R_Q = re.compile(r"^open question:\s*(.+)$", re.I)
R_OWNER = re.compile(r"^([A-Z][a-z]+)\s+(?:to|will)\b")
R_IT = re.compile(r"^(?:it|this feature|this)\s+(.+)$", re.I)

STATUS_MAP = {"blocked": "blocked", "done": "done", "complete": "done", "finished": "done",
              "active": "active", "in progress": "active"}


def _clean(s: str) -> str:
    return s.strip().strip(" .;:,").strip()


def _title_and_desc(s: str) -> tuple[str, str]:
    s = _clean(s)
    for sep in (" — ", " - ", ": ", ", which ", ", that "):
        if sep in s:
            a, b = s.split(sep, 1)
            return _clean(a), _clean(b)
    return s, ""


def _split_list(s: str) -> list[str]:
    parts = re.split(r";|,\s*and\s+|\s+and\s+|,", s)
    return [_clean(p) for p in parts if _clean(p)]


def extract(segments: list[dict], title: str = "Meeting") -> dict:
    feats: dict[str, dict] = {}
    order: list[str] = []
    decisions, actions, questions = [], [], []
    last_feature = None

    def feat(name: str, action: str) -> dict:
        key = name.lower()
        if key not in feats:
            feats[key] = {"title": name, "action": action, "description": "", "status": "", "subtasks": [],
                          "depends_on": [], "impact": "", "uncertainty": "", "evidence": [],
                          "key_points": [], "relations": []}
            order.append(key)
        f = feats[key]
        rank = {"change": 0, "add": 1, "archive": 2}
        if rank[action] > rank[f["action"]]:
            f["action"] = action
        return f

    def ev(f: dict, seg: dict, quote: str):
        f["evidence"].append({"start": seg["start"], "end": seg["end"], "quote": quote})

    for seg in segments:
        for sent in SENT.split(seg["text"]):
            s = sent.strip()
            if not s:
                continue
            body = s.rstrip(".!")
            e = {"start": seg["start"], "end": seg["end"], "quote": s}
            m = R_NEW.match(body)
            if m:
                name, desc = _title_and_desc(m.group(1))
                f = feat(name, "add")
                if desc and not f["description"]:
                    f["description"] = desc
                ev(f, seg, s)
                last_feature = f
                continue
            m = R_SUB.match(body)
            if m:
                f = feat(_clean(m.group(1)), "change")
                f["subtasks"] += [x for x in _split_list(m.group(2)) if x not in f["subtasks"]]
                ev(f, seg, s)
                last_feature = f
                continue
            m = R_CHG.match(body)
            if m:
                f = feat(_clean(m.group(1)), "change")
                f["description"] = _clean(m.group(2))
                ev(f, seg, s)
                last_feature = f
                continue
            m = R_IMP.match(body)
            if m:
                f = feat(_clean(m.group(1)), "change")
                f["impact"] = _clean(m.group(2))
                ev(f, seg, s)
                continue
            m = R_DEC.match(body)
            if m:
                decisions.append({"text": _clean(m.group(1)), **e})
                continue
            m = R_ACT.match(body)
            if m:
                text = _clean(m.group(1))
                om = R_OWNER.match(text)
                actions.append({"text": text, "owner": om.group(1) if om else "", **e})
                continue
            m = R_Q.match(body)
            if m:
                questions.append({"text": _clean(m.group(1)), **e})
                continue
            m = R_ARCH.match(body)
            if m:
                f = feat(_clean(m.group(1)), "archive")
                ev(f, seg, s)
                continue
            m = R_DEP.match(body)
            if m:
                f = feat(_clean(m.group(1)), "change")
                for d in _split_list(m.group(2)):
                    if d not in f["depends_on"]:
                        f["depends_on"].append(d)
                        f["relations"].append({"to": d, "type": "needs", "why": _clean(s)})
                ev(f, seg, s)
                continue
            m = R_STATUS.match(body)
            if m:
                f = feat(_clean(m.group(1)), "change")
                f["status"] = STATUS_MAP[m.group(2).lower()]
                ev(f, seg, s)
                continue
            m = R_IT.match(body)
            if m and last_feature is not None and not last_feature["description"]:
                last_feature["description"] = _clean(s)
                ev(last_feature, seg, s)
                continue
            if body.endswith("?") or s.endswith("?"):
                questions.append({"text": _clean(s), **e})

    mentions = [feats[k] for k in order]
    names = ", ".join(f["title"] for f in mentions) or "no features"
    summary = (f"Discussed {names}. {len(decisions)} decision(s), {len(actions)} action item(s), "
               f"{len(questions)} open question(s). (Rule-based summary: set an LLM backend for richer summaries.)")
    return {"title": title, "summary": summary, "decisions": decisions, "action_items": actions,
            "open_questions": questions, "feature_mentions": mentions}
