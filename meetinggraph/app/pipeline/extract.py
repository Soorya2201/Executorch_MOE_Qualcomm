"""Transcript -> draft (summary + feature mentions), via an LLM or the rule-based fallback."""
from __future__ import annotations

import json
import re

from ..models.llm import BaseLLM, LLMError
from . import heuristic, prompts
from .transcript import chunk_lines, norm, quote_supported, to_text


class ExtractionError(RuntimeError):
    def __init__(self, msg: str, raw: str = ""):
        super().__init__(msg)
        self.raw = raw


def _first_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                obj = json.loads(text[start:i + 1])
                if not isinstance(obj, dict):
                    raise ValueError("top-level JSON is not an object")
                return obj
    raise ValueError("unterminated JSON object")


def salvage_json(text: str) -> dict | None:
    """Recover a JSON object whose end was cut off: keep every complete element, close the brackets."""
    start = text.find("{")
    if start < 0:
        return None
    stack: list[str] = []
    safe: list[tuple[int, list[str]]] = []      # (cut index, open brackets at that point)
    in_str = esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack:
                break
            stack.pop()
            safe.append((i + 1, list(stack)))
            if not stack:
                break
        elif ch == "," and stack:
            safe.append((i, list(stack)))           # cut just before this comma
    for cut, open_ in reversed(safe[-200:]):
        candidate = text[start:cut].rstrip().rstrip(",") + "".join(reversed(open_))
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _ask_json(llm: BaseLLM, prompt: str, max_tokens: int, schema: dict | None = None,
              raws: list | None = None) -> dict:
    """Ask for JSON. Falls back to salvaging a cut-off reply, then one shorter retry."""
    raws = raws if raws is not None else []
    for attempt, p in enumerate((prompt, prompt + prompts.SHORTER)):
        raw = llm.generate(p, max_tokens=max_tokens, json_mode=True, schema=schema)
        raws.append(raw)
        try:
            return _first_json(raw)
        except (ValueError, json.JSONDecodeError) as e:
            err = e
            obj = salvage_json(raw)
            if obj is not None:
                obj["_salvaged"] = True
                return obj
    raise ExtractionError(f"The AI's reply was cut off or not valid JSON, even after a shorter retry ({err}).",
                          "\n\n----- next attempt -----\n\n".join(raws))


def _as_list(x) -> list:
    return x if isinstance(x, list) else []


def _s(x) -> str:
    return x.strip() if isinstance(x, str) else ""


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _normalize_draft(d: dict) -> dict:
    """Coerce model output into the strict draft schema; drop malformed items."""
    def item(x, owner=False):
        if not isinstance(x, dict) or not _s(x.get("text")):
            return None
        o = {"text": _s(x["text"]), "start": _num(x.get("start")), "end": _num(x.get("end")),
             "quote": _s(x.get("quote"))}
        if owner:
            o["owner"] = _s(x.get("owner"))
        return o

    out = {"summary": _s(d.get("summary")),
           "decisions": [i for i in (item(x) for x in _as_list(d.get("decisions"))) if i],
           "action_items": [i for i in (item(x, True) for x in _as_list(d.get("action_items"))) if i],
           "open_questions": [i for i in (item(x) for x in _as_list(d.get("open_questions"))) if i],
           "feature_mentions": []}
    for f in _as_list(d.get("feature_mentions")):
        if not isinstance(f, dict) or not _s(f.get("title")):
            continue
        action = _s(f.get("action")).lower()
        out["feature_mentions"].append({
            "title": _s(f["title"]),
            "action": action if action in ("add", "change", "archive") else "change",
            "description": _s(f.get("description")),
            "status": _s(f.get("status")).lower(),
            "subtasks": [_s(x) for x in _as_list(f.get("subtasks")) if _s(x)],
            "depends_on": [_s(x) for x in _as_list(f.get("depends_on")) if _s(x)],
            "impact": _s(f.get("impact")),
            "uncertainty": _s(f.get("uncertainty")),
            "evidence": [{"start": _num(e.get("start")), "end": _num(e.get("end")), "quote": _s(e.get("quote"))}
                         for e in _as_list(f.get("evidence")) if isinstance(e, dict)],
        })
    return out


def _merge_drafts(parts: list[dict]) -> dict:
    out = {"summary": "", "decisions": [], "action_items": [], "open_questions": [], "feature_mentions": []}
    by_title: dict[str, dict] = {}
    for p in parts:
        for k in ("decisions", "action_items", "open_questions"):
            seen = {norm(x["text"]) for x in out[k]}
            out[k] += [x for x in p[k] if norm(x["text"]) not in seen]
        for f in p["feature_mentions"]:
            key = norm(f["title"])
            if key not in by_title:
                by_title[key] = f
                out["feature_mentions"].append(f)
                continue
            g = by_title[key]
            for fld in ("description", "impact", "uncertainty", "status"):
                if len(f[fld]) > len(g[fld]):
                    g[fld] = f[fld]
            g["subtasks"] += [s for s in f["subtasks"] if s not in g["subtasks"]]
            g["depends_on"] += [s for s in f["depends_on"] if s not in g["depends_on"]]
            g["evidence"] += f["evidence"]
            if f["action"] == "archive":
                g["action"] = "archive"
            elif f["action"] == "add" and g["action"] == "change":
                g["action"] = "add"
    return out


def check_evidence(draft: dict, segments: list[dict]) -> None:
    """Mark each evidence quote supported/unsupported against the transcript (in place)."""
    for k in ("decisions", "action_items", "open_questions"):
        for x in draft[k]:
            seg = quote_supported(x.get("quote") or x["text"], segments)
            x["supported"] = seg is not None
            if seg:
                x["start"], x["end"] = seg["start"], seg["end"]
    for f in draft["feature_mentions"]:
        for e in f["evidence"]:
            seg = quote_supported(e["quote"], segments)
            e["supported"] = seg is not None
            if seg:
                e["start"], e["end"] = seg["start"], seg["end"]


def extract_draft(llm: BaseLLM, segments: list[dict], title: str, existing_titles: list[str],
                  max_chars: int = 6000) -> dict:
    if not segments:
        raise ExtractionError("The transcript is empty.")
    if llm.heuristic:
        draft = heuristic.extract(segments, title)
        check_evidence(draft, segments)
        return draft
    text = to_text(segments)
    feats = "\n".join(f"- {t}" for t in existing_titles) or "- (none yet)"
    parts = []
    try:
        for chunk in chunk_lines(text, max_chars):
            # Two small requests instead of one big one: small models keep short replies intact.
            raws: list[str] = []
            try:
                notes = _ask_json(llm, prompts.NOTES.format(transcript=chunk), 1200, prompts.NOTES_SCHEMA, raws)
            except ExtractionError:
                notes = {}
            found = _ask_json(llm, prompts.FEATURES.format(features=feats, transcript=chunk), 2500,
                              prompts.FEATURES_SCHEMA, raws)
            parts.append(_normalize_draft({**notes, "feature_mentions": found.get("feature_mentions", [])}))
        draft = _merge_drafts(parts)
        if len(parts) > 1:
            joined = "\n\n".join(f"Part {i + 1}: {p['summary']}" for i, p in enumerate(parts))
            try:
                draft["summary"] = _s(_ask_json(llm, prompts.MERGE_SUMMARY.format(parts=joined), 400,
                                                prompts.SUMMARY_SCHEMA).get("summary")) \
                    or " ".join(p["summary"] for p in parts)
            except ExtractionError:
                draft["summary"] = " ".join(p["summary"] for p in parts)
        else:
            draft["summary"] = parts[0]["summary"]
    except LLMError as e:
        raise ExtractionError(str(e)) from e
    draft["title"] = title
    check_evidence(draft, segments)
    return draft
