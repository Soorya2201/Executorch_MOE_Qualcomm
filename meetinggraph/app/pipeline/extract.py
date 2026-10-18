"""Transcript -> draft (summary + feature mentions), via an LLM or the rule-based fallback."""
from __future__ import annotations

import json
import re

from ..models.llm import PLACEHOLDER_RE, _skeleton, BaseLLM, LLMError
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


def _wrap_bare_array(raw: str, schema: dict | None) -> str:
    """Small models often answer {"feature_mentions": [...]} with just the [...] list. When the schema has a
    single list field, put the list back under that field so the normal parser (and salvage) can read it."""
    props = (schema or {}).get("properties", {})
    lists = [k for k, v in props.items() if v.get("type") == "array"]
    text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    if len(props) == 1 and lists and text.startswith("["):
        return '{"%s": %s}' % (lists[0], text)
    return raw


def _ask_json(llm: BaseLLM, prompt: str, max_tokens: int, schema: dict | None = None,
              raws: list | None = None) -> dict:
    """Ask for JSON. Falls back to salvaging a cut-off reply, then one shorter retry."""
    raws = raws if raws is not None else []
    for attempt, p in enumerate((prompt, prompt + prompts.SHORTER)):
        raw = llm.generate(p, max_tokens=max_tokens, json_mode=True, schema=schema)
        raws.append(raw)
        raw = _wrap_bare_array(raw, schema)
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
    """A clean string; a JSON-shape placeholder copied back verbatim (e.g. "<one sentence>") counts as empty."""
    if not isinstance(x, str):
        return ""
    x = x.strip()
    return "" if PLACEHOLDER_RE.fullmatch(x) else x


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
        title = _s(f["title"])
        rels = normalize_relations(f.get("relations"), title)
        deps = [_s(x) for x in _as_list(f.get("depends_on")) if _s(x)]
        # "needs" relations ARE the dependencies; keep both views in step
        for r in rels:
            if r["type"] == "needs" and norm(r["to"]) not in {norm(x) for x in deps}:
                deps.append(r["to"])
        for x in deps:
            if not any(r["type"] == "needs" and norm(r["to"]) == norm(x) for r in rels):
                rels.append({"to": x, "type": "needs", "why": ""})
        out["feature_mentions"].append({
            "title": title,
            "action": action if action in ("add", "change", "archive") else "change",
            "description": _s(f.get("description")),
            "key_points": _dedupe([_s(x) for x in _as_list(f.get("key_points")) if _s(x)]),
            "relations": rels,
            "status": _s(f.get("status")).lower(),
            "subtasks": [_s(x) for x in _as_list(f.get("subtasks")) if _s(x)],
            "depends_on": [x for x in deps if norm(x) != norm(title)],
            "impact": _s(f.get("impact")),
            "uncertainty": _s(f.get("uncertainty")),
            "evidence": [{"start": _num(e.get("start")), "end": _num(e.get("end")), "quote": _s(e.get("quote"))}
                         if isinstance(e, dict) else {"start": 0.0, "end": 0.0, "quote": _s(e)}
                         for e in _as_list(f.get("evidence")) if isinstance(e, dict) or _s(e)],
        })
    return out


RELATION_TYPES = ("needs", "part of", "produces", "controls", "uses", "related")


def _dedupe(xs: list[str]) -> list[str]:
    out, seen = [], set()
    for x in xs:
        if norm(x) and norm(x) not in seen:
            seen.add(norm(x))
            out.append(x)
    return out


def normalize_relations(raw, owner_title: str = "") -> list[dict]:
    """[{to, type, why}] with known types, no self-links, one entry per (type, to)."""
    out, seen = [], set()
    for r in _as_list(raw):
        if not isinstance(r, dict) or not _s(r.get("to")):
            continue
        to = _s(r["to"])
        typ = _s(r.get("type")).lower().replace("_", " ")
        typ = {"depends on": "needs", "requires": "needs", "need": "needs", "part": "part of",
               "produce": "produces", "control": "controls", "use": "uses"}.get(typ, typ)
        if typ not in RELATION_TYPES:
            typ = "related"
        if norm(to) == norm(owner_title) or (typ, norm(to)) in seen:
            continue
        seen.add((typ, norm(to)))
        out.append({"to": to, "type": typ, "why": _s(r.get("why"))})
    return out


def merge_relations(old: list[dict], new: list[dict]) -> list[dict]:
    """Union by (type, to); a non-empty newer 'why' replaces an older one."""
    out = [dict(r) for r in old or []]
    for r in new or []:
        hit = next((o for o in out if o["type"] == r["type"] and norm(o["to"]) == norm(r["to"])), None)
        if hit is None:
            out.append(dict(r))
        elif r.get("why") and len(r["why"]) >= len(hit.get("why") or ""):
            hit["why"] = r["why"]
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
            g["key_points"] = _dedupe(g["key_points"] + f["key_points"])
            g["relations"] = merge_relations(g["relations"], f["relations"])
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


SMALL_CONTEXT = 4096         # tokens; at or below this the compact feature prompt is used
CHARS_PER_TOKEN = 3.5       # same conservative estimate as the NPU runner adapter
NOTES_OUT, FEATURES_OUT = 450, 750


def plan_requests(llm: BaseLLM, max_chars: int, existing: str) -> tuple[int, str, dict, int, int]:
    """Size the transcript chunk to the model's context so the prompt AND a complete answer fit.

    Returns (chunk_chars, features_prompt_template, features_schema, notes_budget, features_budget)."""
    ctx = getattr(llm, "max_seq_len", None)
    if not ctx:
        return max_chars, prompts.FEATURES, prompts.FEATURES_SCHEMA, 1200, 3500
    compact = ctx <= SMALL_CONTEXT
    template = prompts.FEATURES_COMPACT if compact else prompts.FEATURES
    schema = prompts.FEATURES_COMPACT_SCHEMA if compact else prompts.FEATURES_SCHEMA
    feat_out = FEATURES_OUT if compact else 1500
    fixed = len(template.format(features=existing, transcript="", max_items=5)) + \
        len(json.dumps(_skeleton(schema), separators=(",", ":"))) + 120     # + the adapter's JSON instruction
    fixed_notes = len(prompts.NOTES.format(transcript="")) + \
        len(json.dumps(_skeleton(prompts.NOTES_SCHEMA), separators=(",", ":"))) + 120
    usable = ctx - 64                                                     # chat template
    chars = min(int((usable - feat_out) * CHARS_PER_TOKEN) - fixed,
                int((usable - NOTES_OUT) * CHARS_PER_TOKEN) - fixed_notes)
    return max(800, min(max_chars, chars)), template, schema, NOTES_OUT, feat_out


def extract_part(llm: BaseLLM, segments: list[dict], existing_titles: list[str],
                 max_chars: int = 6000) -> list[dict]:
    """Notes + features for one stretch of transcript (one part per chunk). Used for whole meetings and
    for the live notes built while recording."""
    if not segments:
        return []
    if llm.heuristic:
        return [heuristic.extract(segments, "")]
    feats = "\n".join(f"- {t}" for t in existing_titles) or "- (none yet)"
    chunk_chars, feat_template, feat_schema, notes_out, feat_out = plan_requests(llm, max_chars, feats)
    parts = []
    try:
        for chunk in chunk_lines(to_text(segments), chunk_chars):
            # Two small requests instead of one big one: small models keep short replies intact.
            raws: list[str] = []
            try:
                notes = _ask_json(llm, prompts.NOTES.format(transcript=chunk), notes_out, prompts.NOTES_SCHEMA,
                                  raws)
            except ExtractionError:
                notes = {}
            found = _ask_json(llm, feat_template.format(features=feats, transcript=chunk, max_items=5), feat_out,
                              feat_schema, raws)
            parts.append(_normalize_draft({**notes, "feature_mentions": found.get("feature_mentions", [])}))
    except LLMError as e:
        raise ExtractionError(str(e)) from e
    return parts


def combine_parts(llm: BaseLLM, parts: list[dict], segments: list[dict], title: str) -> dict:
    """Merge parts into one draft, write one overall summary, and check every quote against the transcript."""
    if not parts:
        raise ExtractionError("No notes could be made from this transcript.")
    draft = _merge_drafts(parts)
    if llm.heuristic:
        names = ", ".join(f["title"] for f in draft["feature_mentions"]) or "no features"
        draft["summary"] = (f"Discussed {names}. {len(draft['decisions'])} decision(s), "
                            f"{len(draft['action_items'])} action item(s), {len(draft['open_questions'])} open "
                            "question(s). (Rule-based summary: set an LLM backend for richer summaries.)")
    elif len(parts) > 1:
        joined = "\n\n".join(f"Part {i + 1}: {p['summary']}" for i, p in enumerate(parts) if p["summary"])
        try:
            draft["summary"] = _s(_ask_json(llm, prompts.MERGE_SUMMARY.format(parts=joined), 400,
                                            prompts.SUMMARY_SCHEMA).get("summary")) \
                or " ".join(p["summary"] for p in parts)
        except (ExtractionError, LLMError):
            draft["summary"] = " ".join(p["summary"] for p in parts)
    else:
        draft["summary"] = parts[0]["summary"]
    draft["title"] = title
    check_evidence(draft, segments)
    return draft


def extract_draft(llm: BaseLLM, segments: list[dict], title: str, existing_titles: list[str],
                  max_chars: int = 6000) -> dict:
    if not segments:
        raise ExtractionError("The transcript is empty.")
    if llm.heuristic:
        draft = heuristic.extract(segments, title)
        check_evidence(draft, segments)
        return draft
    return combine_parts(llm, extract_part(llm, segments, existing_titles, max_chars), segments, title)
