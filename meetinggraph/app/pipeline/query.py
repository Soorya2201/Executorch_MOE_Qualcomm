"""Project-scoped, cited Q&A over approved files only."""
from __future__ import annotations

import re

from ..models.llm import BaseLLM, LLMError
from ..workspace.index import ProjectIndex, tokens
from ..workspace.store import Store
from . import prompts

IDK = "I don't know from this project."


MIN_SCORE = 0.8
WIKILINK = re.compile(r"\[\[[^|\]]+\|([^\]]+)\]\]")


def norm_eq_key(a: str) -> str:
    return re.sub(r"\W+", "", a.lower())


def norm_eq(a: str, b: str) -> bool:
    return re.sub(r"\W+", "", a.lower()) == re.sub(r"\W+", "", b.lower())


def _excerpt(text: str, q: list[str], limit: int = 1200) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("---")]
    hits = [ln for ln in lines if any(t in ln.lower() for t in q)]
    chosen = (hits + [ln for ln in lines if ln not in hits])
    out, size = [], 0
    for ln in chosen:
        if size + len(ln) > limit:
            break
        out.append(ln)
        size += len(ln)
    return "\n".join(out)


def answer(store: Store, llm: BaseLLM, project_id: str, question: str, k: int = 5) -> dict:
    """Returns {"answer": str, "citations": [{"n", "doc_id", "kind", "title", "path"}]}."""
    idx = ProjectIndex(store, project_id)
    hits = idx.search(question, k=k)
    hits = [h for h in hits if h["score"] >= MIN_SCORE]
    if not hits:
        return {"answer": IDK, "citations": []}
    q = tokens(question)
    cites = [{"n": i + 1, "doc_id": h["doc_id"], "kind": h["kind"], "title": h["title"], "path": h["path"]}
             for i, h in enumerate(hits)]

    if llm.heuristic:
        # Extractive answer: the most relevant content lines from the top documents, each cited.
        parts, used, seen = [], set(), set()
        for c, h in zip(cites, hits):
            best = []
            for ln in h["text"].splitlines():
                t = ln.strip().lstrip("-* ").strip()
                if not t or t.startswith(("#", "---", "r1 ", "r2 ", "r3 ", "r4 ", "r5 ", "ADD ", "CHANGE ", "ARCHIVE ")) \
                        or re.match(r"r\d+ ·", t) or norm_eq(t, h["title"]) or t.startswith("[[") and t.endswith("]]"):
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
                parts.append(f"{c['title']}: {t} [{c['n']}]")
                used.add(c["n"])
            if len(parts) >= 4:
                break
        if not parts:
            return {"answer": IDK, "citations": []}
        return {"answer": "\n".join(parts), "citations": [c for c in cites if c["n"] in used]}

    ex = "\n\n".join(f"[{c['n']}] {c['kind']}: {c['title']}\n{_excerpt(h['text'], q)}" for c, h in zip(cites, hits))
    try:
        text = llm.generate(prompts.ANSWER.format(excerpts=ex, question=question), max_tokens=300).strip()
    except LLMError as e:
        return {"answer": f"Local model error: {e}", "citations": []}
    if not text or IDK.lower().rstrip(".") in text.lower():
        return {"answer": IDK, "citations": []}
    used = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    cited = [c for c in cites if c["n"] in used]
    if not cited:
        text += " (Model gave no citation; sources it was shown are listed.)"
        cited = cites
    return {"answer": text, "citations": cited}
