"""Build a plan (one entry per file), optionally review it, apply it with an undo log, undo it."""
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .config import Config
from .digest import FileDigest, iter_files, make_digest
from .route import Decision, SimilarityIndex, rule_tier, similarity_tier


@dataclass
class PlanItem:
    source: str
    folder: Optional[str]
    target: Optional[str]
    confidence: float
    tier: str
    reason: str
    status: str  # "move" | "review" | "duplicate" | "skip"


def _free_target(folder: Path, name: str) -> Path:
    target = folder / name
    stem, suffix, i = target.stem, target.suffix, 1
    while target.exists():
        target = folder / f"{stem} ({i}){suffix}"
        i += 1
    return target


def _existing_hashes(cfg: Config) -> Dict[str, str]:
    """sha256 -> path of files already in destination folders (to catch duplicates)."""
    from .digest import sha256

    hashes = {}
    for dest in cfg.destinations.values():
        if dest.path.exists():
            for p in iter_files(dest.path, recursive=True):
                try:
                    hashes.setdefault(sha256(p), str(p))
                except OSError:
                    pass
    return hashes


def decide(d: FileDigest, cfg: Config, index: SimilarityIndex, llm) -> Decision:
    """Rules -> similarity -> Phi-4-mini. Returns the best decision found."""
    for tier in (lambda: rule_tier(d, cfg.destinations),
                 lambda: similarity_tier(d, index, cfg.similarity_threshold)):
        dec = tier()
        if dec and dec.confidence >= cfg.confidence_threshold:
            return dec
    llm_dec = None
    if llm is not None:
        try:
            llm_dec = llm.classify(d, cfg.destinations)
        except Exception as e:  # runner failure: fall through to the best non-LLM guess
            print(f"  ! LLM failed for {d.name}: {e}")
    if llm_dec:
        # LLMs state confidence they don't have. Trust it only when the independent
        # content-similarity signal agrees; otherwise cap it below the review threshold.
        ranked = index.rank(d)
        top = ranked[0][0] if ranked and ranked[0][1] > 0 else None
        if top is None or top == llm_dec.folder:
            llm_dec.confidence = min(llm_dec.confidence, 0.9)
        else:
            llm_dec.confidence = min(llm_dec.confidence, cfg.confidence_threshold - 0.05)
            llm_dec.reason += f" (similarity suggested {top})"
        return llm_dec
    # No confident answer: best similarity guess (low confidence) so review-off can still sort.
    ranked = index.rank(d)
    if ranked and ranked[0][1] > 0:
        return Decision(ranked[0][0], min(0.4, ranked[0][1]), "fallback", "closest folder by content")
    return Decision(None, 0.0, "none", "no signal")


def build_plan(cfg: Config, llm) -> List[PlanItem]:
    index = SimilarityIndex(cfg)
    known = _existing_hashes(cfg)
    dest_roots = [d.path.resolve() for d in cfg.destinations.values()]
    plan: List[PlanItem] = []
    for src in cfg.sources:
        for p in iter_files(src, cfg.recursive):
            rp = p.resolve()
            if any(rp.is_relative_to(r) for r in dest_roots):
                continue  # already organised
            d = make_digest(p, cfg.page_chars)
            if d.sha256 in known:
                plan.append(PlanItem(str(p), None, None, 1.0, "hash", f"identical to {known[d.sha256]}", "duplicate"))
                continue
            dec = decide(d, cfg, index, llm)
            sure = dec.folder is not None and dec.confidence >= cfg.confidence_threshold
            if dec.folder is None:
                status = "review" if cfg.review else "skip"
            elif sure or not cfg.review:
                status = "move"
            else:
                status = "review"
            target = str(_free_target(cfg.destinations[dec.folder].path, p.name)) if dec.folder else None
            plan.append(PlanItem(str(p), dec.folder, target, round(dec.confidence, 2), dec.tier, dec.reason, status))
            print(f"  {status:9s} {dec.tier:10s} {dec.confidence:4.2f}  {p.name} -> {dec.folder}")
    return plan


def _ask(prompt: str) -> str:
    # lstrip BOM: Windows PowerShell can prefix piped/redirected input with U+FEFF.
    return input(prompt).strip().lstrip("﻿").strip().lower()


def review(plan: List[PlanItem], cfg: Config) -> List[PlanItem]:
    """Interactive: resolve each 'review' item, then confirm the whole plan."""
    names = list(cfg.destinations)
    for it in plan:
        if it.status != "review":
            continue
        print(f"\nUNSURE: {Path(it.source).name}")
        print(f"  guess: {it.folder} ({it.confidence}, {it.tier}: {it.reason})")
        for i, n in enumerate(names, 1):
            print(f"  {i}. {n}")
        while True:  # re-ask on anything unrecognised; never silently accept the guess
            ans = _ask("  folder number, Enter = accept guess, s = skip: ")
            if ans == "s" or (ans == "" and it.folder is not None):
                break
            if ans.isdigit() and 1 <= int(ans) <= len(names):
                break
            print(f"  please type 1-{len(names)}, s, or Enter" + ("" if it.folder else " (no guess to accept)"))
        if ans == "s":
            it.status = "skip"
            continue
        if ans.isdigit():
            it.folder, it.tier, it.reason, it.confidence = names[int(ans) - 1], "you", "chosen in review", 1.0
        it.target = str(_free_target(cfg.destinations[it.folder].path, Path(it.source).name))
        it.status = "move"
    moves = [i for i in plan if i.status == "move"]
    print(f"\nPLAN: {len(moves)} moves")
    for i in moves:
        print(f"  {Path(i.source).name}  ->  {i.folder}   [{i.tier} {i.confidence}]")
    if _ask("Apply these moves? [y/N] ") != "y":
        for i in moves:
            i.status = "skip"
    return plan


def apply(plan: List[PlanItem], cfg: Config) -> Path:
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    log = cfg.state_dir / f"moves-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    with open(log, "w", encoding="utf-8") as f:
        for it in plan:
            if it.status != "move":
                continue
            target = _free_target(Path(it.target).parent, Path(it.source).name)  # re-check at move time
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(it.source, target)
            f.write(json.dumps({"from": it.source, "to": str(target), "folder": it.folder,
                                "tier": it.tier, "confidence": it.confidence}) + "\n")
    return log


def undo(cfg: Config, log_path: Optional[str] = None) -> int:
    logs = sorted(cfg.state_dir.glob("moves-*.jsonl"))
    log = Path(log_path) if log_path else (logs[-1] if logs else None)
    if not log or not log.exists():
        print("nothing to undo")
        return 0
    entries = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    restored = 0
    for e in reversed(entries):
        src, dst = Path(e["to"]), Path(e["from"])
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            restored += 1
        else:
            print(f"  cannot restore {src} -> {dst} (missing source or destination taken)")
    log.rename(log.with_suffix(".undone"))
    return restored


def save_plan(plan: List[PlanItem], cfg: Config) -> Path:
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    p = cfg.state_dir / f"plan-{time.strftime('%Y%m%d-%H%M%S')}.json"
    p.write_text(json.dumps([asdict(i) for i in plan], indent=1), encoding="utf-8")
    return p
