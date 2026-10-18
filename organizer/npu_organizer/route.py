"""Tiered routing: rules -> similarity -> Phi-4-mini (NPU). Each tier returns a Decision or None."""
import fnmatch
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import Config, Destination
from .digest import FileDigest, iter_files, make_digest


@dataclass
class Decision:
    folder: Optional[str]  # destination name, or None if no tier could decide
    confidence: float
    tier: str  # "rule" | "similarity" | "llm" | "fallback" | "none"
    reason: str


# ---------- tier 1: rules ----------

def rule_tier(d: FileDigest, dests: Dict[str, Destination]) -> Optional[Decision]:
    name = d.name.lower()
    by_pattern = [n for n, x in dests.items() if any(fnmatch.fnmatch(name, p) for p in x.patterns)]
    if len(by_pattern) == 1:
        return Decision(by_pattern[0], 1.0, "rule", "filename matches a rule pattern")
    by_ext = [n for n, x in dests.items() if d.ext in x.extensions]
    if len(by_ext) == 1 and not by_pattern:
        return Decision(by_ext[0], 0.95, "rule", f"extension {d.ext} belongs to this folder")
    return None  # no rule, or several rules disagree -> ambiguous


# ---------- tier 2: similarity to what is already in each folder ----------

_TOKEN = re.compile(r"[a-z0-9]{3,}")


def _tokens(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class SimilarityIndex:
    """TF-IDF centroid per destination, built from existing files' names + first/last pages."""

    def __init__(self, cfg: Config, max_files_per_folder: int = 200):
        docs: Dict[str, Counter] = {}
        for name, dest in cfg.destinations.items():
            tf = Counter(_tokens(name + " " + dest.description) * 3)
            if dest.path.exists():
                for i, p in enumerate(iter_files(dest.path, recursive=True)):
                    if i >= max_files_per_folder:
                        break
                    try:
                        tf.update(_tokens(make_digest(p, 400).text()))
                    except OSError:
                        continue
            docs[name] = tf
        n = len(docs)
        df = Counter(t for tf in docs.values() for t in tf)
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self.centroids = {name: self._vec(tf) for name, tf in docs.items()}

    def _vec(self, tf: Counter) -> Dict[str, float]:
        v = {t: c * self.idf.get(t, 1.0) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def rank(self, d: FileDigest):
        q = self._vec(Counter(_tokens(d.text())))
        scores = {n: sum(q.get(t, 0) * w for t, w in c.items()) for n, c in self.centroids.items()}
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def similarity_tier(d: FileDigest, index: SimilarityIndex, threshold: float) -> Optional[Decision]:
    ranked = index.rank(d)
    if not ranked:
        return None
    (best, s1), s2 = ranked[0], (ranked[1][1] if len(ranked) > 1 else 0.0)
    # Accept only a clear winner: above threshold and well ahead of the runner-up.
    if s1 >= threshold and s1 >= 1.5 * s2:
        return Decision(best, min(0.9, 0.5 + s1), "similarity", f"content resembles {best} (score {s1:.2f} vs {s2:.2f})")
    return None
