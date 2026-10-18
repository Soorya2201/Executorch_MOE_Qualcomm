"""Configuration: where files come from, where they can go, and how decisions are made."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class Destination:
    name: str
    path: Path
    description: str
    extensions: List[str] = field(default_factory=list)  # e.g. [".py", ".ipynb"]
    patterns: List[str] = field(default_factory=list)  # filename globs, e.g. ["invoice*"]


@dataclass
class LLMConfig:
    enabled: bool
    runner: Path
    model: Path
    tokenizer: Path
    decoder_model_version: str
    max_seq_len: int  # must match the export's --max_seq_len
    max_new_tokens: int


@dataclass
class Config:
    sources: List[Path]
    destinations: Dict[str, Destination]
    review: bool  # True: ask about unsure files and confirm the plan. False: sort everything.
    confidence_threshold: float
    similarity_threshold: float
    page_chars: int  # characters taken from the first and from the last page
    recursive: bool
    state_dir: Path
    llm: LLMConfig


def load(path: str) -> Config:
    # utf-8-sig: Notepad and Windows PowerShell 5.1 write a BOM.
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    dests = {
        name: Destination(
            name=name,
            path=Path(d["path"]),
            description=d.get("description", ""),
            extensions=[e.lower() for e in d.get("extensions", [])],
            patterns=[p.lower() for p in d.get("patterns", [])],
        )
        for name, d in raw["destinations"].items()
    }
    llm = raw["llm"]
    return Config(
        sources=[Path(s) for s in raw["sources"]],
        destinations=dests,
        review=bool(raw.get("review", True)),
        confidence_threshold=float(raw.get("confidence_threshold", 0.6)),
        similarity_threshold=float(raw.get("similarity_threshold", 0.25)),
        page_chars=int(raw.get("page_chars", 1200)),
        recursive=bool(raw.get("recursive", False)),
        state_dir=Path(raw.get("state_dir", "state")),
        llm=LLMConfig(
            enabled=bool(llm.get("enabled", True)),
            runner=Path(llm["runner"]),
            model=Path(llm["model"]),
            tokenizer=Path(llm["tokenizer"]),
            decoder_model_version=llm.get("decoder_model_version", "phi_4_mini"),
            max_seq_len=int(llm.get("max_seq_len", 2048)),
            max_new_tokens=int(llm.get("max_new_tokens", 96)),
        ),
    )
