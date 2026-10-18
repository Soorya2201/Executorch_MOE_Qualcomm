"""Exercise the NPU backends through the app's own code paths (config.yaml -> make_asr/make_llm).

  python tools\\npu_check.py asr C:\\path\\meeting.wav [--ref refs.json]   # Whisper on the NPU (+ WER)
  python tools\\npu_check.py extract tests\\fixtures\\atlas_03_natural_design_review.txt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_config  # noqa: E402
from app.models.asr import make_asr  # noqa: E402
from app.models.llm import make_llm  # noqa: E402
from app.pipeline.extract import extract_draft  # noqa: E402
from app.pipeline.transcript import from_text, to_text  # noqa: E402


def _words(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", s.lower()).split()


def _wer(ref: list[str], hyp: list[str]) -> float:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1] / max(1, len(ref))


def cmd_asr(args, cfg):
    asr = make_asr(cfg)
    print(f"ASR backend: {asr.label}")
    t0 = time.perf_counter()
    segs = asr.transcribe(args.audio)
    dt = time.perf_counter() - t0
    print(to_text(segs))
    print(f"{len(segs)} segments in {dt:.1f} s (includes one model load)")
    if args.ref:
        ref = " ".join(json.loads(Path(args.ref).read_text(encoding="utf-8")))
        hyp = " ".join(s["text"] for s in segs)
        print(f"WER vs reference: {_wer(_words(ref), _words(hyp)) * 100:.1f}%")


def cmd_extract(args, cfg):
    llm = make_llm(cfg)
    print(f"LLM backend: {llm.label}")
    segs = from_text(Path(args.transcript).read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    draft = extract_draft(llm, segs, Path(args.transcript).stem, args.existing or [],
                          cfg["llm"]["max_transcript_chars_per_chunk"])
    dt = time.perf_counter() - t0
    print(f"\nSUMMARY: {draft['summary']}")
    for k in ("decisions", "action_items", "open_questions"):
        print(f"\n{k.upper()}:")
        for x in draft[k]:
            owner = f" [owner: {x.get('owner') or '-'}]" if k == "action_items" else ""
            print(f"  {'ok ' if x.get('supported') else 'UNSUPPORTED'} {x['text']}{owner}  <- \"{x.get('quote', '')}\"")
    print("\nFEATURES:")
    for f in draft["feature_mentions"]:
        ev = sum(1 for e in f["evidence"] if e.get("supported"))
        print(f"  {f['action']:7s} {f['title']}  deps={f['depends_on']}  evidence {ev}/{len(f['evidence'])} supported")
    print(f"\nextraction took {dt:.0f} s")


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
a = sub.add_parser("asr")
a.add_argument("audio")
a.add_argument("--ref")
e = sub.add_parser("extract")
e.add_argument("transcript")
e.add_argument("--existing", nargs="*")
args = ap.parse_args()
config = load_config()
{"asr": cmd_asr, "extract": cmd_extract}[args.cmd](args, config)
