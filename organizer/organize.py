"""NPU file organizer.

  python organize.py plan  --config config.json            # dry run: decide and print, move nothing
  python organize.py run   --config config.json            # decide, then apply (asks first if review is on)
  python organize.py run   --config config.json --review off
  python organize.py undo  --config config.json            # put the last run's files back
"""
import argparse
import sys
from collections import Counter

from npu_organizer import config as config_mod
from npu_organizer.llm import NpuLLM
from npu_organizer.plan import apply, build_plan, review, save_plan, undo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "run", "undo"])
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--review", choices=["on", "off"], help="override the config's review switch")
    ap.add_argument("--no-llm", action="store_true", help="rules + similarity only")
    args = ap.parse_args()

    cfg = config_mod.load(args.config)
    if args.review:
        cfg.review = args.review == "on"

    if args.command == "undo":
        print(f"restored {undo(cfg)} files")
        return

    llm = None
    if cfg.llm.enabled and not args.no_llm:
        llm = NpuLLM(cfg.llm)
        print(f"LLM: {cfg.llm.model.name} ({cfg.llm.decoder_model_version}) on the NPU")
    print(f"review: {'ON (asks about unsure files)' if cfg.review else 'OFF (sorts everything)'}")

    plan = build_plan(cfg, llm)
    print(f"\nplan saved: {save_plan(plan, cfg)}")
    print("summary:", dict(Counter(i.status for i in plan)), dict(Counter(i.tier for i in plan)))

    if args.command == "plan":
        return
    if cfg.review:
        plan = review(plan, cfg)
    log = apply(plan, cfg)
    moved = sum(1 for i in plan if i.status == "move")
    print(f"moved {moved} files; undo log: {log}")
    for i in plan:
        if i.status == "duplicate":
            print(f"  duplicate left in place: {i.source} ({i.reason})")


if __name__ == "__main__":
    sys.exit(main())
