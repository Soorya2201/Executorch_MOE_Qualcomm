"""Summarize an ExecuTorch ETDump: which events ran and whether they were QNN-delegated."""
import argparse

from executorch.devtools import Inspector

parser = argparse.ArgumentParser()
parser.add_argument("--etdump", required=True)
args = parser.parse_args()

inspector = Inspector(etdump_path=args.etdump)
for block in inspector.event_blocks:
    print(f"== event block: {block.name}  ({len(block.events)} events)")
    for event in block.events:
        backend = getattr(event, "delegate_backend_name", None)
        delegated = getattr(event, "is_delegated_op", False)
        times = event.perf_data.raw if event.perf_data is not None else []
        mean = sum(times) / len(times) if times else float("nan")
        print(f"  {event.name:<45} delegated={delegated!s:<5} backend={backend}  n={len(times)} mean={mean:.3f}")
