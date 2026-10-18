"""Print a HF repo's commit sha and its files as name|size|lfs_sha256 (one per line)."""
import sys

from huggingface_hub import HfApi

info = HfApi().model_info(sys.argv[1], files_metadata=True)
print("sha", info.sha)
for s in info.siblings:
    print(f"{s.rfilename}|{s.size}|{s.lfs.sha256 if s.lfs else ''}")
