"""Checks this machine for what MeetingGraph needs.  python tools\\check_env.py"""
from __future__ import annotations

import json
import platform
import struct
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
quiet = "--quiet" in sys.argv
ok = True


def say(status: str, msg: str):
    if not quiet or status != "OK":
        print(f"[{status:4}] {msg}")


say("INFO", f"Python {platform.python_version()} · {struct.calcsize('P') * 8}-bit · machine {platform.machine()} · {sys.platform}")
if sys.version_info < (3, 10):
    say("FAIL", "Python 3.10+ required (3.11 or 3.12 recommended).")
    ok = False

for mod, why in (("yaml", "config"), ("PySide6", "desktop UI")):
    try:
        __import__(mod)
        say("OK", f"{mod} ({why})")
    except Exception as e:
        say("FAIL", f"{mod} missing ({why}): {e}  ->  pip install -r requirements.txt")
        ok = False

try:
    from app.capture.recorder import list_devices
    d = list_devices()
    if d["error"]:
        say("WARN", f"Audio capture unavailable: {d['error']} (import/typed transcripts still work)")
    else:
        say("OK", f"Audio: {len(d['mics'])} microphone(s), {len(d['speakers'])} output device(s)")
        if not quiet:
            for m in d["mics"]:
                print(f"         mic: {m['name']}")
            for s in d["speakers"]:
                print(f"         out: {s['name']}  (loopback source)")
except Exception as e:
    say("WARN", f"Audio check failed: {e}")

from app.config import load_config  # noqa: E402

cfg = load_config()
say("INFO", f"LLM backend: {cfg['llm']['backend']} · ASR backend: {cfg['asr']['backend']} · workspace: {cfg['workspace']}")

if cfg["asr"]["backend"] == "faster_whisper":
    try:
        import faster_whisper  # noqa: F401
        say("OK", "faster-whisper installed")
    except Exception as e:
        say("FAIL", f"asr.backend is faster_whisper but it is not installed: {e} -> pip install -r requirements-asr.txt")
        ok = False

if cfg["llm"]["backend"] == "ollama":
    url = cfg["llm"]["ollama"]["url"].rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            names = [m["name"] for m in json.loads(r.read().decode()).get("models", [])]
        want = cfg["llm"]["ollama"]["model"]
        if any(n == want or n.startswith(want + ":") or n.split(":")[0] == want for n in names):
            say("OK", f"Ollama running with {want}")
        else:
            say("WARN", f"Ollama running but model '{want}' not pulled -> ollama pull {want}. Have: {names}")
    except Exception as e:
        say("WARN", f"Ollama not reachable at {url} ({e}). The app falls back to the rule-based extractor.")

for be in ("llm", "asr"):
    if cfg[be]["backend"] == "executorch_runner" and not cfg[be]["executorch_runner"]["command"]:
        say("FAIL", f"{be}.backend is executorch_runner but {be}.executorch_runner.command is empty in config.yaml")
        ok = False

ws = Path(cfg["workspace"])
try:
    ws.mkdir(parents=True, exist_ok=True)
    (ws / ".write_test").write_text("x")
    (ws / ".write_test").unlink()
    say("OK", f"Workspace writable: {ws}")
except Exception as e:
    say("FAIL", f"Workspace not writable: {e}")
    ok = False

if not quiet:
    print("\nAll required checks passed." if ok else "\nSome required checks failed.")
sys.exit(0 if ok else 1)
