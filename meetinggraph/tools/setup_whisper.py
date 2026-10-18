"""Download the Whisper speech model once (needs internet this one time), then test it.

Usage:  python tools\\setup_whisper.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("faster-whisper is not installed. Run:  pip install -r requirements-asr.txt")
    sys.exit(1)

from app.config import load_config

cfg = load_config()["asr"]["faster_whisper"]
name = cfg.get("model", "base.en")
print(f"Downloading / loading Whisper model '{name}' (about 150 MB for base.en, one time)...")
t0 = time.time()
model = WhisperModel(name, device=cfg.get("device", "cpu"), compute_type=cfg.get("compute_type", "int8"))
print(f"Model ready in {time.time() - t0:.1f} s. It now works offline.")

try:
    import numpy as np
    silence = np.zeros(16000, dtype=np.float32)
    list(model.transcribe(silence)[0])
    print("Quick test passed. You can now use the red Record button in MeetingGraph.")
except Exception as e:  # the model loaded, which is what matters
    print(f"Model loaded; quick test skipped ({e}).")
