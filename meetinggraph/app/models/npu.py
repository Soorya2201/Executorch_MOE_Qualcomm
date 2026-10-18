"""One Hexagon NPU, one job at a time.

Whisper (speech) and Phi (notes) both run on the NPU. While recording, live transcription and live notes
run on different threads, so without a lock they would execute on the NPU at the same time. Every NPU call
in this app takes NPU_LOCK, so they queue instead (transcription catches up as soon as the notes are done).
"""
import threading

NPU_LOCK = threading.RLock()
