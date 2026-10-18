"""Shared UI helpers: background jobs, icons, markdown rendering of workspace files."""
from __future__ import annotations

import html
import re
import traceback

from PySide6.QtCore import QObject, QRunnable, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPixmap

STAGE_TEXT = {
    "recording": "Recording", "recorded": "Recorded", "transcribing": "Transcribing",
    "transcribed": "Transcript ready", "summarizing": "Summarizing + comparing features",
    "ready": "Ready for review", "saving": "Saving", "failed": "Failed",
    "cleanup_failed": "Saved — cleanup failed",
}
STAGE_COLOR = {"ready": "#2e7d32", "failed": "#c62828", "cleanup_failed": "#c62828",
               "transcribing": "#1565c0", "summarizing": "#1565c0", "saving": "#1565c0", "recording": "#ad1457"}
STATUS_COLOR = {"proposed": "#9e9e9e", "active": "#1e88e5", "blocked": "#e53935", "done": "#43a047",
                "archived": "#bdbdbd"}


class _Signals(QObject):
    done = Signal(object)
    failed = Signal(str)


class Job(QRunnable):
    """Run fn(*args) on a pool thread; deliver the result or the error on the GUI thread.

    Jobs are kept alive in _live until their result has been delivered; otherwise the
    runnable (and its signal object) can be freed before the queued signal arrives.
    """
    _live: set = set()

    def __init__(self, fn, *args, on_done=None, on_error=None):
        super().__init__()
        self.setAutoDelete(False)
        self.fn, self.args = fn, args
        self.signals = _Signals()
        if on_done:
            self.signals.done.connect(on_done)
        if on_error:
            self.signals.failed.connect(on_error)
        self.signals.done.connect(self._release)
        self.signals.failed.connect(self._release)
        Job._live.add(self)

    def _release(self, *_):
        Job._live.discard(self)

    def run(self):
        try:
            res = self.fn(*self.args)
        except Exception as e:
            traceback.print_exc()
            self.signals.failed.emit(f"{type(e).__name__}: {e}")
            return
        self.signals.done.emit(res)


def dot_icon(color: str, size: int = 32) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QBrush(QColor(color)))
    p.setPen(Qt.NoPen)
    p.drawEllipse(2, 2, size - 4, size - 4)
    p.end()
    return QIcon(pm)


WIKI = re.compile(r"\[\[([0-9a-f-]{36})\|([^\]]+)\]\]")


def md_with_links(body: str, kind_of) -> str:
    """Turn [[id|Title]] wikilinks into clickable markdown links: open:<kind>:<id>."""
    def rep(m):
        fid, title = m.group(1), m.group(2)
        return f"[{title}](open:{kind_of(fid)}:{fid})"
    return WIKI.sub(rep, body)


def esc(s: str) -> str:
    return html.escape(s or "")
