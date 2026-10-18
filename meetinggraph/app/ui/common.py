"""Shared UI helpers: background jobs, icons, markdown rendering of workspace files."""
from __future__ import annotations

import html
import re
import traceback

from PySide6.QtCore import QObject, QRect, QRunnable, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFontMetrics, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QListView, QListWidget, QStyledItemDelegate

STAGE_TEXT = {
    "recording": "Recording", "recorded": "Recorded", "transcribing": "Transcribing",
    "transcribed": "Transcript ready", "summarizing": "Summarizing + comparing features",
    "ready": "Ready for review", "saving": "Saving", "failed": "Failed",
    "cleanup_failed": "Saved — cleanup failed",
}
STAGE_COLOR = {"ready": "#107c10", "failed": "#c50f1f", "cleanup_failed": "#c50f1f",
               "transcribing": "#0f6cbd", "summarizing": "#0f6cbd", "saving": "#0f6cbd", "recording": "#c239b3"}
STATUS_COLOR = {"proposed": "#9e9e9e", "active": "#0f6cbd", "blocked": "#c50f1f", "done": "#107c10",
                "archived": "#d1d1d1"}


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

    def run_async(self) -> None:
        """Run on Qt's global thread pool; results arrive on the GUI thread via on_done/on_error."""
        from PySide6.QtCore import QThreadPool
        QThreadPool.globalInstance().start(self)

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


class _WrapDelegate(QStyledItemDelegate):
    """Item height for wrapped text that includes the stylesheet's item padding and margins (theme.py:
    QListWidget::item padding 9px 10px, margin 1px 2px). Qt's own size hint ignores them, which cut off
    the last line of multi-line items."""
    PAD_W, PAD_H = 2 * 10 + 2 * 2 + 8, 2 * 9 + 2 * 1 + 4
    ICON_GAP = 8

    def sizeHint(self, option, index):
        view = self.parent()
        width = view.viewport().width()
        text_w = width - self.PAD_W
        if index.data(Qt.DecorationRole) is not None:
            text_w -= max(16, view.iconSize().width()) + self.ICON_GAP
        r = QFontMetrics(option.font).boundingRect(QRect(0, 0, max(40, text_w), 100000), Qt.TextWordWrap,
                                                    str(index.data(Qt.DisplayRole) or ""))
        return QSize(max(40, width - 6), r.height() + self.PAD_H)


def wrapping_list() -> QListWidget:
    """A list whose items wrap to the panel width (never cut off at the right edge, no sideways scrolling)
    and re-measure their height whenever the panel is resized."""
    lst = QListWidget()
    lst.setItemDelegate(_WrapDelegate(lst))
    lst.setWordWrap(True)
    lst.setTextElideMode(Qt.ElideNone)
    lst.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    lst.setResizeMode(QListView.Adjust)
    return lst
