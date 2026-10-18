"""Reusable UI components built only from design tokens: status dot, chip, segmented control, sidebar row,
card, toast, popover, thinking indicator. Painted widgets read colors from theme.c(), so they follow light/dark.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFontMetrics, QKeyEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QListWidget, QPushButton, QSizePolicy, QStyle,
                               QStyledItemDelegate, QStyleOptionViewItem, QVBoxLayout, QWidget)

from . import motion, theme
from . import tokens as tk

ROLE_SUBTITLE = Qt.UserRole + 1     # second line of a sidebar row ("3 meetings")
ROLE_DOT = Qt.UserRole + 2          # optional status-dot color for a row


def draw_shadow(p: QPainter, rect: QRectF, radius: float, spread: int = 10) -> None:
    """One soft shadow level, painted (cheaper and crisper than QGraphicsDropShadowEffect)."""
    r, g, b, a = tk.palette()["shadow"]
    p.setPen(Qt.NoPen)
    for i in range(spread, 0, -1):
        col = QColor(r, g, b, int(a * (1 - i / spread) ** 2 / 3))
        p.setBrush(col)
        p.drawRoundedRect(rect.adjusted(-i, -i + 3, i, i + 3), radius + i, radius + i)


# ---------------------------------------------------------------- status dot
class StatusDot(QWidget):
    def __init__(self, status: str = "proposed", size: int = 8, parent=None):
        super().__init__(parent)
        self._d = size
        self.setFixedSize(size + 4, size + 4)
        self.set_status(status)

    def set_status(self, status: str):
        self.status = status or "proposed"
        self.setToolTip(self.status.capitalize())
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(tk.status_color(self.status)))
        p.drawEllipse(QRectF(2, 2, self._d, self._d))


# ---------------------------------------------------------------- chip
class Chip(QPushButton):
    """Small rounded, clickable label: '+ DNA'."""
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setObjectName("chip")
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)


class FlowWrap(QWidget):
    """Lays out children left-to-right and wraps to new lines (for chips)."""
    def __init__(self, spacing: int = tk.SPACE[8], parent=None):
        super().__init__(parent)
        self._items: list[QWidget] = []
        self._sp = spacing
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

    def add(self, w: QWidget):
        w.setParent(self)
        w.show()
        self._items.append(w)
        self.updateGeometry()
        self._relayout(self.width())

    def clear(self):
        for w in self._items:
            w.deleteLater()
        self._items = []
        self.updateGeometry()

    def _relayout(self, width: int) -> int:
        x = y = line_h = 0
        for w in self._items:
            s = w.sizeHint()
            if x and x + s.width() > width:
                x, y = 0, y + line_h + self._sp
                line_h = 0
            w.setGeometry(x, y, s.width(), s.height())
            x += s.width() + self._sp
            line_h = max(line_h, s.height())
        return y + line_h

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        x = y = line_h = 0
        for it in self._items:
            s = it.sizeHint()
            if x and x + s.width() > w:
                x, y, line_h = 0, y + line_h + self._sp, 0
            x += s.width() + self._sp
            line_h = max(line_h, s.height())
        return y + line_h

    def sizeHint(self):
        return QSize(200, self.heightForWidth(max(self.width(), 200)))

    def resizeEvent(self, e):
        self._relayout(e.size().width())


# ---------------------------------------------------------------- segmented control
class SegmentedControl(QWidget):
    """Apple-style segmented control. The selection indicator slides between segments. Optional count badges.
    Keyboard: Left/Right (or Tab into it then arrows)."""
    currentChanged = Signal(int)
    PAD_X, H, INSET = 16, 32, 3

    def __init__(self, labels: list[str], parent=None):
        super().__init__(parent)
        self.labels = list(labels)
        self.badges: dict[int, int] = {}
        self._cur = 0
        self._hover = -1
        self._ind: QRectF | None = None
        self._kbd = False
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setFixedHeight(self.H)
        self.setAccessibleName("View")
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    # geometry
    def _seg_widths(self) -> list[int]:
        fm = QFontMetrics(theme.font("body"))
        out = []
        for i, t in enumerate(self.labels):
            w = fm.horizontalAdvance(t) + 2 * self.PAD_X
            if self.badges.get(i):
                w += fm.horizontalAdvance(str(self.badges[i])) + 16
            out.append(max(w, 72))
        return out

    def _seg_rects(self) -> list[QRectF]:
        x, out = float(self.INSET), []
        for w in self._seg_widths():
            out.append(QRectF(x, self.INSET, w, self.H - 2 * self.INSET))
            x += w
        return out

    def sizeHint(self):
        return QSize(sum(self._seg_widths()) + 2 * self.INSET, self.H)

    def minimumSizeHint(self):
        return self.sizeHint()

    # api
    def currentIndex(self) -> int:
        return self._cur

    def setCurrentIndex(self, i: int, animate: bool = True):
        if not 0 <= i < len(self.labels):
            return
        rects = self._seg_rects()
        start = self._ind or rects[self._cur]
        changed = i != self._cur
        self._cur = i
        if animate:
            motion.value(self, QRectF(start), rects[i], "select", self._set_ind)
        else:
            self._set_ind(rects[i])
        if changed:
            self.currentChanged.emit(i)

    def setBadge(self, i: int, n: int):
        if self.badges.get(i, 0) == n:
            return
        self.badges[i] = n
        self.setFixedWidth(self.sizeHint().width())
        self._ind = self._seg_rects()[self._cur]
        self.update()

    def _set_ind(self, r):
        self._ind = QRectF(r)
        self.update()

    # painting
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rects = self._seg_rects()
        if self._ind is None:
            self._ind = rects[self._cur]
        track = QRectF(self.rect())
        p.setPen(Qt.NoPen)
        p.setBrush(theme.qcolor("control"))
        p.drawRoundedRect(track, 8, 8)
        if 0 <= self._hover < len(rects) and self._hover != self._cur:
            p.setBrush(theme.qcolor("hover"))
            p.drawRoundedRect(rects[self._hover], 6, 6)
        draw_shadow(p, self._ind, 6, spread=3)
        p.setBrush(theme.qcolor("surface"))
        p.drawRoundedRect(self._ind, 6, 6)
        f = theme.font("body")
        fm = QFontMetrics(f)
        for i, (r, t) in enumerate(zip(rects, self.labels)):
            sel = i == self._cur
            f.setWeight(QFont_weight(600 if sel else 400))
            p.setFont(f)
            p.setPen(theme.qcolor("text" if sel else "secondary"))
            n = self.badges.get(i, 0)
            if n:
                bw = fm.horizontalAdvance(str(n)) + 10
                tw = fm.horizontalAdvance(t)
                x0 = r.center().x() - (tw + 6 + bw) / 2
                p.drawText(QRectF(x0, r.top(), tw + 2, r.height()), Qt.AlignVCenter | Qt.AlignLeft, t)
                br = QRectF(x0 + tw + 6, r.center().y() - 9, bw, 18)
                p.setPen(Qt.NoPen)
                p.setBrush(theme.qcolor("accent"))
                p.drawRoundedRect(br, 9, 9)
                bf = theme.font("meta")
                bf.setWeight(QFont_weight(600))
                p.setFont(bf)
                p.setPen(theme.qcolor("on_accent"))
                p.drawText(br, Qt.AlignCenter, str(n))
            else:
                p.drawText(r, Qt.AlignCenter, t)
        if self.hasFocus() and self._kbd:
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(theme.qcolor("accent"), 2))
            p.drawRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), 8, 8)

    # input
    def _index_at(self, pos) -> int:
        for i, r in enumerate(self._seg_rects()):
            if r.contains(pos):
                return i
        return -1

    def mouseMoveEvent(self, e):
        h = self._index_at(e.position())
        if h != self._hover:
            self._hover = h
            self.update()

    def leaveEvent(self, _):
        self._hover = -1
        self.update()

    def mousePressEvent(self, e):
        self._kbd = False
        i = self._index_at(e.position())
        if i >= 0:
            self.setCurrentIndex(i)

    def keyPressEvent(self, e: QKeyEvent):
        if e.key() in (Qt.Key_Left, Qt.Key_Right):
            self._kbd = True
            self.setCurrentIndex(max(0, min(len(self.labels) - 1, self._cur + (1 if e.key() == Qt.Key_Right else -1))))
        else:
            super().keyPressEvent(e)

    def focusInEvent(self, e):
        self._kbd = e.reason() in (Qt.TabFocusReason, Qt.BacktabFocusReason, Qt.ShortcutFocusReason)
        self.update()

    def focusOutEvent(self, e):
        self.update()

    def resizeEvent(self, e):
        self._ind = self._seg_rects()[self._cur]


def QFont_weight(w: int):
    from PySide6.QtGui import QFont
    return QFont.Weight(w)


# ---------------------------------------------------------------- sidebar / list rows
class RowDelegate(QStyledItemDelegate):
    """List row: optional status dot, title (body) and optional subtitle (meta). Rounded soft selection (accent at
    ~12%) and a very light rounded hover. Works for QListWidget items:
        item.setText(title); item.setData(ROLE_SUBTITLE, '3 meetings'); item.setData(ROLE_DOT, '#34C759')"""
    def __init__(self, parent=None, dense: bool = False):
        super().__init__(parent)
        self.dense = dense

    def sizeHint(self, option, index):
        two = bool(index.data(ROLE_SUBTITLE))
        return QSize(option.rect.width(), (48 if two else 34) if not self.dense else (40 if two else 30))

    def paint(self, p: QPainter, option: QStyleOptionViewItem, index):
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(option.rect).adjusted(6, 1, -6, -1)
        enabled = bool(index.flags() & Qt.ItemIsEnabled)
        if option.state & QStyle.State_Selected:
            p.setPen(Qt.NoPen)
            p.setBrush(theme.qcolor("accent_soft"))
            p.drawRoundedRect(r, tk.RADIUS["control"], tk.RADIUS["control"])
        elif option.state & QStyle.State_MouseOver and enabled:
            p.setPen(Qt.NoPen)
            p.setBrush(theme.qcolor("hover"))
            p.drawRoundedRect(r, tk.RADIUS["control"], tk.RADIUS["control"])
        view = option.widget
        if (option.state & QStyle.State_HasFocus) and view is not None and view.hasFocus() \
                and view.property("kbd_focus"):
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(theme.qcolor("accent"), 2))
            p.drawRoundedRect(r.adjusted(1, 1, -1, -1), tk.RADIUS["control"], tk.RADIUS["control"])
        x = r.left() + 10
        dot = index.data(ROLE_DOT)
        if dot:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(dot))
            p.drawEllipse(QRectF(x, r.center().y() - 4, 8, 8))
            x += 18
        title = index.data(Qt.DisplayRole) or ""
        sub = index.data(ROLE_SUBTITLE) or ""
        tf, mf = theme.font("body"), theme.font("meta")
        if option.state & QStyle.State_Selected:
            tf.setWeight(QFont_weight(600))
        text_w = r.right() - x - 8
        p.setPen(theme.qcolor("text" if enabled else "secondary"))
        p.setFont(tf)
        tfm = QFontMetrics(tf)
        if sub:
            p.drawText(QRectF(x, r.top() + 5, text_w, tfm.height()), Qt.AlignLeft | Qt.AlignVCenter,
                       tfm.elidedText(title, Qt.ElideRight, int(text_w)))
            p.setFont(mf)
            p.setPen(theme.qcolor("secondary"))
            mfm = QFontMetrics(mf)
            p.drawText(QRectF(x, r.top() + 5 + tfm.height(), text_w, mfm.height() + 2), Qt.AlignLeft | Qt.AlignVCenter,
                       mfm.elidedText(sub, Qt.ElideRight, int(text_w)))
        else:
            p.drawText(QRectF(x, r.top(), text_w, r.height()), Qt.AlignLeft | Qt.AlignVCenter,
                       tfm.elidedText(title, Qt.ElideRight, int(text_w)))
        p.restore()


class _KbdFocus(QObject):
    """Marks a list as keyboard-focused (for focus rings) only when focus arrived by keyboard."""
    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.FocusIn:
            obj.setProperty("kbd_focus", ev.reason() in (Qt.TabFocusReason, Qt.BacktabFocusReason,
                                                         Qt.ShortcutFocusReason))
        elif ev.type() == QEvent.MouseButtonPress:
            obj.setProperty("kbd_focus", False)
        elif ev.type() == QEvent.KeyPress:
            obj.setProperty("kbd_focus", True)
        return False


_kbd = _KbdFocus()


def styled_list(lst: QListWidget, dense: bool = False) -> QListWidget:
    """Give a QListWidget the sidebar row look."""
    lst.setItemDelegate(RowDelegate(lst, dense))
    lst.setMouseTracking(True)
    lst.viewport().setAttribute(Qt.WA_Hover, True)
    lst.installEventFilter(_kbd)
    lst.setVerticalScrollMode(QListWidget.ScrollPerPixel)
    lst.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    return lst


# ---------------------------------------------------------------- card
class Card(QFrame):
    """Rounded surface without a border; contents laid out with token spacing."""
    def __init__(self, parent=None, pad: int = tk.SPACE[16]):
        super().__init__(parent)
        self.setObjectName("card")
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(pad, pad, pad, pad)
        self.lay.setSpacing(tk.SPACE[8])


def hairline() -> QFrame:
    f = QFrame()
    f.setObjectName("hairline")
    return f


# ---------------------------------------------------------------- floating surfaces
class _Floating(QWidget):
    """Rounded surface with the single soft shadow level, painted."""
    SHADOW = 12

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(self.SHADOW, self.SHADOW, -self.SHADOW, -self.SHADOW)
        draw_shadow(p, r, tk.RADIUS["card"], spread=self.SHADOW - 2)
        p.setPen(QPen(theme.qcolor("separator"), 1))
        p.setBrush(theme.qcolor("surface"))
        p.drawRoundedRect(r, tk.RADIUS["card"], tk.RADIUS["card"])


class Toast(_Floating):
    """Bottom-right notice that auto-dismisses (~6 s). One at a time; a new one replaces the old.
        toast.show_message("Meeting Oct 18 needs review", "Open", callback)"""
    def __init__(self, host: QWidget):
        super().__init__(host)
        self.host = host
        self.setAttribute(Qt.WA_TranslucentBackground)
        lay = QHBoxLayout(self)
        s = self.SHADOW
        lay.setContentsMargins(s + 14, s + 10, s + 8, s + 10)
        lay.setSpacing(tk.SPACE[8])
        self.text = QLabel()
        self.text.setWordWrap(True)
        self.text.setMinimumWidth(240)
        self.text.setMaximumWidth(340)
        self.action = QPushButton()
        self.action.setObjectName("link")
        self.action.setCursor(Qt.PointingHandCursor)
        self.close_btn = QPushButton("✕")
        self.close_btn.setObjectName("quiet")
        self.close_btn.setAccessibleName("Dismiss")
        self.close_btn.setFixedWidth(28)
        lay.addWidget(self.text, 1)
        lay.addWidget(self.action)
        lay.addWidget(self.close_btn)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.dismiss)
        self.close_btn.clicked.connect(self.dismiss)
        self._cb = None
        self.action.clicked.connect(self._act)
        host.installEventFilter(self)
        self.hide()

    def show_message(self, text: str, action: str = "", callback=None, ms: int = tk.TOAST_MS):
        self.text.setText(text)
        self.action.setText(action)
        self.action.setVisible(bool(action))
        self._cb = callback
        self.adjustSize()
        self._place()
        self.show()
        self.raise_()
        motion.fade(self, 1.0, "toast")
        self.timer.start(ms)

    def enterEvent(self, _):
        self.timer.stop()                 # don't vanish while the pointer is on it

    def leaveEvent(self, _):
        if self.isVisible():
            self.timer.start(2500)

    def _act(self):
        cb = self._cb
        self.dismiss()
        if cb:
            cb()

    def dismiss(self):
        self.timer.stop()
        motion.fade(self, 0.0, "toast", on_done=self.hide)

    def _place(self):
        m = tk.SPACE[16]
        self.move(self.host.width() - self.width() - m, self.host.height() - self.height() - m - 24)

    def eventFilter(self, obj, ev):
        if obj is self.host and ev.type() == QEvent.Resize and self.isVisible():
            self._place()
        return False


class Popover(_Floating):
    """Small floating panel anchored to a widget. Closes on Esc or click outside (Qt.Popup)."""
    def __init__(self, parent=None):
        super().__init__(parent, Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.lay = QVBoxLayout(self)
        s = self.SHADOW
        self.lay.setContentsMargins(s + 14, s + 12, s + 14, s + 12)
        self.lay.setSpacing(tk.SPACE[8])

    def show_for(self, anchor: QWidget, above: bool = False, align_right: bool = True):
        self.adjustSize()
        g = anchor.mapToGlobal(QPoint(0, 0))
        x = g.x() + anchor.width() - self.width() + self.SHADOW if align_right else g.x() - self.SHADOW
        y = g.y() - self.height() + self.SHADOW - 4 if above else g.y() + anchor.height() - self.SHADOW + 4
        self.move(x, y)
        self.show()
        motion.fade(self, 1.0, "hover")


# ---------------------------------------------------------------- thinking indicator
class Thinking(QWidget):
    """Gentle 'Thinking…' indicator: three dots that softly pulse in turn (no spinner)."""
    def __init__(self, text: str = "Thinking", parent=None):
        super().__init__(parent)
        self.text = text
        self._phase = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.setFixedHeight(24)
        self.setMinimumWidth(110)

    def start(self):
        self.show()
        if motion.enabled():
            self.timer.start(260)

    def stop(self):
        self.timer.stop()
        self.hide()

    def _tick(self):
        self._phase = (self._phase + 1) % 3
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setFont(theme.font("body"))
        p.setPen(theme.qcolor("secondary"))
        fm = QFontMetrics(p.font())
        p.drawText(QRect(0, 0, fm.horizontalAdvance(self.text) + 2, self.height()), Qt.AlignVCenter, self.text)
        x = fm.horizontalAdvance(self.text) + 6
        base = theme.qcolor("secondary")
        for i in range(3):
            col = QColor(base)
            col.setAlphaF(1.0 if (i == self._phase or not motion.enabled()) else 0.35)
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            p.drawEllipse(QRectF(x + i * 9, self.height() / 2 - 2.5, 5, 5))
