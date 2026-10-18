"""Map: an Obsidian-style knowledge graph drawn with plain QGraphicsView items (software raster, no OpenGL,
no QGraphicsEffect, no timers while idle). Every size, color and duration comes from tokens.GRAPH.

Layout is computed rarely, in a LowestPriority worker (map_jobs.py), cached per project and never while the
AI is busy; opening the Map just loads saved positions and draws once.

Public API kept for the main window:  MapView.open_node(kind, id)  MapView.show_graph(graph_dict)
"""
from __future__ import annotations

import math

from PySide6.QtCore import (QEasingCurve, QEvent, QLineF, QPoint, QPointF, QRect, QRectF, QSize, Qt, QThread, QTimer,
                            QVariantAnimation, Signal)
from PySide6.QtGui import (QBrush, QColor, QFont, QFontMetrics, QFontMetricsF, QKeySequence, QPainter, QPainterPath,
                           QPen, QShortcut, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QGraphicsItem, QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QPushButton, QSizePolicy, QStyle, QStyledItemDelegate,
                               QVBoxLayout, QWidget)

from . import motion, theme
from . import tokens as tk
from .components import Popover
from .map_jobs import MapJob, save_settings_async
from .map_panel import DEFAULTS, FocusChip, MapPanel, SearchBox, map_qss

STRONG_VERBS = {"depends on", "needs", "produces"}
SETTINGS_V = 2          # bump when defaults change in a way old saved settings should not override


# ------------------------------------------------------------------ helpers
def node_color(status: str | None, revision: int = 1) -> QColor:
    if status == "blocked":
        return QColor(tk.g("conflict"))
    if status in ("proposed", "active", "done", None, "") and revision and int(revision) > 1:
        return QColor(tk.STATUS["changed"])
    if status in ("active", "done"):
        return QColor(tk.STATUS["active"])
    if status == "archived":
        return QColor(tk.STATUS["archived"])
    return QColor(tk.STATUS["proposed"])


def node_radius(degree: int, scale: float = 1.0) -> float:
    r = tk.g("node.base_r") + tk.g("node.deg_k") * math.sqrt(max(degree, 0))
    return max(tk.g("node.min_r"), min(tk.g("node.max_r"), r)) * scale


def text_color(alpha: float) -> QColor:
    c = theme.qcolor("text")
    c.setAlphaF(alpha)
    return c


def short(name: str) -> str:
    n = tk.g("label.max_chars")
    return name if len(name) <= n else name[: n - 1].rstrip() + "…"


def edge_kind(e: dict) -> str:
    """strong | weak | meeting"""
    if e["type"].startswith("meeting_"):
        return "meeting"
    verb = (e.get("verb") or ("depends on" if e["type"] == "feature_depends_on_feature" else "")).lower()
    return "strong" if verb in STRONG_VERBS else "weak"


def edge_verb(e: dict) -> str:
    if e["type"] == "feature_depends_on_feature":
        return "depends on"
    if e["type"] == "meeting_updates_feature":
        return "updated"
    if e["type"] == "meeting_relates_to_meeting":
        return "shares a feature with"
    return e.get("verb") or "is connected to"


# ------------------------------------------------------------------ items
class NodeItem(QGraphicsItem):
    """Flat filled circle (feature) or small rounded square (meeting). No outline; ring only when focused."""
    def __init__(self, canvas: "MapCanvas", data: dict, degree: int):
        super().__init__()
        self.canvas, self.data, self.degree = canvas, data, degree
        self.nid, self.kind = data["id"], data["type"]
        self.ring = False
        self.kbd = False
        self.pinned = False
        self._drag = False
        self._press_screen = None
        self._grab = QPointF()
        self.setZValue(2)
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)
        self.setCursor(Qt.PointingHandCursor)
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        self.update_style()

    def update_style(self):
        s = self.canvas.settings
        if self.kind == "feature":
            self.r = node_radius(self.degree, s["node_scale"])
            mode = s.get("color_by", "group")
            if mode == "group":
                gi = self.canvas.groups.get(self.nid, -1)
                pal = tk.g("categorical")
                self.color = QColor(pal[gi % len(pal)] if gi >= 0 else tk.g("ungrouped"))
            elif mode == "meeting":
                self.color = QColor(self.canvas.meeting_color.get(self.nid, tk.STATUS["proposed"]))
            elif mode == "none":
                self.color = QColor(tk.STATUS["proposed"])
            else:
                self.color = node_color(self.data.get("status"), self.data.get("revision", 1))
        else:
            self.r = tk.g("meeting.size") / 2 * s["node_scale"]
            self.color = theme.qcolor("accent")
        self.prepareGeometryChange()
        self.update()

    def boundingRect(self):
        m = self.r + tk.g("node.ring_gap") + tk.g("node.ring_w") + 1
        return QRectF(-m, -m, 2 * m, 2 * m)

    def paint(self, p: QPainter, _opt, _w=None):
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(self.color)
        if self.kind == "feature":
            p.drawEllipse(QPointF(0, 0), self.r, self.r)
        else:
            rr = tk.g("meeting.radius")
            p.drawRoundedRect(QRectF(-self.r, -self.r, 2 * self.r, 2 * self.r), rr, rr)
        if self.ring:
            g = tk.g("node.ring_gap") + tk.g("node.ring_w") / 2
            p.setBrush(Qt.NoBrush)
            # keyboard focus ring uses the accent so it is visible on any node color
            p.setPen(QPen(theme.qcolor("accent") if self.kbd else self.color, tk.g("node.ring_w")))
            if self.kind == "feature":
                p.drawEllipse(QPointF(0, 0), self.r + g, self.r + g)
            else:
                p.drawRoundedRect(QRectF(-self.r - g, -self.r - g, 2 * (self.r + g), 2 * (self.r + g)), 4, 4)
        if self.pinned:                       # small pin dot: this node was placed by hand
            d = tk.g("pin.r")
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(tk.g("bg")))
            c = QPointF(self.r * 0.75, -self.r * 0.75)
            p.drawEllipse(c, d + 1, d + 1)
            p.setBrush(theme.qcolor("secondary"))
            p.drawEllipse(c, d, d)

    # hover: the Obsidian focus effect
    def hoverEnterEvent(self, ev):
        self.canvas.hover_node(self)

    def hoverLeaveEvent(self, ev):
        self.canvas.hover_node(None, leaving=self)

    # click selects, drag moves only this node, double-click opens
    def mousePressEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            ev.ignore()                       # right button -> context menu
            return
        self._press_screen = ev.screenPos()
        self._grab = ev.scenePos() - self.pos()
        self._drag = False
        ev.accept()

    def mouseMoveEvent(self, ev):
        if self._press_screen is None:
            return
        if not self._drag and (ev.screenPos() - self._press_screen).manhattanLength() > 3:
            self._drag = True
            self.setCursor(Qt.ClosedHandCursor)
        if self._drag:
            self.canvas.move_node(self, ev.scenePos() - self._grab)

    def mouseReleaseEvent(self, ev):
        if self._press_screen is None:
            return
        self._press_screen = None
        if self._drag:
            self._drag = False
            self.setCursor(Qt.PointingHandCursor)
            self.canvas.end_drag(self)
        else:
            self.canvas.select_node(self)

    def mouseDoubleClickEvent(self, ev):
        self.canvas.open_node.emit(self.kind, self.nid)
        ev.accept()

    def contextMenuEvent(self, ev):
        from PySide6.QtWidgets import QMenu
        m = QMenu()
        a_open = m.addAction("Open")
        a_unpin = m.addAction("Unpin") if self.pinned else None
        chosen = m.exec(ev.screenPos())
        if chosen is a_open:
            self.canvas.open_node.emit(self.kind, self.nid)
        elif a_unpin is not None and chosen is a_unpin:
            self.canvas.unpin(self)


class LabelItem(QGraphicsItem):
    """Constant-size label under its node (ignores zoom). Never overlaps the node: it starts `gap` px below."""
    def __init__(self, node: NodeItem):
        super().__init__()
        self.node = node
        self.full = node.data.get("label", "")
        self.text = short(self.full)
        self.strong = False
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setZValue(3)
        if self.text != self.full:
            self.setToolTip(self.full)
        self._font()

    def _font(self):
        f = QFont()
        f.setFamilies([tk.FONT_FAMILY, tk.FONT_FALLBACK])
        f.setPixelSize(tk.g("label.px"))
        f.setWeight(QFont.DemiBold if self.strong else QFont.Normal)
        self.font = f
        fm = QFontMetricsF(f)
        self.w, self.h = fm.horizontalAdvance(self.text) + 4, fm.height()
        self.prepareGeometryChange()

    def set_strong(self, on: bool):
        if on != self.strong:
            self.strong = on
            self._font()
            self.update()

    def boundingRect(self):
        return QRectF(-self.w / 2, tk.g("label.gap"), self.w, self.h)

    def paint(self, p: QPainter, _o, _w=None):
        p.setFont(self.font)
        p.setPen(theme.qcolor("text" if self.strong else "secondary"))
        p.drawText(self.boundingRect(), Qt.AlignHCenter | Qt.AlignTop, self.text)

    def place(self):
        """Anchor at the bottom of the node (scene units); the text itself is offset in screen pixels."""
        self.setPos(self.node.pos() + QPointF(0, self.node.r))


class EdgeItem(QGraphicsItem):
    """Hair-thin straight link, ~1 px on screen at any zoom. Reciprocal links share one item."""
    MARGIN = 30.0   # scene units: covers a cosmetic 2 px pen and 5 px chevrons down to zoom 0.2

    def __init__(self, canvas: "MapCanvas", a: NodeItem, b: NodeItem, edges: list[dict]):
        super().__init__()
        self.canvas, self.a, self.b, self.edges = canvas, a, b, edges
        kinds = {edge_kind(e) for e in edges}
        self.kind = "strong" if "strong" in kinds else ("weak" if "weak" in kinds else "meeting")
        self.fwd = any(e["src"] == a.nid for e in edges)        # a -> b
        self.back = any(e["src"] == b.nid for e in edges)       # b -> a
        self.highlight = False
        self.focus_color: QColor | None = None     # set while hovering a node: link takes that node's color
        self.setZValue(1)
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setToolTip(self.describe())

    def describe(self) -> str:
        out = []
        for e in self.edges:
            s = self.canvas.label_of(e["src"])
            d = self.canvas.label_of(e["dst"])
            line = f"{s} → {edge_verb(e)} → {d}"
            if e.get("why"):
                line += f"\nWhy: {e['why']}"
            out.append(line)
        return "\n\n".join(out)

    def boundingRect(self):
        pa, pb = self.a.pos(), self.b.pos()
        return QRectF(pa, pb).normalized().adjusted(-self.MARGIN, -self.MARGIN, self.MARGIN, self.MARGIN)

    def shape(self):
        path = QPainterPath(self.a.pos())
        path.lineTo(self.b.pos())
        from PySide6.QtGui import QPainterPathStroker
        st = QPainterPathStroker()
        st.setWidth(8 / max(self.canvas.zoom, 0.2))
        return st.createStroke(path)

    def _end(self, tgt: NodeItem, src: NodeItem, zoom: float) -> QPointF:
        """Point where a line from src stops at tgt: the node edge, or below its label if coming from below."""
        v = QLineF(tgt.pos(), src.pos())
        if v.length() == 0:
            return tgt.pos()
        stop = tgt.r + 1.5 / zoom
        dx, dy = v.dx(), v.dy()
        if dy > 0 and abs(dx) < dy * 0.8:            # arriving from straight below: keep clear of the label
            stop = tgt.r + (tk.g("label.gap") + tk.g("label.px") + 4) / zoom
        v.setLength(min(stop, v.length() / 2))
        return v.p2()

    def paint(self, p: QPainter, _o, _w=None):
        s = self.canvas.settings
        zoom = max(self.canvas.zoom, 0.05)
        alpha = {"strong": tk.g("edge.alpha.strong"), "weak": tk.g("edge.alpha.weak"),
                 "meeting": tk.g("edge.alpha.meeting")}[self.kind]
        color = text_color(alpha)
        if s.get("color_by", "group") != "none":
            if self.kind == "meeting":
                color = QColor(theme.qcolor("accent"))
                color.setAlphaF(alpha)
            else:                                      # tint with the colour of the node it starts from
                src = self.a if self.fwd or not self.back else self.b
                color = QColor(src.color)
                color.setAlphaF(tk.g("edge.alpha.tint.strong" if self.kind == "strong" else "edge.alpha.tint.weak"))
        if self.highlight or self.isSelected():
            color = QColor(theme.qcolor("accent"))
        elif self.focus_color is not None:
            color = QColor(self.focus_color)
        width = max(tk.g("edge.min_w"), min(tk.g("edge.max_w"), tk.g("edge.width") * s["link_scale"]))
        if self.kind == "strong":
            width = min(tk.g("edge.max_w"), width * 1.15)
        pen = QPen(color, width)
        pen.setCosmetic(True)
        pen.setCapStyle(Qt.RoundCap)
        if self.kind == "meeting":
            pen.setDashPattern([3, 3])
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(pen)
        arrows = s["arrows"] and self.kind != "meeting"
        pa = self._end(self.a, self.b, zoom) if (arrows and self.back) else self.a.pos()
        pb = self._end(self.b, self.a, zoom) if (arrows and self.fwd) else self.b.pos()
        p.drawLine(pa, pb)
        if arrows:
            size = tk.g("arrow.size") / zoom
            for tip, frm, on in ((pb, pa, self.fwd), (pa, pb, self.back)):
                if not on:
                    continue
                ang = math.atan2(tip.y() - frm.y(), tip.x() - frm.x())
                for sgn in (-1, 1):
                    a2 = ang + math.pi - sgn * 0.5
                    p.drawLine(tip, QPointF(tip.x() + size * math.cos(a2), tip.y() + size * math.sin(a2)))

    def mousePressEvent(self, ev):
        self.canvas.select_edge(self)
        ev.accept()

    def hoverEnterEvent(self, ev):
        self.canvas.hover_edge(self)

    def hoverLeaveEvent(self, ev):
        self.canvas.hover_edge(None, leaving=self)


# ------------------------------------------------------------------ canvas
def status_word(status: str | None, revision: int = 1) -> str:
    if status == "blocked":
        return "Blocked"
    if status == "archived":
        return "Archived"
    if status in ("proposed", "active", "done", None, "") and revision and int(revision) > 1:
        return "Changed"
    return "Accepted" if status in ("active", "done") else "Proposed"


class NodeCard(QWidget):
    """Floating card beside the selected node: name, status pill, first 2 lines of "What it is", number of
    connections, and Open / Focus buttons. Painted by hand (radius 10, hairline border, no shadow effect) so it can
    fade in over 120 ms without any QGraphicsEffect. It stays until the selection is cleared."""
    open_clicked = Signal()
    focus_clicked = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.node: NodeItem | None = None
        self._alpha = 0.0
        self._hot = ""
        self._btns: dict[str, QRectF] = {}
        self.show_focus = False
        self.hide()

    def show_for(self, node: "NodeItem", connections: int):
        self.node, self.connections = node, connections
        d = node.data
        self.title = d.get("label", "")
        self.status = status_word(d.get("status"), d.get("revision", 1)) if node.kind == "feature" else "Meeting"
        self.summary = (d.get("summary") or "").strip() or ("No description saved yet." if node.kind == "feature"
                                                            else f"Meeting · {d.get('date', '')}")
        self._layout()
        self.show()
        self.raise_()
        if motion.enabled():
            motion.value(self, 0.0, 1.0, "hover", self._set_alpha)
        else:
            self._set_alpha(1.0)

    def _set_alpha(self, a):
        self._alpha = float(a)
        self.update()

    def _wrap(self, text: str, fm: QFontMetrics, width: int, max_lines: int = 2) -> list[str]:
        words, lines, cur = text.split(), [], ""
        for w in words:
            t = (cur + " " + w).strip()
            if fm.horizontalAdvance(t) <= width:
                cur = t
            else:
                lines.append(cur)
                cur = w
                if len(lines) == max_lines:
                    break
        if len(lines) < max_lines and cur:
            lines.append(cur)
        if len(lines) == max_lines and (len(" ".join(lines).split()) < len(words)):
            lines[-1] = fm.elidedText(lines[-1] + " …", Qt.ElideRight, width)
        return lines[:max_lines]

    def _layout(self):
        pad, w = tk.g("card.pad"), tk.g("card.w")
        self.f_title, self.f_body, self.f_meta = theme.font("body"), theme.font("body"), theme.font("meta")
        self.f_title.setWeight(QFont.DemiBold)
        self.f_body.setPixelSize(13)
        fb = QFontMetrics(self.f_body)
        self.lines = self._wrap(self.summary, fb, w - 2 * pad)
        h = pad + 20 + 22 + len(self.lines) * fb.lineSpacing() + 12 + 28 + pad
        self.resize(w, h)

    def paintEvent(self, _):
        if self.node is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setOpacity(self._alpha)
        pad, w = tk.g("card.pad"), self.width()
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(theme.qcolor("separator_solid"), 1))
        p.setBrush(theme.qcolor("surface"))
        p.drawRoundedRect(r, tk.RADIUS["card"], tk.RADIUS["card"])
        y = pad
        p.setFont(self.f_title)
        p.setPen(theme.qcolor("text"))
        p.drawText(QRectF(pad, y, w - 2 * pad, 20), Qt.AlignLeft | Qt.AlignVCenter,
                   QFontMetrics(self.f_title).elidedText(self.title, Qt.ElideRight, w - 2 * pad))
        y += 24
        # status pill
        p.setFont(self.f_meta)
        fm = QFontMetrics(self.f_meta)
        pw = fm.horizontalAdvance(self.status) + 22
        pill = QRectF(pad, y, pw, 18)
        soft = QColor(self.node.color)
        soft.setAlphaF(0.16)
        p.setPen(Qt.NoPen)
        p.setBrush(soft)
        p.drawRoundedRect(pill, 9, 9)
        p.setBrush(self.node.color)
        p.drawEllipse(QPointF(pad + 9, y + 9), 3.5, 3.5)
        p.setPen(theme.qcolor("text"))
        p.drawText(pill.adjusted(16, 0, 0, 0), Qt.AlignLeft | Qt.AlignVCenter, self.status)
        p.setPen(theme.qcolor("secondary"))
        n = self.connections
        p.drawText(QRectF(pad + pw + 8, y, w, 18), Qt.AlignLeft | Qt.AlignVCenter,
                   f"{n} connection{'s' if n != 1 else ''}")
        y += 26
        p.setFont(self.f_body)
        p.setPen(theme.qcolor("secondary"))
        fb = QFontMetrics(self.f_body)
        for ln in self.lines:
            p.drawText(QRectF(pad, y, w - 2 * pad, fb.lineSpacing()), Qt.AlignLeft | Qt.AlignVCenter, ln)
            y += fb.lineSpacing()
        y += 12
        # buttons
        self._btns = {}
        x = pad
        for key, text, primary in (("open", "Open", True), ("focus", "Focus", False)):
            if key == "focus" and not self.show_focus:
                continue
            p.setFont(self.f_meta)
            bw = QFontMetrics(self.f_meta).horizontalAdvance(text) + 24
            br = QRectF(x, y, bw, 26)
            self._btns[key] = br
            hot = self._hot == key
            if primary:
                p.setBrush(theme.qcolor("accent_hover" if hot else "accent"))
                p.setPen(Qt.NoPen)
            else:
                p.setBrush(theme.qcolor("control_hover" if hot else "control"))
                p.setPen(Qt.NoPen)
            p.drawRoundedRect(br, tk.RADIUS["control"], tk.RADIUS["control"])
            p.setPen(theme.qcolor("on_accent" if primary else "text"))
            p.drawText(br, Qt.AlignCenter, text)
            x += bw + 8

    def _key_at(self, pos) -> str:
        for k, r in self._btns.items():
            if r.contains(pos):
                return k
        return ""

    def mouseMoveEvent(self, ev):
        k = self._key_at(ev.position())
        if k != self._hot:
            self._hot = k
            self.setCursor(Qt.PointingHandCursor if k else Qt.ArrowCursor)
            self.update()

    def leaveEvent(self, _):
        if self._hot:
            self._hot = ""
            self.update()

    def mousePressEvent(self, ev):
        k = self._key_at(ev.position())
        if k == "open":
            self.open_clicked.emit()
        elif k == "focus":
            self.focus_clicked.emit()
        ev.accept()


class MapCanvas(QGraphicsView):
    open_node = Signal(str, str)
    edge_selected = Signal(object)     # EdgeItem or None
    zoom_changed = Signal(float)
    node_moved = Signal(str, QPointF)  # dragged by hand -> pinned
    node_unpinned = Signal(str)
    escape = Signal()                  # the Map decides what Esc closes first
    local_requested = Signal(str)      # L / "Focus": show this node's neighbourhood

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene_ = QGraphicsScene(self)
        self.scene_.setItemIndexMethod(QGraphicsScene.BspTreeIndex)
        self.setScene(self.scene_)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setViewportUpdateMode(QGraphicsView.MinimalViewportUpdate)
        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.NoFrame)
        self.setDragMode(QGraphicsView.ScrollHandDrag)       # open hand / closed hand while panning
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setFocusPolicy(Qt.StrongFocus)
        self.settings = {"node_scale": 1.0, "link_scale": 1.0, "arrows": False, "show_meetings": False,
                         "label_hide": tk.g("label.zoom.hide"), "label_full": tk.g("label.zoom.full")}
        self.zoom = 1.0
        self.nodes: dict[str, NodeItem] = {}
        self.labels: dict[str, LabelItem] = {}
        self.edges: list[EdgeItem] = []
        self.top_ids: set[str] = set()
        self.selected: NodeItem | None = None
        self._labels_alpha: dict[str, float] = {}
        # one animation object per kind; they only run right after user input and stop by themselves
        self._zoom_target = 1.0
        self._zoom_anchor = (QPointF(), None)
        self._zoom_anim = QVariantAnimation(self)
        self._zoom_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._zoom_anim.valueChanged.connect(self._zoom_step)
        self._view_anim = QVariantAnimation(self)
        self._view_anim.setStartValue(0.0)
        self._view_anim.setEndValue(1.0)
        self._view_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._view_anim.valueChanged.connect(self._view_step)
        self._glide_anim = QVariantAnimation(self)
        self._glide_anim.setStartValue(0.0)
        self._glide_anim.setEndValue(1.0)
        self._glide_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._glide_anim.valueChanged.connect(self._glide_step)
        self._glide_from: dict = {}
        self._glide_to: dict = {}
        self._fade: set = set()
        # hover / selection focus: ONE shared animation drives every item's opacity
        self.adj: dict[str, set[str]] = {}
        self.node_edges: dict[str, list] = {}
        self.pinned: set[str] = set()
        self._hover: NodeItem | None = None
        self._hover_e: EdgeItem | None = None
        self._focus_nodes: set[str] | None = None
        self._focus_done = None
        self._kbd: NodeItem | None = None             # node focused with Tab (keyboard)
        self.search_ids: set[str] | None = None       # Ctrl+F matches
        self.vis_nodes: set[str] | None = None        # None = everything visible (filters / local graph)
        self.vis_edges: set | None = None
        self.meeting_color: dict[str, str] = {}       # feature id -> color, for "Color by first meeting"
        self.groups: dict[str, int] = {}              # feature id -> topic group (colour), from the worker
        self._focus_from: dict = {}
        self._focus_to: dict = {}
        self._focus_anim = QVariantAnimation(self)
        self._focus_anim.setStartValue(0.0)
        self._focus_anim.setEndValue(1.0)
        self._focus_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._focus_anim.valueChanged.connect(self._focus_step)
        self.card = NodeCard(self.viewport())
        self.card.open_clicked.connect(lambda: self.selected and self.open_node.emit(self.selected.kind,
                                                                                       self.selected.nid))
        self.horizontalScrollBar().valueChanged.connect(self._place_card)
        self.verticalScrollBar().valueChanged.connect(self._place_card)
        self.zoom_changed.connect(self._place_card)

    # ----- data -----
    def label_of(self, nid: str) -> str:
        n = self.nodes.get(nid)
        return n.data.get("label", "?") if n else "?"

    def build(self, nodes: list[dict], edges: list[dict], positions: dict[str, QPointF]):
        """Create items once for this graph (UI thread only creates items and sets positions)."""
        keep_center = self.mapToScene(self.viewport().rect().center()) if self.nodes else None
        self._glide_anim.stop()
        sel_id = self.selected.nid if self.selected is not None else None
        self._focus_anim.stop()
        self.scene_.clear()
        self.nodes, self.labels, self.edges, self.selected = {}, {}, [], None
        self._hover = self._hover_e = None
        self._kbd = None
        self._focus_nodes = None
        self._labels_alpha = {}
        self.adj, self.node_edges = {}, {}
        deg: dict[str, int] = {}
        pairs: dict[tuple, list] = {}
        for e in edges:
            if e["src"] not in positions or e["dst"] not in positions:
                continue
            k = tuple(sorted((e["src"], e["dst"])))
            pairs.setdefault(k, []).append(e)
        for (a, b), es in pairs.items():
            if any(not e["type"].startswith("meeting_") for e in es):
                deg[a] = deg.get(a, 0) + 1
                deg[b] = deg.get(b, 0) + 1
        for n in nodes:
            if n["id"] not in positions:
                continue
            it = NodeItem(self, n, deg.get(n["id"], 0))
            it.pinned = n["id"] in self.pinned
            it.setPos(positions[n["id"]])
            self.scene_.addItem(it)
            self.nodes[n["id"]] = it
            lab = LabelItem(it)
            lab.place()
            self.scene_.addItem(lab)
            self.labels[n["id"]] = lab
        for (a, b), es in pairs.items():
            ed = EdgeItem(self, self.nodes[es[0]["src"]], self.nodes[es[0]["dst"]], es)
            self.scene_.addItem(ed)
            self.edges.append(ed)
            self.adj.setdefault(a, set()).add(b)
            self.adj.setdefault(b, set()).add(a)
            self.node_edges.setdefault(a, []).append(ed)
            self.node_edges.setdefault(b, []).append(ed)
        feats = sorted((n for n in self.nodes.values() if n.kind == "feature"), key=lambda n: -n.degree)
        self.top_ids = {n.nid for n in feats[: tk.g("label.top_n")]} | \
            {n.nid for n in self.nodes.values() if n.kind == "meeting"}         # transcript names always readable
        r = self.scene_.itemsBoundingRect()
        self.scene_.setSceneRect(r.adjusted(-4000, -4000, 4000, 4000))   # free panning, no scrollbars
        self._apply_label_visibility()
        if keep_center is not None:
            self.centerOn(keep_center)                                     # data changed: the view doesn't jump
        if sel_id in self.nodes:
            self.select_node(self.nodes[sel_id], animate=False)            # keep the selection across reloads
        else:
            self.card.hide()

    def node_positions(self) -> dict[str, QPointF]:
        return {nid: QPointF(n.pos()) for nid, n in self.nodes.items()}

    def glide(self, final: dict[str, QPointF], fade_in: set[str]):
        """New layout: every node glides from where it is to its new spot in one 300 ms tween; new nodes fade in."""
        self._glide_anim.stop()
        self._glide_from = self.node_positions()
        self._glide_to = final
        self._fade = set(fade_in)
        if not motion.enabled():
            self._glide_step(1.0)
            return
        self._glide_anim.setDuration(tk.g("anim.glide_ms"))
        self._glide_anim.start()

    def _glide_step(self, t):
        t = float(t)
        for e in self.edges:
            e.prepareGeometryChange()
        for nid, n in self.nodes.items():
            a = self._glide_from.get(nid, n.pos())
            b = self._glide_to.get(nid, a)
            n.setPos(a + (b - a) * t)
            if nid in self.labels:
                self.labels[nid].place()
            if nid in self._fade:
                n.setOpacity(t)
                if nid in self.labels:
                    self.labels[nid].setOpacity(min(t, self._labels_alpha.get(nid, 1.0)))
        self._place_card()
        if t >= 1.0:
            self._fade = set()
            r = self.scene_.itemsBoundingRect()
            self.scene_.setSceneRect(r.adjusted(-4000, -4000, 4000, 4000))

    def restyle(self):
        """Theme or display settings changed: recolor everything once."""
        for n in self.nodes.values():
            n.update_style()
        for lab in self.labels.values():
            lab.place()
            lab.update()
        for e in self.edges:
            e.update()
        self.viewport().update()

    # ----- look -----
    def drawBackground(self, p: QPainter, rect: QRectF):
        p.fillRect(rect, QColor(tk.g("bg")))

    def _ramp(self) -> float:
        lo, hi = self.settings["label_hide"], self.settings["label_full"]
        return 0.0 if self.zoom <= lo else 1.0 if self.zoom >= hi else (self.zoom - lo) / (hi - lo)

    def _label_target(self, nid: str, nodes_f: set | None) -> float:
        """Label opacity = zoom rule x focus rule. Top-N, selected and focused labels ignore the zoom rule."""
        forced = nid in self.top_ids or (self.selected is not None and self.selected.nid == nid) \
            or (nodes_f is not None and nid in nodes_f)
        base = 1.0 if forced else self._ramp()
        return base * (1.0 if nodes_f is None or nid in nodes_f else tk.g("dim.alpha"))

    def _apply_label_visibility(self):
        """Zoom changed: set label opacities directly from zoom (no timer, no animation)."""
        for nid, lab in self.labels.items():
            a = 0.0 if self.is_hidden(nid) else self._label_target(nid, self._focus_nodes)
            if self._focus_anim.state() == QVariantAnimation.Running and lab in self._focus_to:
                self._focus_to[lab] = a
                continue
            if self._labels_alpha.get(nid) != a:          # touch only labels whose opacity changes
                self._labels_alpha[nid] = a
                lab.setOpacity(a)
                lab.setVisible(a > 0.01)

    # ----- focus (hover / selection) -----
    def _focus_spec(self):
        """(focused node ids, focused edges, centre node) or (None, None, None) for no focus.
        Priority: hovered node > hovered link > keyboard-focused node > search matches > selection."""
        if self._hover is not None:
            c = self._hover
        elif self._hover_e is not None:
            e = self._hover_e
            return {e.a.nid, e.b.nid}, {e}, None
        elif self._kbd is not None:
            c = self._kbd
        elif self.search_ids is not None:
            ids = self.search_ids
            return set(ids), {e for e in self.edges if e.a.nid in ids and e.b.nid in ids}, None
        elif self.selected is not None:
            c = self.selected
        else:
            return None, None, None
        return {c.nid} | self.adj.get(c.nid, set()), set(self.node_edges.get(c.nid, [])), c

    def is_hidden(self, nid: str) -> bool:
        return self.vis_nodes is not None and nid not in self.vis_nodes

    def refocus(self, animate: bool = True, ms: int | None = None, on_done=None):
        """One pass sets every item's target opacity: hidden (filters / local graph) -> 0, outside the focus
        -> ~18 %, otherwise full. A single shared animation fades to the targets (120 ms for hover, 150 ms for
        filters) and retargets if it is already running. Items that end at 0 are hidden."""
        nodes_f, edges_f, centre = self._focus_spec()
        self._focus_nodes = nodes_f
        dim = tk.g("dim.alpha")
        to = {}
        for nid, n in self.nodes.items():
            hidden = self.is_hidden(nid)
            to[n] = 0.0 if hidden else 1.0 if nodes_f is None or nid in nodes_f else dim
            ring = n is self._hover or n is self.selected or n is self._kbd
            if n.ring != ring or n.kbd != (n is self._kbd):
                n.ring, n.kbd = ring, n is self._kbd
                n.update()
            lab = self.labels[nid]
            to[lab] = 0.0 if hidden else self._label_target(nid, nodes_f)
            lab.set_strong(n is self._hover or n is self.selected or n is self._kbd)
        for e in self.edges:
            hidden = self.vis_edges is not None and e not in self.vis_edges
            to[e] = 0.0 if hidden else 1.0 if edges_f is None or e in edges_f else dim
            col = None
            if centre is not None and e in (edges_f or ()):
                col = QColor(centre.color)
                col.setAlphaF(tk.g("edge.alpha.hover"))
            hl = e is self._hover_e
            if e.focus_color != col or e.highlight != hl:
                e.focus_color, e.highlight = col, hl
                e.update()
        for it, v in to.items():                 # things coming back start transparent, then fade in
            if v > 0.01 and not it.isVisible():
                it.setOpacity(0.0)
                it.setVisible(True)
        self._focus_anim.stop()
        self._focus_from = {it: it.opacity() for it in to}
        self._focus_to = to
        self._focus_done = on_done
        if animate and motion.enabled():
            self._focus_anim.setDuration(ms or tk.g("anim.hover_ms"))
            self._focus_anim.start()
        else:
            self._focus_step(1.0)

    def _focus_step(self, t):
        t = float(t)
        for it, b in self._focus_to.items():
            a = self._focus_from.get(it, b)
            v = a + (b - a) * t
            if abs(it.opacity() - v) > 0.003:
                it.setOpacity(v)
        if t >= 1.0:
            for it, b in self._focus_to.items():
                if b <= 0.01 and it.isVisible():
                    it.setVisible(False)
            for nid, lab in self.labels.items():
                self._labels_alpha[nid] = self._focus_to.get(lab, lab.opacity())
            done, self._focus_done = self._focus_done, None
            if done:
                done()

    def hover_node(self, n, leaving=None):
        if n is None and leaving is not self._hover:
            return                                 # a stale "leave" after we already moved to another node
        if n is self._hover:
            return
        self._hover = n
        self.refocus()

    def hover_edge(self, e, leaving=None):
        if e is None and leaving is not self._hover_e:
            return
        if e is self._hover_e:
            return
        self._hover_e = e
        if self._hover is None:
            self.refocus()

    # ----- dragging / pins -----
    def move_node(self, n: NodeItem, pos: QPointF):
        """Only this node and its links move; nothing else reacts."""
        for e in self.node_edges.get(n.nid, []):
            e.prepareGeometryChange()
        n.setPos(pos)
        self.labels[n.nid].place()
        if self.selected is n:
            self._place_card()

    def end_drag(self, n: NodeItem):
        n.pinned = True
        self.pinned.add(n.nid)
        n.update()
        self.node_moved.emit(n.nid, QPointF(n.pos()))

    def unpin(self, n: NodeItem):
        n.pinned = False
        self.pinned.discard(n.nid)
        n.update()
        self.node_unpinned.emit(n.nid)

    # ----- card -----
    def _place_card(self, *_):
        n = self.selected
        if n is None or not self.card.isVisible():
            return
        c = self.mapFromScene(n.pos())
        lab = self.labels.get(n.nid)
        half_label = lab.w / 2 if lab is not None and lab.isVisible() else 0
        off = max(n.r * self.zoom, half_label) + tk.g("card.offset")      # clear of the node AND its label
        x, y = c.x() + off, c.y() - self.card.height() / 2
        if x + self.card.width() > self.viewport().width() - 8:
            x = c.x() - off - self.card.width()          # no room on the right: show it on the left
        x = max(8, min(x, self.viewport().width() - self.card.width() - 8))
        y = max(8, min(y, self.viewport().height() - self.card.height() - 8))
        self.card.move(int(x), int(y))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._place_card()

    # ----- zoom / pan -----
    def _set_zoom(self, z: float, anchor_scene: QPointF | None = None, anchor_view=None):
        z = max(tk.g("zoom.min"), min(tk.g("zoom.max"), z))
        if anchor_scene is None:
            anchor_view = self.viewport().rect().center()
            anchor_scene = self.mapToScene(anchor_view)
        self.zoom = z
        self.setTransform(QTransform.fromScale(z, z))
        # keep the anchor point under the cursor
        now = self.mapFromScene(anchor_scene)
        d = now - anchor_view
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + d.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() + d.y())
        self._apply_label_visibility()
        self.zoom_changed.emit(z)

    def zoom_by(self, factor: float, view_pos=None, ms: int | None = None):
        """Ease toward a new zoom; consecutive steps retarget the same animation instead of stacking."""
        view_pos = view_pos if view_pos is not None else self.viewport().rect().center()
        anchor = self.mapToScene(view_pos)
        running = self._zoom_anim.state() == QVariantAnimation.Running
        base = self._zoom_target if running else self.zoom
        self._zoom_target = max(tk.g("zoom.min"), min(tk.g("zoom.max"), base * factor))
        self._zoom_anim.stop()
        dur = (ms or tk.g("anim.wheel_ms")) if motion.enabled() else 0
        if not dur:
            self._set_zoom(self._zoom_target, anchor, view_pos)
            return
        self._zoom_anchor = (anchor, view_pos)
        self._zoom_anim.setStartValue(float(self.zoom))
        self._zoom_anim.setEndValue(float(self._zoom_target))
        self._zoom_anim.setDuration(dur)
        self._zoom_anim.start()

    def _zoom_step(self, z):
        anchor, view_pos = self._zoom_anchor
        self._set_zoom(float(z), anchor, view_pos)

    def wheelEvent(self, ev):
        steps = ev.angleDelta().y() / 120.0
        if not steps:
            return
        self.zoom_by(tk.g("zoom.step") ** steps, ev.position().toPoint())
        ev.accept()

    def visible_rect(self) -> QRectF:
        r = QRectF()
        for n in self.nodes.values():
            if n.isVisible() and not self.is_hidden(n.nid):
                r = r.united(n.sceneBoundingRect())
        return r

    def fit(self, animate: bool = True):
        r = self.visible_rect()
        if r.isNull():
            return
        pad = tk.g("fit.pad")
        vw, vh = max(self.viewport().width() - 2 * pad, 50), max(self.viewport().height() - 2 * pad - 20, 50)
        z = min(vw / max(r.width(), 1), vh / max(r.height(), 1), tk.g("zoom.fit_max"))
        self.animate_view(r.center(), z, tk.g("anim.fit_ms") if animate else 0)

    def animate_view(self, center: QPointF, zoom: float, ms: int):
        """Glide center + zoom together (one animation object; a new call retargets it)."""
        zoom = max(tk.g("zoom.min"), min(tk.g("zoom.max"), zoom))
        self._view_anim.stop()
        self._zoom_anim.stop()
        self._view_from = (self.mapToScene(self.viewport().rect().center()), self.zoom)
        self._view_to = (QPointF(center), zoom)
        if not ms or not motion.enabled():
            self._view_step(1.0)
            return
        self._view_anim.setDuration(ms)
        self._view_anim.start()

    def _view_step(self, t):
        t = float(t)
        (c0, z0), (c1, z1) = self._view_from, self._view_to
        self.zoom = max(tk.g("zoom.min"), min(tk.g("zoom.max"), z0 + (z1 - z0) * t))
        self.setTransform(QTransform.fromScale(self.zoom, self.zoom))
        self.centerOn(c0 + (c1 - c0) * t)
        self._apply_label_visibility()
        self.zoom_changed.emit(self.zoom)

    def mouseDoubleClickEvent(self, ev):
        if self.itemAt(ev.position().toPoint()) is None:      # empty space: one zoom step toward the point
            self.zoom_by(tk.g("zoom.step") ** 2, ev.position().toPoint(), tk.g("anim.fit_ms"))
            ev.accept()
            return
        super().mouseDoubleClickEvent(ev)

    def mousePressEvent(self, ev):
        self.clear_kbd()
        if self.itemAt(ev.position().toPoint()) is None and ev.button() == Qt.LeftButton:
            self.clear_selection()
        super().mousePressEvent(ev)

    def keyPressEvent(self, ev):
        k = ev.key()
        if k == Qt.Key_F:
            self.fit()
        elif k in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom_by(tk.g("zoom.step"))
        elif k in (Qt.Key_Minus, Qt.Key_Underscore):
            self.zoom_by(1 / tk.g("zoom.step"))
        elif k in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            d = 60
            dx = {Qt.Key_Left: -d, Qt.Key_Right: d}.get(k, 0)
            dy = {Qt.Key_Up: -d, Qt.Key_Down: d}.get(k, 0)
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + dx)
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + dy)
        elif k == Qt.Key_Escape:
            self.escape.emit()
        elif k in (Qt.Key_Return, Qt.Key_Enter):
            n = self._kbd or self.selected
            if n is not None:
                self.open_node.emit(n.kind, n.nid)
        elif k == Qt.Key_L:
            n = self._kbd or self.selected
            if n is not None:
                self.local_requested.emit(n.nid)
        elif k == Qt.Key_Space and self._kbd is not None:
            self.select_node(self._kbd)
        else:
            super().keyPressEvent(ev)

    # ----- keyboard: Tab walks the nodes, biggest first -----
    def focusNextPrevChild(self, forward: bool) -> bool:
        if not self.hasFocus():                  # e.g. a child box closing: normal focus handling
            return super().focusNextPrevChild(forward)
        order = sorted((n for n in self.nodes.values() if n.isVisible() and not self.is_hidden(n.nid)),
                       key=lambda n: (-n.degree, n.data.get("label", "").lower()))
        if not order:
            return super().focusNextPrevChild(forward)
        if self._kbd in order:
            i = order.index(self._kbd) + (1 if forward else -1)
        else:
            i = 0 if forward else len(order) - 1
        if not 0 <= i < len(order):              # past the last node: let Tab leave the map
            self.clear_kbd()
            return super().focusNextPrevChild(forward)
        self._kbd = order[i]
        self.refocus()
        vr = self.mapToScene(self.viewport().rect()).boundingRect().adjusted(40, 40, -40, -40)
        if not vr.contains(self._kbd.pos()):
            self.animate_view(self._kbd.pos(), self.zoom, tk.g("anim.fit_ms"))
        return True

    def clear_kbd(self):
        if self._kbd is not None:
            self._kbd = None
            self.refocus()

    def focusOutEvent(self, ev):
        super().focusOutEvent(ev)
        if ev.reason() != Qt.PopupFocusReason:
            self.clear_kbd()

    # ----- selection -----
    def select_node(self, n: NodeItem, animate: bool = True):
        for e in self.scene_.selectedItems():
            e.setSelected(False)
            e.update()
        self.selected = n
        self.card.show_for(n, len(self.adj.get(n.nid, ())))
        self._place_card()
        self.refocus(animate)
        self._apply_label_visibility()

    def select_edge(self, e: EdgeItem):
        self.clear_selection()
        e.setSelected(True)
        e.update()
        self.edge_selected.emit(e)

    def clear_selection(self):
        had = self.selected is not None
        self.selected = None
        self.card.hide()
        for e in self.scene_.selectedItems():
            e.setSelected(False)
            e.update()
        if had:
            self.refocus()
            self._apply_label_visibility()
        self.edge_selected.emit(None)

    def center_on_edge(self, e: EdgeItem):
        self.select_edge(e)
        mid = (e.a.pos() + e.b.pos()) / 2
        self.animate_view(mid, max(self.zoom, 1.0), tk.g("anim.fit_ms"))


# ------------------------------------------------------------------ relationship rows
class WrapDelegate(QStyledItemDelegate):
    """Rows that wrap onto as many lines as needed (never "…", no horizontal scrolling). The height is
    measured and the text drawn with the same rectangle, so they always agree."""
    PAD = 8

    def _rect(self, width: int) -> QRect:
        return QRect(0, 0, max(60, width - 2 * self.PAD - 4), 10_000)

    def sizeHint(self, option, index):
        view = self.parent()
        w = view.viewport().width() if view is not None else option.rect.width()
        fm = QFontMetrics(theme.font("body"))
        h = fm.boundingRect(self._rect(w), Qt.TextWordWrap, index.data() or "").height()
        return QSize(w, h + 2 * self.PAD - 2)

    def paint(self, p: QPainter, option, index):
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(option.rect).adjusted(2, 1, -2, -1)
        if option.state & QStyle.State_Selected:
            p.setPen(Qt.NoPen)
            p.setBrush(theme.qcolor("accent_soft"))
            p.drawRoundedRect(r, tk.RADIUS["control"], tk.RADIUS["control"])
        elif option.state & QStyle.State_MouseOver:
            p.setPen(Qt.NoPen)
            p.setBrush(theme.qcolor("hover"))
            p.drawRoundedRect(r, tk.RADIUS["control"], tk.RADIUS["control"])
        p.setFont(theme.font("body"))
        p.setPen(theme.qcolor("text"))
        tr = option.rect.adjusted(self.PAD + 2, self.PAD - 1, -self.PAD - 2, -self.PAD + 1)
        p.drawText(tr, Qt.TextWordWrap | Qt.AlignLeft | Qt.AlignTop, index.data() or "")
        p.restore()


# ------------------------------------------------------------------ the Map page
class MapView(QWidget):
    """Canvas edge to edge. Floating: Fit + gear (top-right), search and focus chip (top-left), legend "?"
    (bottom-left), controls panel and relationship list (slide in from the right), empty state."""
    open_node = Signal(str, str)
    go_meetings = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._graph = {"nodes": [], "edges": []}
        self._override = {"nodes": [], "edges": []}
        self._fitted = False
        self._store = self._root = None
        self._pid: str | None = None
        self._settings_for = None
        self._shown_sig = None
        self._token = 0
        self._jobs: set = set()
        self._busy = False
        self._pending = False
        self._pinned: set = set()
        self._row_edges: list = []
        self.local: tuple[str, int] | None = None          # (node id, depth) while "Focus" is on
        self._matches: list[str] = []
        self._match_i = -1
        self.st = dict(DEFAULTS)
        self.st["label_full"] = self.st["label_hide"] + 0.4

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.canvas = MapCanvas(self)
        self.canvas.settings = self.st                       # one settings dict shared with the canvas
        root.addWidget(self.canvas, 1)

        # relationship list: slides in over the right edge (closed by default)
        self.side = QWidget(self)
        self.side.setObjectName("mapSide")
        self.side.setFixedWidth(300)
        sl = QVBoxLayout(self.side)
        sl.setContentsMargins(tk.SPACE[12], tk.SPACE[12], tk.SPACE[12], tk.SPACE[12])
        cap = QLabel("Relationships")
        cap.setObjectName("caption")
        self.reason = QLabel("Hover a row to find it on the map; click to go there.")
        self.reason.setObjectName("note")
        self.reason.setWordWrap(True)
        self.rel_list = QListWidget()
        self.rel_list.setWordWrap(True)
        self.rel_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.rel_list.setTextElideMode(Qt.ElideNone)
        self.rel_list.setResizeMode(QListWidget.Adjust)
        self.rel_list.setUniformItemSizes(False)
        self.rel_list.setItemDelegate(WrapDelegate(self.rel_list))
        self.rel_list.setMouseTracking(True)
        self.rel_list.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        sl.addWidget(cap)
        sl.addWidget(self.reason)
        sl.addWidget(self.rel_list, 1)
        self.side.hide()

        # floating controls
        self.bar = QWidget(self.canvas)
        bl = QHBoxLayout(self.bar)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(tk.SPACE[4])
        self.note = QLabel("")
        self.note.setObjectName("meta")
        self.note.hide()
        self.b_fit = self._tool("Fit", "Fit the map to the window (F)")
        self.b_gear = self._tool("⚙", "Filters, display and layout", checkable=True)
        self.b_gear.setFixedWidth(34)
        bl.addWidget(self.note)
        bl.addWidget(self.b_fit)
        bl.addWidget(self.b_gear)
        self.b_help = self._tool("?", "What the colors and shapes mean")
        self.b_help.setParent(self.canvas)
        self.b_help.setFixedWidth(30)
        self.panel = MapPanel(self)
        self.panel.hide()
        self.search = SearchBox(self.canvas)
        self.chip = FocusChip(self.canvas)

        # empty state
        self.empty = QWidget(self.canvas)
        el = QVBoxLayout(self.empty)
        msg = QLabel("No features yet. Save a meeting to build the map.")
        msg.setObjectName("muted")
        msg.setAlignment(Qt.AlignCenter)
        b_go = QPushButton("Go to Meetings")
        b_go.setObjectName("quiet")
        el.addStretch(1)
        el.addWidget(msg)
        el.addWidget(b_go, 0, Qt.AlignHCenter)
        el.addStretch(1)
        self.empty.hide()

        # one animation each for the two sliding panels; a single-shot timer debounces search (150 ms)
        self._panel_anim = self._slide_anim(self.panel)
        self._side_anim = self._slide_anim(self.side)
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(tk.g("anim.filter_ms"))
        self._search_timer.timeout.connect(self._run_search)
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(400)
        self._save_timer.timeout.connect(self._save_settings)

        c = self.canvas
        c.open_node.connect(self.open_node)
        c.edge_selected.connect(self._on_edge)
        c.node_moved.connect(self._on_moved)
        c.node_unpinned.connect(self._on_unpinned)
        c.escape.connect(self._on_escape)
        c.local_requested.connect(lambda nid: self.enter_local(nid))
        c.card.show_focus = True
        c.card.focus_clicked.connect(lambda: c.selected and self.enter_local(c.selected.nid))
        b_go.clicked.connect(self.go_meetings)
        self.b_fit.clicked.connect(lambda: c.fit())
        self.b_gear.toggled.connect(self._toggle_panel)
        self.b_help.clicked.connect(self._legend)
        self.panel.changed.connect(self._on_setting)
        self.panel.rearrange.connect(self.rearrange)
        self.panel.reset.connect(self._reset)
        self.search.text_changed.connect(lambda _t: self._search_timer.start())
        self.search.step.connect(self._search_step)
        self.search.closed.connect(self.close_search)
        self.chip.depth_changed.connect(lambda d: self.local and self.enter_local(self.local[0], d))
        self.chip.closed.connect(self.exit_local)
        self.rel_list.currentRowChanged.connect(self._on_row)
        self.rel_list.itemEntered.connect(self._on_row_hover)
        self.rel_list.viewport().installEventFilter(self)
        sc = QShortcut(QKeySequence.Find, self, activated=self.open_search)
        sc.setContext(Qt.WidgetWithChildrenShortcut)
        theme.signals.changed.connect(lambda *_: self._restyle())
        self._restyle()

    # ----- small builders -----
    def _tool(self, text, tip, checkable=False) -> QPushButton:
        b = QPushButton(text)
        b.setObjectName("mapTool")
        b.setToolTip(tip)
        b.setAccessibleName(tip)
        b.setCheckable(checkable)
        b.setCursor(Qt.PointingHandCursor)
        b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        return b

    def _slide_anim(self, w: QWidget) -> QVariantAnimation:
        a = QVariantAnimation(self)
        a.setEasingCurve(QEasingCurve.OutCubic)
        a.setDuration(tk.g("anim.panel_ms"))
        a.valueChanged.connect(lambda v, w=w: w.move(int(v.x()), int(v.y())))
        a.finished.connect(lambda w=w: (w.hide() if getattr(w, "_closing", False) else None,
                                        self._place_overlays()))
        return a

    def _restyle(self):
        self.setStyleSheet(map_qss())
        self.canvas.restyle()

    # ----- data (loaded and laid out in a background thread) -----
    def set_context(self, store, workspace_root):
        """Where to read graph data and keep the per-project position cache (called once by the main window)."""
        self._store, self._root = store, workspace_root

    def load_project(self, project_id: str | None):
        """Show a project's map: saved positions draw at once; layout runs only if something is new."""
        if project_id != self._pid:
            self._pid, self._fitted, self._settings_for = project_id, False, None
            self._shown_sig = None
            self.local = None
            self.chip.hide()
            self.close_search()
        if not project_id:
            self.show_graph({"nodes": [], "edges": []})
            return
        self._start_job("load")

    def show_graph(self, g: dict):
        """Show a graph given directly (no cache): used for empty projects and tests."""
        self._pid = None
        self._override = g or {"nodes": [], "edges": []}
        self._start_job("load")

    def rearrange(self):
        if self._busy:
            return
        self._start_job("rearrange")

    def set_busy(self, busy: bool):
        """The main window says whether the AI (Ask, transcription, notes) is working. Layout waits for it."""
        busy = bool(busy)
        if busy == self._busy:
            return
        self._busy = busy
        self.panel.set_busy(busy)
        if not busy and self._pending:
            self._pending = False
            self._start_job("load")

    def _set_note(self, text: str):
        self.note.setText(text)
        self.note.setVisible(bool(text))
        self.panel.set_arranging(text)
        self._place_overlays()

    def _start_job(self, mode: str):
        self._token += 1
        req = {"token": self._token, "mode": mode, "allow_layout": not self._busy,
               "show_meetings": self.st["show_meetings"], "spacing": self.st["spacing"], "spread": self.st["spread"]}
        if self._pid:
            req.update(store=self._store, project_id=self._pid, workspace_root=self._root)
        else:
            req.update(graph=self._override)
        if mode == "rearrange":
            self._set_note("Arranging…")
        job = MapJob(req, self)
        job.done.connect(self._on_job)
        job.finished.connect(lambda j=job: (self._jobs.discard(j), j.deleteLater()))
        self._jobs.add(job)
        job.start(QThread.LowestPriority)

    def _on_job(self, res: dict):
        if res.get("token") != self._token:
            return                                   # an older request; a newer one is on its way
        self._set_note("")
        if res.get("error"):
            return                                   # keep what is shown; never a traceback
        if res.get("project_id") and self._settings_for != res["project_id"]:
            self._settings_for = res["project_id"]
            if self._apply_settings(res.get("settings") or {}):
                self._start_job("load")              # saved setting says meetings on: reload with them
                return
        self._graph = res["graph"]
        self.canvas.groups = res.get("groups") or {}
        self._pinned = set(res.get("pinned") or [])
        self.canvas.pinned = set(self._pinned)
        self._pending = res.get("pending", False)
        if self._pending:
            self._set_note("Arranging when the AI is free…")
        nodes, edges = res["nodes"], res["edges"]
        pos = {k: QPointF(v[0], v[1]) for k, v in res["positions"].items()}
        has_features = any(n["type"] == "feature" for n in self._graph["nodes"])
        self.empty.setVisible(not has_features)
        self.bar.setVisible(has_features)
        self.b_help.setVisible(has_features)
        self._meeting_colors()
        sig = (tuple(sorted((k, round(p.x(), 1), round(p.y(), 1)) for k, p in pos.items())),
               tuple(sorted((e["src"], e["dst"], e["type"], e.get("why") or "") for e in edges)),
               tuple(sorted((n["id"], n.get("label"), n.get("status"), n.get("revision")) for n in nodes)),
               theme.c("text"))
        if sig == self._shown_sig:
            return                                   # nothing changed: no rebuild, no redraw
        self._shown_sig = sig
        old = self.canvas.node_positions()
        if old and self.isVisible() and self._fitted:
            start = {k: old.get(k, p) for k, p in pos.items()}
            self.canvas.build(nodes, edges, start)
            fade = {k for k in pos if k not in old}
            for k in fade:
                self.canvas.nodes[k].setOpacity(0.0)
                self.canvas.labels[k].setOpacity(0.0)
            self.canvas.glide(pos, fade)
        else:
            self.canvas.build(nodes, edges, pos)
        if self.local and self.local[0] not in self.canvas.nodes:
            self.local = None
            self.chip.hide()
        self._refresh_panel()
        self.apply_filters(animate=False)
        if not self._fitted:
            self._fitted = True
            self._need_fit = True
            self._fit_if_ready()

    # ----- settings (persisted per project) -----
    def _apply_settings(self, saved: dict) -> bool:
        """Load saved settings. Returns True if the graph must be reloaded (meetings visibility changed)."""
        was_meet = self.st["show_meetings"]
        self.st.update({k: v for k, v in DEFAULTS.items()})
        if saved.get("settings_v") != SETTINGS_V:
            # saved by an older version, whose defaults hid transcripts and coloured by status: use the new ones
            saved = {k: v for k, v in saved.items() if k not in ("show_meetings", "color_by")}
        for k, v in saved.items():
            if k in DEFAULTS and type(v) is type(DEFAULTS[k]) or (k in DEFAULTS and isinstance(v, (int, float))
                                                                  and isinstance(DEFAULTS[k], float)):
                self.st[k] = v
        self.st["label_full"] = self.st["label_hide"] + 0.4
        self.side.setVisible(False)
        if self.st["show_list"]:
            self._toggle_side(True, animate=False)
        self.canvas.restyle()
        return self.st["show_meetings"] != was_meet       # reload so meetings appear / disappear

    def _save_settings(self):
        if self._pid and self._root is not None:
            save_settings_async(self._root, self._pid, dict({k: self.st[k] for k in DEFAULTS}, settings_v=SETTINGS_V))

    def _on_setting(self, key: str, value):
        self.st[key] = value
        self._save_timer.start()
        if key == "show_meetings":
            self._start_job("load")              # meetings are laid out (once) only when shown
        elif key in ("filter_text", "hidden_types", "hidden_status", "show_orphans"):
            self.apply_filters()
        elif key == "arrows":
            for e in self.canvas.edges:
                e.update()
        elif key == "show_list":
            self._toggle_side(bool(value))
        elif key in ("node_scale", "link_scale", "color_by"):
            self.canvas.restyle()
        elif key == "label_hide":
            self.st["label_full"] = value + 0.4
            self.canvas._apply_label_visibility()
        elif key in ("spacing", "spread"):
            self.rearrange()

    def _reset(self):
        layout_changed = (self.st["spacing"], self.st["spread"]) != (DEFAULTS["spacing"], DEFAULTS["spread"])
        meet_changed = self.st["show_meetings"] != DEFAULTS["show_meetings"]
        self.st.update(DEFAULTS)
        self.st["label_full"] = self.st["label_hide"] + 0.4
        self._toggle_side(False)
        self._refresh_panel()
        self.canvas.restyle()
        self.apply_filters()
        self._save_settings()
        if meet_changed:
            self._start_job("load")
        elif layout_changed:
            self.rearrange()

    def _refresh_panel(self):
        types = sorted({edge_verb(e) for it in self.canvas.edges for e in it.edges})
        statuses = []
        for word in ("Proposed", "Accepted", "Changed", "Blocked", "Archived"):
            if any(status_word(n.data.get("status"), n.data.get("revision", 1)) == word
                   for n in self.canvas.nodes.values() if n.kind == "feature"):
                statuses.append(word)
        self.panel.set_values(self.st, types, statuses)

    def _meeting_colors(self):
        """For 'Color by first meeting': each feature gets the color of the earliest meeting that touched it."""
        meetings = {n["id"]: n for n in self._graph["nodes"] if n["type"] == "meeting"}
        order = sorted(meetings, key=lambda m: (str(meetings[m].get("date", "")), m))
        pal = tk.g("categorical")
        mcol = {m: pal[i % len(pal)] for i, m in enumerate(order)}
        first: dict[str, str] = {}
        for e in self._graph["edges"]:
            if e["type"] == "meeting_updates_feature" and e["src"] in mcol:
                cur = first.get(e["dst"])
                if cur is None or order.index(e["src"]) < order.index(cur):
                    first[e["dst"]] = e["src"]
        self.canvas.meeting_color = {f: mcol[m] for f, m in first.items()}
        self._meeting_legend = [(mcol[m], meetings[m].get("label", "Meeting")) for m in order]

    # ----- filters + local graph (no layout work: items just fade out / in) -----
    def apply_filters(self, animate: bool = True, on_done=None):
        c = self.canvas
        txt = self.st["filter_text"].strip().lower()
        hid_status = set(self.st["hidden_status"])
        hid_types = set(self.st["hidden_types"])
        vis = set()
        for nid, n in c.nodes.items():
            if n.kind == "feature":
                if status_word(n.data.get("status"), n.data.get("revision", 1)) in hid_status:
                    continue
                if txt and txt not in n.data.get("label", "").lower():
                    continue
            vis.add(nid)
        if self.local:
            vis &= self._neighbourhood(*self.local)
        edges = {e for e in c.edges if e.a.nid in vis and e.b.nid in vis
                 and any(edge_verb(x) not in hid_types for x in e.edges)}
        if not self.st["show_orphans"] and not self.local:
            linked = {x for e in edges if e.kind != "meeting" for x in (e.a.nid, e.b.nid)}
            vis = {nid for nid in vis if c.nodes[nid].kind == "meeting" or nid in linked}
            edges = {e for e in edges if e.a.nid in vis and e.b.nid in vis}
        everything = len(vis) == len(c.nodes) and len(edges) == len(c.edges)
        c.vis_nodes = None if everything else vis
        c.vis_edges = None if everything else edges
        if c.selected is not None and c.is_hidden(c.selected.nid):
            c.clear_selection()
        c.refocus(animate, tk.g("anim.filter_ms"), on_done)
        self._fill_list()
        if self.search.isVisible():
            self._run_search()

    def _neighbourhood(self, nid: str, depth: int) -> set[str]:
        seen, frontier = {nid}, {nid}
        for _ in range(depth):
            frontier = {m for x in frontier for m in self.canvas.adj.get(x, ())} - seen
            seen |= frontier
        return seen

    def enter_local(self, nid: str, depth: int = 1):
        """'Focus': only this node and its neighbours (depth 1-3), in their saved positions, then fit them."""
        if nid not in self.canvas.nodes:
            return
        self.local = (nid, max(1, min(3, depth)))
        self.canvas.card.hide()                  # the card did its job; it would cover a neighbour
        self.chip.set_state(short(self.canvas.label_of(nid)), self.local[1])
        self.chip.show()
        self._place_overlays()
        self.apply_filters(on_done=lambda: self.canvas.fit())

    def exit_local(self):
        if not self.local:
            return
        self.local = None
        self.chip.hide()
        self.apply_filters(on_done=lambda: self.canvas.fit())

    # ----- search (Ctrl+F) -----
    def open_search(self):
        if not self.canvas.nodes:
            return
        self.search.show()
        self._place_overlays()
        self.search.edit.setFocus()
        self.search.edit.selectAll()

    def close_search(self):
        if self.search.isVisible():
            if self.search.isAncestorOf(QApplication.focusWidget()):
                self.canvas.setFocus(Qt.OtherFocusReason)   # give focus back before the box disappears
            self.search.hide()
            self.search.edit.clear()
        self._matches, self._match_i = [], -1
        if self.canvas.search_ids is not None:
            self.canvas.search_ids = None
            self.canvas.refocus()
        self._place_overlays()

    def _run_search(self):
        q = self.search.edit.text().strip().lower()
        c = self.canvas
        if not q:
            self._matches, self._match_i = [], -1
            c.search_ids = None
            self.search.count.setText("")
        else:
            ms = [n for n in c.nodes.values() if not c.is_hidden(n.nid) and q in n.data.get("label", "").lower()]
            ms.sort(key=lambda n: (not n.data.get("label", "").lower().startswith(q), -n.degree))
            self._matches = [n.nid for n in ms]
            self._match_i = -1
            c.search_ids = set(self._matches)
            self.search.count.setText(f"{len(ms)} found" if ms else "none")
        c.refocus()

    def _search_step(self, d: int):
        if self._search_timer.isActive():
            self._search_timer.stop()
            self._run_search()
        if not self._matches:
            return
        self._match_i = 0 if (d == 0 and self._match_i < 0) else (self._match_i + (d or 1)) % len(self._matches)
        n = self.canvas.nodes[self._matches[self._match_i]]
        self.search.count.setText(f"{self._match_i + 1} of {len(self._matches)}")
        self.canvas.animate_view(n.pos(), max(self.canvas.zoom, 1.0), tk.g("anim.fit_ms"))
        self.canvas.select_node(n)

    # ----- Esc: close the innermost thing first -----
    def _on_escape(self):
        if self.search.isVisible():
            self.close_search()
        elif self.panel.isVisible():
            self.b_gear.setChecked(False)
        elif self.local:
            self.exit_local()
        elif self.canvas._kbd is not None:
            self.canvas.clear_kbd()
        else:
            self.canvas.clear_selection()

    # ----- hand-placed nodes -----
    def _on_moved(self, nid: str, pos):
        self._pinned.add(nid)
        if self._pid and self._root is not None:
            save_settings_async(self._root, self._pid, {}, pinned=self._pinned,
                                positions_update={nid: (pos.x(), pos.y())})
        self._shown_sig = None          # positions changed on screen; next reload compares afresh

    def _on_unpinned(self, nid: str):
        self._pinned.discard(nid)
        if self._pid and self._root is not None:
            save_settings_async(self._root, self._pid, {}, pinned=self._pinned)

    # ----- back navigation: keep zoom, pan and selection -----
    def view_state(self) -> dict:
        c = self.canvas
        centre = c.mapToScene(c.viewport().rect().center())
        return {"centre": (centre.x(), centre.y()), "zoom": c.zoom,
                "selected": c.selected.nid if c.selected is not None else None, "project": self._pid}

    def restore_state(self, st: dict):
        if not st or st.get("project") != self._pid:
            return
        c = self.canvas
        self._need_fit = False
        c.animate_view(QPointF(*st["centre"]), st["zoom"], 0)
        sel = st.get("selected")
        if sel in c.nodes and not c.is_hidden(sel):
            c.select_node(c.nodes[sel], animate=False)
        else:
            c.clear_selection()

    def _fit_if_ready(self):
        """Fit only once the canvas is on screen with its real size (a hidden tab reports a tiny viewport)."""
        if getattr(self, "_need_fit", False) and self.canvas.isVisible() \
                and self.canvas.viewport().width() > 200 and self.canvas.viewport().height() > 150:
            self._need_fit = False
            self.canvas.fit(animate=False)

    # ----- relationship list -----
    def _fill_list(self):
        self.rel_list.blockSignals(True)
        self.rel_list.clear()
        self._row_edges = []
        c = self.canvas
        hid = set(self.st["hidden_types"])
        for it in c.edges:
            if c.vis_edges is not None and it not in c.vis_edges:
                continue
            for e in it.edges:
                if edge_verb(e) in hid:
                    continue
                self.rel_list.addItem(QListWidgetItem(f"{c.label_of(e['src'])} → {edge_verb(e)} → "
                                                      f"{c.label_of(e['dst'])}"))
                self._row_edges.append((it, e))
        self.rel_list.blockSignals(False)

    def _on_row_hover(self, item):
        row = self.rel_list.row(item)
        if 0 <= row < len(self._row_edges):
            self.canvas.hover_edge(self._row_edges[row][0])

    def eventFilter(self, obj, ev):
        if obj is self.rel_list.viewport() and ev.type() == QEvent.Leave:
            self.canvas.hover_edge(None, leaving=self.canvas._hover_e)
        return False

    def _on_edge(self, item):
        if item is None:
            self.reason.setText("Hover a row to find it on the map; click to go there.")
            return
        self.reason.setText(item.describe())

    def _on_row(self, row):
        if 0 <= row < len(self._row_edges):
            it, e = self._row_edges[row]
            self.canvas.center_on_edge(it)
            self.reason.setText(f"{self.canvas.label_of(e['src'])} → {edge_verb(e)} → "
                                f"{self.canvas.label_of(e['dst'])}" + (f"\nWhy: {e['why']}" if e.get("why") else "")
                                + (f"\n\n{e['reason']}" if e.get("reason") and not e.get("why") else ""))

    # ----- sliding panels -----
    def _toggle_panel(self, on: bool):
        self._slide(self.panel, self._panel_anim, on, self._panel_pos)
        if on:
            self._refresh_panel()

    def _toggle_side(self, on: bool, animate: bool = True):
        if self.st["show_list"] != on:
            self.st["show_list"] = on
        self._slide(self.side, self._side_anim, on, self._side_pos, animate)
        if on:
            self._fill_list()

    def _panel_pos(self) -> QPoint:
        right = self.width() - (self.side.width() if self.side.isVisible() else 0)
        return QPoint(right - self.panel.width() - tk.SPACE[12], tk.SPACE[12] + 36)

    def _side_pos(self) -> QPoint:
        return QPoint(self.width() - self.side.width(), 0)

    def _slide(self, w: QWidget, anim: QVariantAnimation, on: bool, target, animate: bool = True):
        anim.stop()
        w._closing = not on
        if w is self.panel:
            w.resize(w.width(), min(560, self.height() - 60))
        else:
            w.resize(w.width(), self.height())
        end = target()
        off = QPoint(self.width() + 4, end.y())
        if on:
            if not w.isVisible():
                w.move(off)
                w.show()
            w.raise_()
            start, stop = w.pos(), end
        else:
            if not w.isVisible():
                return
            start, stop = w.pos(), off
        if animate and motion.enabled():
            anim.setStartValue(QPointF(start))
            anim.setEndValue(QPointF(stop))
            anim.start()
        else:
            w.move(stop)
            if not on:
                w.hide()
        self._place_overlays()                   # Fit / gear move left of the list while it is open

    # ----- legend -----
    def _legend(self):
        pop = Popover(self)
        t = QLabel("What you see")
        t.setObjectName("caption")
        pop.lay.addWidget(t)
        mode = self.st["color_by"]
        if mode == "group":
            rows = [(tk.g("categorical")[0], "Colour = topic group: features linked together share a colour"),
                    (tk.g("ungrouped"), "Not linked to anything yet")]
        elif mode == "status":
            rows = [(tk.STATUS["proposed"], "Proposed"), (tk.STATUS["active"], "Accepted"),
                    (tk.STATUS["changed"], "Changed since first saved"), (tk.g("conflict"), "Blocked / conflict")]
        elif mode == "meeting":
            rows = [(c, f"First discussed in {name}") for c, name in getattr(self, "_meeting_legend", [])[:8]]
        else:
            rows = [(tk.STATUS["proposed"], "Feature")]
        for color, text in rows:
            pop.lay.addWidget(QLabel(f"<span style='color:{color}'>●</span> {text}"))
        pop.lay.addWidget(QLabel(f"<span style='color:{theme.c('accent')}'>■</span> Meeting (when shown)"))
        for text in ("Bigger dot = more connections", "Stronger line = depends on / produces",
                     "Fainter line = uses / is part of", "Dashed blue = a transcript/meeting added or changed it",
                     "Dot on a node = you placed it by hand"):
            lab = QLabel(text)
            lab.setObjectName("meta")
            pop.lay.addWidget(lab)
        keys = QLabel("Keys: F fit · L focus · Ctrl+F search · Tab next node · Enter open · Esc close")
        keys.setObjectName("meta")
        keys.setWordWrap(True)
        pop.lay.addWidget(keys)
        pop.show_for(self.b_help, above=True, align_right=False)

    # ----- geometry -----
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._place_overlays()
        self._fit_if_ready()

    def showEvent(self, e):
        super().showEvent(e)
        self._place_overlays()
        self._fit_if_ready()

    def _place_overlays(self):
        m = tk.SPACE[12]
        cw = self.canvas.width() - (self.side.width() if self.side.isVisible() else 0)
        self.bar.adjustSize()
        self.bar.move(cw - self.bar.width() - m, m)
        self.b_help.adjustSize()
        self.b_help.move(m, self.canvas.height() - self.b_help.height() - m)
        self.empty.setGeometry(self.canvas.rect())
        y = m
        if self.search.isVisible():
            self.search.adjustSize()
            self.search.move(m, y)
            y += self.search.height() + tk.SPACE[8]
        if self.chip.isVisible():
            self.chip.adjustSize()
            self.chip.move(m, y)
        if self.side.isVisible() and self._side_anim.state() != QVariantAnimation.Running:
            self.side.resize(self.side.width(), self.height())
            self.side.move(self._side_pos())
        if self.panel.isVisible() and self._panel_anim.state() != QVariantAnimation.Running:
            self.panel.move(self._panel_pos())
        for w in (self.bar, self.b_help, self.search, self.chip):
            w.raise_()
