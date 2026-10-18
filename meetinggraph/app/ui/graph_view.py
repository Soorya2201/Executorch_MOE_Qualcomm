"""Pan/zoom map of one project's approved meetings and features, plus a relationship list with reasons.

Fluent look: a white canvas with a faint dot grid (like Whiteboard), feature nodes coloured by status with a
white ring and soft shadow, meetings as brand-blue tiles, filled arrowheads, and a side panel that wraps text.
"""
from __future__ import annotations

import math
import random

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPainterPathStroker, QPen, QPolygonF
from PySide6.QtWidgets import (QFrame, QGraphicsDropShadowEffect, QGraphicsEllipseItem, QGraphicsItem,
                               QGraphicsPathItem, QGraphicsPolygonItem, QGraphicsRectItem, QGraphicsScene,
                               QGraphicsSimpleTextItem, QGraphicsView, QHBoxLayout, QLabel, QSplitter,
                               QVBoxLayout, QWidget, QListWidgetItem)

from . import theme
from .common import STATUS_COLOR, wrapping_list

EDGE_STYLE = {
    "feature_depends_on_feature": (QColor(theme.ACCENT), Qt.SolidLine, "depends on"),
    "feature_relates_to_feature": (QColor("#7f6bd9"), Qt.SolidLine, "is connected to"),
    "meeting_updates_feature": (QColor("#a8a8a8"), Qt.DashLine, "updated"),
    "meeting_relates_to_meeting": (QColor("#c239b3"), Qt.DotLine, "relates to"),
}
ARROWED = ("feature_depends_on_feature", "feature_relates_to_feature")
FEATURE_R, MEETING_S = 20.0, 30.0


def layout(nodes: list[dict], edges: list[dict], iters: int = 300) -> dict[str, QPointF]:
    """Deterministic Fruchterman–Reingold layout."""
    rnd = random.Random(7)
    k = 140.0
    pos = {nd["id"]: [rnd.uniform(-200, 200), rnd.uniform(-200, 200)] for nd in nodes}
    links = [(e["src"], e["dst"]) for e in edges if e["src"] in pos and e["dst"] in pos]
    t = 120.0
    for _ in range(iters):
        disp = {i: [0.0, 0.0] for i in pos}
        ids = list(pos)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                dx, dy = pos[a][0] - pos[b][0], pos[a][1] - pos[b][1]
                d = math.hypot(dx, dy) or 0.01
                f = k * k / d
                disp[a][0] += dx / d * f
                disp[a][1] += dy / d * f
                disp[b][0] -= dx / d * f
                disp[b][1] -= dy / d * f
        for a, b in links:
            dx, dy = pos[a][0] - pos[b][0], pos[a][1] - pos[b][1]
            d = math.hypot(dx, dy) or 0.01
            f = d * d / k
            disp[a][0] -= dx / d * f
            disp[a][1] -= dy / d * f
            disp[b][0] += dx / d * f
            disp[b][1] += dy / d * f
        for i in pos:
            dx, dy = disp[i]
            d = math.hypot(dx, dy) or 0.01
            pos[i][0] += dx / d * min(d, t)
            pos[i][1] += dy / d * min(d, t)
        t = max(2.0, t * 0.97)
    return {i: QPointF(x, y) for i, (x, y) in pos.items()}


class EdgeItem(QGraphicsPathItem):
    def __init__(self, a: QPointF, b: QPointF, edge: dict, end_gap: float):
        super().__init__()
        self.edge = edge
        color, style, _ = EDGE_STYLE.get(edge["type"], (QColor("#a8a8a8"), Qt.SolidLine, ""))
        self.color = color
        self.base_pen = QPen(color, 1.6, style, Qt.RoundCap)
        self.setPen(self.base_pen)
        ang = math.atan2(b.y() - a.y(), b.x() - a.x())
        end = QPointF(b.x() - end_gap * math.cos(ang), b.y() - end_gap * math.sin(ang))
        path = QPainterPath(a)
        path.lineTo(end)
        self.setPath(path)
        self.head = None
        if edge["type"] in ARROWED:                                    # filled arrowhead touching the node
            s = 9.0
            pts = [end, QPointF(end.x() - s * math.cos(ang - 0.42), end.y() - s * math.sin(ang - 0.42)),
                   QPointF(end.x() - s * math.cos(ang + 0.42), end.y() - s * math.sin(ang + 0.42))]
            self.head = QGraphicsPolygonItem(QPolygonF(pts), self)
            self.head.setBrush(color)
            self.head.setPen(Qt.NoPen)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setToolTip(edge["reason"])
        self.setZValue(-1)

    def shape(self):
        s = QPainterPathStroker()
        s.setWidth(12)
        return s.createStroke(self.path())

    def highlight(self, on: bool):
        p = QPen(self.base_pen)
        p.setWidthF(3.2 if on else 1.6)
        self.setPen(p)


class _Canvas(QGraphicsView):
    """White board with a faint dot grid that pans and zooms with the content."""
    GRID = 24

    def drawBackground(self, p: QPainter, rect: QRectF):
        p.fillRect(rect, QColor(theme.SURFACE))
        p.setPen(QPen(QColor("#dcdcdc"), 1.6, Qt.SolidLine, Qt.RoundCap))
        g = self.GRID
        x0, y0 = math.floor(rect.left() / g) * g, math.floor(rect.top() / g) * g
        y = y0
        while y < rect.bottom():
            x = x0
            while x < rect.right():
                p.drawPoint(QPointF(x, y))
                x += g
            y += g


def _panel() -> tuple[QFrame, QVBoxLayout]:
    fr = QFrame()
    fr.setObjectName("panel")
    lay = QVBoxLayout(fr)
    lay.setContentsMargins(1, 1, 1, 1)
    return fr, lay


class GraphView(QWidget):
    open_node = Signal(str, str)   # kind, id

    def __init__(self):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter()
        split.setChildrenCollapsible(False)
        lay.addWidget(split)

        canvas_panel, cl = _panel()
        self.scene = QGraphicsScene()
        self.view = _Canvas(self.scene)
        self.view.setFrameShape(QFrame.NoFrame)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setRenderHint(QPainter.TextAntialiasing)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)   # dot grid stays crisp while panning
        self.view.wheelEvent = self._wheel
        cl.addWidget(self.view)
        split.addWidget(canvas_panel)

        side, sl = _panel()
        side.setMinimumWidth(240)
        sl.setContentsMargins(16, 16, 16, 12)
        sl.setSpacing(10)
        title = QLabel("Map")
        title.setObjectName("h2")
        legend = QLabel(
            f"<span style='color:{theme.ACCENT}'>━━</span>&nbsp; needs (arrow points to what it needs)<br>"
            "<span style='color:#7f6bd9'>━━</span>&nbsp; connected to<br>"
            "<span style='color:#a8a8a8'>╍╍</span>&nbsp; a meeting updated a feature<br>"
            "<span style='color:#c239b3'>┈┈</span>&nbsp; meetings share a feature<br>"
            f"<span style='color:{theme.MUTED}'>● feature (colour = status) &nbsp;▢ meeting<br>"
            "Double-click a node to open it · scroll to zoom · drag to move</span>")
        legend.setWordWrap(True)
        legend.setTextFormat(Qt.RichText)
        legend.setObjectName("small")
        self.reason = QLabel("Click a line or a row to see why it exists.")
        self.reason.setWordWrap(True)
        self.reason.setObjectName("hint")
        rel = QLabel("RELATIONSHIPS")
        rel.setObjectName("eyebrow")
        self.list = wrapping_list()
        self.list.setObjectName("bare")
        for w in (title, legend, self.reason, rel):
            sl.addWidget(w)
        sl.addWidget(self.list, 1)
        split.addWidget(side)
        split.setSizes([700, 300])
        self.panels = (canvas_panel, side)       # the main window paints shadows behind these
        self.edge_items: list[EdgeItem] = []
        self.scene.selectionChanged.connect(self._on_select)
        self.list.currentRowChanged.connect(self._on_row)

    def _wheel(self, ev):
        f = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
        self.view.scale(f, f)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._fit()

    def _fit(self):
        if self.scene.items():
            self.view.fitInView(self.scene.itemsBoundingRect().adjusted(-48, -48, 48, 48), Qt.KeepAspectRatio)

    def show_graph(self, g: dict):
        self.scene.clear()
        self.list.clear()
        self.edge_items = []
        nodes, edges = g["nodes"], g["edges"]
        if not nodes:
            t = self.scene.addSimpleText("No saved meetings yet. Save one from Review and it appears here.")
            t.setBrush(QColor(theme.MUTED))
            return
        pos = layout(nodes, edges)
        label = {n["id"]: n["label"] for n in nodes}
        kind = {n["id"]: n["type"] for n in nodes}
        for e in edges:
            if e["src"] in pos and e["dst"] in pos:
                gap = (FEATURE_R + 3) if kind.get(e["dst"]) == "feature" else (MEETING_S / 2 + 3)
                it = EdgeItem(pos[e["src"]], pos[e["dst"]], e, gap)
                self.scene.addItem(it)
                self.edge_items.append(it)
                verb = e.get("verb") or EDGE_STYLE.get(e["type"], (None, None, e["type"]))[2]
                QListWidgetItem(f"{label[e['src']]} — {verb} → {label[e['dst']]}", self.list)
        font = QFont()
        font.setFamilies(["Segoe UI Variable Text", "Segoe UI"])
        font.setPointSizeF(8.5)
        for n in nodes:
            p = pos[n["id"]]
            if n["type"] == "feature":
                item = QGraphicsEllipseItem(QRectF(p.x() - FEATURE_R, p.y() - FEATURE_R, 2 * FEATURE_R, 2 * FEATURE_R))
                item.setBrush(QBrush(QColor(STATUS_COLOR.get(n.get("status"), "#9e9e9e"))))
                item.setPen(QPen(QColor("#ffffff"), 3))
                tip = f"Feature · {n['label']} · {n.get('status', '')}"
                below = FEATURE_R + 6
            else:
                h = MEETING_S / 2
                item = QGraphicsRectItem(QRectF(p.x() - h, p.y() - h, MEETING_S, MEETING_S))
                item.setBrush(QBrush(QColor(theme.ACCENT_SOFT)))
                item.setPen(QPen(QColor(theme.ACCENT), 1.6))
                tip = f"Meeting · {n['label']} · {n.get('date', '')}"
                below = h + 6
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(10)
            shadow.setOffset(0, 1.5)
            shadow.setColor(QColor(0, 0, 0, 45))
            item.setGraphicsEffect(shadow)
            item.setToolTip(tip)
            item.setData(0, (n["type"], n["id"]))
            item.setCursor(Qt.PointingHandCursor)
            item.mouseDoubleClickEvent = (lambda ev, t=n["type"], i=n["id"]: self.open_node.emit(t, i))
            self.scene.addItem(item)
            name = n["label"] + (" (archived)" if n.get("status") == "archived" else "")
            if len(name) > 34:
                name = name[:33].rstrip() + "…"
            txt = QGraphicsSimpleTextItem(name)
            txt.setFont(font)
            txt.setBrush(QColor(theme.TEXT_2))
            txt.setToolTip(n["label"])
            br = txt.boundingRect()
            txt.setPos(p.x() - br.width() / 2, p.y() + below)
            self.scene.addItem(txt)
        self._fit()

    def _on_select(self):
        for i, it in enumerate(self.edge_items):
            sel = it.isSelected()
            it.highlight(sel)
            if sel:
                self.reason.setText(it.edge["reason"])
                self.list.blockSignals(True)
                self.list.setCurrentRow(i)
                self.list.blockSignals(False)

    def _on_row(self, row: int):
        for i, it in enumerate(self.edge_items):
            it.highlight(i == row)
        if 0 <= row < len(self.edge_items):
            self.reason.setText(self.edge_items[row].edge["reason"])
