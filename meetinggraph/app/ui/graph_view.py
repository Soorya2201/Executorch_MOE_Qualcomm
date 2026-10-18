"""Pan/zoom graph of one project's approved meetings and features, plus an edge list with reasons."""
from __future__ import annotations

import math
import random

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPainterPathStroker, QPen
from PySide6.QtWidgets import (QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem, QGraphicsRectItem,
                               QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QSplitter, QVBoxLayout, QWidget)

from .common import STATUS_COLOR

EDGE_STYLE = {
    "feature_depends_on_feature": (QColor("#e53935"), Qt.SolidLine, "depends on"),
    "feature_relates_to_feature": (QColor("#1e88e5"), Qt.SolidLine, "is connected to"),
    "meeting_updates_feature": (QColor("#90a4ae"), Qt.DashLine, "updated"),
    "meeting_relates_to_meeting": (QColor("#8e24aa"), Qt.DotLine, "relates to"),
}


def layout(nodes: list[dict], edges: list[dict], iters: int = 300) -> dict[str, QPointF]:
    """Deterministic Fruchterman–Reingold layout."""
    rnd = random.Random(7)
    n = max(len(nodes), 1)
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
    def __init__(self, a: QPointF, b: QPointF, edge: dict):
        super().__init__()
        self.edge = edge
        color, style, _ = EDGE_STYLE.get(edge["type"], (QColor("#999"), Qt.SolidLine, ""))
        self.base_pen = QPen(color, 1.6, style)
        self.setPen(self.base_pen)
        path = QPainterPath(a)
        path.lineTo(b)
        if edge["type"] in ("feature_depends_on_feature", "feature_relates_to_feature"):   # arrowhead
            ang = math.atan2(b.y() - a.y(), b.x() - a.x())
            tip = QPointF(b.x() - 26 * math.cos(ang), b.y() - 26 * math.sin(ang))
            for s in (-0.4, 0.4):
                path.moveTo(tip)
                path.lineTo(tip.x() - 10 * math.cos(ang + s), tip.y() - 10 * math.sin(ang + s))
        self.setPath(path)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setToolTip(edge["reason"])
        self.setZValue(-1)

    def shape(self):
        s = QPainterPathStroker()
        s.setWidth(10)
        return s.createStroke(self.path())

    def highlight(self, on: bool):
        p = QPen(self.base_pen)
        p.setWidthF(3.5 if on else 1.6)
        self.setPen(p)


class GraphView(QWidget):
    open_node = Signal(str, str)   # kind, id

    def __init__(self):
        super().__init__()
        lay = QHBoxLayout(self)
        split = QSplitter()
        lay.addWidget(split)
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.wheelEvent = self._wheel
        split.addWidget(self.view)
        side = QWidget()
        sl = QVBoxLayout(side)
        self.reason = QLabel("Click an edge or a row to see why it exists.")
        self.reason.setWordWrap(True)
        self.reason.setStyleSheet("padding:6px;background:#f5f5f5;border-radius:4px;color:#222;")
        legend = QLabel("<span style='color:#e53935'>━</span> depends on &nbsp; "
                        "<span style='color:#90a4ae'>╍</span> meeting updated feature &nbsp; "
                        "<span style='color:#8e24aa'>┈</span> meetings share a feature<br>"
                        "● feature (colour = status) &nbsp; ■ meeting · double-click a node to open it")
        legend.setWordWrap(True)
        legend.setTextFormat(Qt.RichText)
        self.list = QListWidget()
        sl.addWidget(legend)
        sl.addWidget(self.reason)
        sl.addWidget(QLabel("Relationships"))
        sl.addWidget(self.list, 1)
        split.addWidget(side)
        split.setSizes([700, 320])
        self.edge_items: list[EdgeItem] = []
        self.scene.selectionChanged.connect(self._on_select)
        self.list.currentRowChanged.connect(self._on_row)

    def _wheel(self, ev):
        f = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
        self.view.scale(f, f)

    def show_graph(self, g: dict):
        self.scene.clear()
        self.list.clear()
        self.edge_items = []
        nodes, edges = g["nodes"], g["edges"]
        if not nodes:
            self.scene.addText("No approved meetings yet.")
            return
        pos = layout(nodes, edges)
        label = {n["id"]: n["label"] for n in nodes}
        for e in edges:
            if e["src"] in pos and e["dst"] in pos:
                it = EdgeItem(pos[e["src"]], pos[e["dst"]], e)
                self.scene.addItem(it)
                self.edge_items.append(it)
                verb = e.get("verb") or EDGE_STYLE.get(e["type"], (None, None, e["type"]))[2]
                QListWidgetItem(f"{label[e['src']]} — {verb} → {label[e['dst']]}", self.list)
        font = QFont()
        font.setPointSize(9)
        for n in nodes:
            p = pos[n["id"]]
            if n["type"] == "feature":
                item = QGraphicsEllipseItem(QRectF(p.x() - 22, p.y() - 22, 44, 44))
                item.setBrush(QBrush(QColor(STATUS_COLOR.get(n.get("status"), "#9e9e9e"))))
                tip = f"Feature · {n['label']} · {n.get('status', '')}"
            else:
                item = QGraphicsRectItem(QRectF(p.x() - 18, p.y() - 18, 36, 36))
                item.setBrush(QBrush(QColor("#ffe082")))
                tip = f"Meeting · {n['label']} · {n.get('date', '')}"
            item.setPen(QPen(QColor("#555"), 1))
            item.setToolTip(tip)
            item.setData(0, (n["type"], n["id"]))
            item.mouseDoubleClickEvent = (lambda ev, t=n["type"], i=n["id"]: self.open_node.emit(t, i))
            self.scene.addItem(item)
            txt = QGraphicsSimpleTextItem(n["label"] + (" (archived)" if n.get("status") == "archived" else ""))
            txt.setFont(font)
            br = txt.boundingRect()
            txt.setPos(p.x() - br.width() / 2, p.y() + 25)
            self.scene.addItem(txt)
        self.view.fitInView(self.scene.itemsBoundingRect().adjusted(-40, -40, 40, 40), Qt.KeepAspectRatio)

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
