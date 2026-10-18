"""Floating widgets that sit on top of the Map canvas: the controls panel (gear), the search field (Ctrl+F)
and the "Focused: X" chip. Plain widgets styled from tokens; they emit what changed and the Map applies it.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QScrollArea, QSlider, QVBoxLayout, QWidget)

from . import theme
from . import tokens as tk

DEFAULTS = {
    "show_meetings": True, "show_orphans": True, "arrows": False, "show_list": False,
    "filter_text": "", "hidden_types": [], "hidden_status": [],
    "color_by": "group", "node_scale": 1.0, "link_scale": 1.0, "label_hide": 0.6,
    "spacing": 1.0, "spread": 1.0,
}
COLOR_BY = [("group", "Topic group"), ("status", "Status"), ("meeting", "First meeting discussed"),
            ("none", "None")]


def map_qss() -> str:
    R, RC = tk.RADIUS["control"], tk.RADIUS["card"]
    return (
        f"QFrame#mapPanel, QFrame#mapFloat {{ background: {theme.c('surface')}; border: 1px solid "
        f"{theme.c('separator_solid')}; border-radius: {RC}px; }}"
        f"QFrame#mapPanel QLabel, QFrame#mapFloat QLabel {{ background: transparent; border: none; }}"
        f"QPushButton#groupHead {{ background: transparent; border: none; text-align: left; padding: 4px 2px;"
        f" font-weight: 600; color: {theme.c('text')}; }}"
        f"QPushButton#groupHead:hover {{ color: {theme.c('accent_text')}; }}"
        f"QPushButton#mapTool {{ background: {theme.c('surface')}; border: 1px solid {theme.c('separator')};"
        f" border-radius: {R}px; padding: 3px 10px; font-size: {tk.TYPE['meta'][0]}px; color: {theme.c('secondary')}; }}"
        f"QPushButton#mapTool:hover {{ color: {theme.c('text')}; }}"
        f"QPushButton#mapTool:checked {{ background: {theme.c('accent_soft')}; color: {theme.c('accent_text')}; }}"
        f"QPushButton#depth {{ background: transparent; border: none; padding: 1px 6px; border-radius: 4px;"
        f" color: {theme.c('secondary')}; font-size: {tk.TYPE['meta'][0]}px; }}"
        f"QPushButton#depth:checked {{ background: {theme.c('accent_soft')}; color: {theme.c('accent_text')}; }}"
        f"QWidget#mapSide {{ background: {theme.c('surface')}; border-left: 1px solid {theme.c('separator_solid')}; }}"
    )


class Group(QWidget):
    """Collapsible group: a header button and a body."""
    def __init__(self, title: str, open_: bool = True):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(tk.SPACE[4])
        self.title = title
        self.head = QPushButton()
        self.head.setObjectName("groupHead")
        self.head.setCursor(Qt.PointingHandCursor)
        self.body = QWidget()
        self.bl = QVBoxLayout(self.body)
        self.bl.setContentsMargins(2, 0, 0, tk.SPACE[8])
        self.bl.setSpacing(tk.SPACE[4])
        lay.addWidget(self.head)
        lay.addWidget(self.body)
        self.head.clicked.connect(lambda: self.set_open(not self.body.isVisible()))
        self.set_open(open_)

    def set_open(self, on: bool):
        self.body.setVisible(on)
        self.head.setText(("▾  " if on else "▸  ") + self.title)


def _meta(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("meta")
    return lab


class MapPanel(QFrame):
    """The gear panel. Emits changed(key, value) for every control; the Map applies and saves it."""
    changed = Signal(str, object)
    rearrange = Signal()
    reset = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("mapPanel")
        self.setFixedWidth(260)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(1, 1, 1, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }")
        outer.addWidget(scroll)
        inner = QWidget()
        scroll.setWidget(inner)
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(tk.SPACE[12], tk.SPACE[8], tk.SPACE[12], tk.SPACE[12])
        lay.setSpacing(tk.SPACE[4])
        self._building = False

        # ---- Filters
        g = Group("Filters")
        self.f_text = QLineEdit()
        self.f_text.setPlaceholderText("Filter by name")
        self.f_text.setClearButtonEnabled(True)
        g.bl.addWidget(self.f_text)
        self.t_meet = QCheckBox("Show meetings (transcripts)")
        self.t_orph = QCheckBox("Show unconnected features")
        self.t_arrows = QCheckBox("Arrows")
        self.t_list = QCheckBox("Relationship list")
        for w in (self.t_meet, self.t_orph, self.t_arrows, self.t_list):
            g.bl.addWidget(w)
        g.bl.addWidget(_meta("Relationship types"))
        self.types_box = QVBoxLayout()
        self.types_box.setSpacing(2)
        g.bl.addLayout(self.types_box)
        g.bl.addWidget(_meta("Status"))
        self.status_box = QVBoxLayout()
        self.status_box.setSpacing(2)
        g.bl.addLayout(self.status_box)
        lay.addWidget(g)

        # ---- Display
        d = Group("Display")
        d.bl.addWidget(_meta("Color by"))
        self.color_by = QComboBox()
        for key, text in COLOR_BY:
            self.color_by.addItem(text, key)
        d.bl.addWidget(self.color_by)
        self.s_node = self._slider(d, "Node size", 50, 200)
        self.s_link = self._slider(d, "Link thickness", 50, 200)
        self.s_label = self._slider(d, "Show labels from zoom", 20, 100, fmt=lambda v: f"{v / 100:.1f}×")
        lay.addWidget(d)

        # ---- Layout
        lg = Group("Layout")
        row = QHBoxLayout()
        self.b_arrange = QPushButton("Re-arrange")
        self.arranging = _meta("")
        row.addWidget(self.b_arrange)
        row.addWidget(self.arranging, 1)
        lg.bl.addLayout(row)
        self.s_spacing = self._slider(lg, "Spacing", 50, 200, release_only=True)
        self.s_spread = self._slider(lg, "Spread", 50, 200, release_only=True)
        self.b_reset = QPushButton("Reset to defaults")
        self.b_reset.setObjectName("link")
        lg.bl.addWidget(self.b_reset)
        lay.addWidget(lg)
        lay.addStretch(1)

        self.f_text.textChanged.connect(lambda t: self._emit("filter_text", t))
        for key, cb in (("show_meetings", self.t_meet), ("show_orphans", self.t_orph), ("arrows", self.t_arrows),
                        ("show_list", self.t_list)):
            cb.toggled.connect(lambda on, k=key: self._emit(k, on))
        self.color_by.currentIndexChanged.connect(lambda _: self._emit("color_by", self.color_by.currentData()))
        self.b_arrange.clicked.connect(self.rearrange)
        self.b_reset.clicked.connect(self.reset)
        self.type_checks: dict[str, QCheckBox] = {}
        self.status_checks: dict[str, QCheckBox] = {}

    def _slider(self, group: Group, title: str, lo: int, hi: int, release_only: bool = False, fmt=None):
        head = QHBoxLayout()
        head.addWidget(_meta(title))
        val = _meta("")
        head.addStretch(1)
        head.addWidget(val)
        group.bl.addLayout(head)
        s = QSlider(Qt.Horizontal)
        s.setRange(lo, hi)
        s.setSingleStep(5)
        s.setPageStep(25)
        group.bl.addWidget(s)
        fmt = fmt or (lambda v: f"{v / 100:.2f}×")
        s.valueChanged.connect(lambda v: val.setText(fmt(v)))
        key = {"Node size": "node_scale", "Link thickness": "link_scale", "Show labels from zoom": "label_hide",
               "Spacing": "spacing", "Spread": "spread"}[title]
        if release_only:          # layout sliders: apply once, when the handle is let go
            s.sliderReleased.connect(lambda: self._emit(key, s.value() / 100))
            s.valueChanged.connect(lambda v: (not s.isSliderDown()) and self._emit(key, v / 100))
        else:
            s.valueChanged.connect(lambda v: self._emit(key, v / 100))
        return s

    def _emit(self, key, value):
        if not self._building:
            self.changed.emit(key, value)

    # ----- fill from the Map's state -----
    def set_values(self, st: dict, types: list[str], statuses: list[str]):
        self._building = True
        try:
            if self.f_text.text() != st["filter_text"]:
                self.f_text.setText(st["filter_text"])
            self.t_meet.setChecked(st["show_meetings"])
            self.t_orph.setChecked(st["show_orphans"])
            self.t_arrows.setChecked(st["arrows"])
            self.t_list.setChecked(st["show_list"])
            self.color_by.setCurrentIndex(max(0, [k for k, _ in COLOR_BY].index(st["color_by"])
                                              if st["color_by"] in [k for k, _ in COLOR_BY] else 0))
            self.s_node.setValue(int(st["node_scale"] * 100))
            self.s_link.setValue(int(st["link_scale"] * 100))
            self.s_label.setValue(int(st["label_hide"] * 100))
            self.s_spacing.setValue(int(st["spacing"] * 100))
            self.s_spread.setValue(int(st["spread"] * 100))
            self._fill(self.types_box, self.type_checks, types, set(st["hidden_types"]), "hidden_types")
            self._fill(self.status_box, self.status_checks, statuses, set(st["hidden_status"]), "hidden_status")
        finally:
            self._building = False

    def _fill(self, box, store: dict, names: list[str], hidden: set, key: str):
        """One checkbox per type/status found in the data (not hard-coded)."""
        if list(store) != names:
            for cb in store.values():
                cb.deleteLater()
            store.clear()
            for n in names:
                cb = QCheckBox(n)
                cb.toggled.connect(lambda _on, k=key, s=store: self._emit(k, [x for x, c in s.items()
                                                                            if not c.isChecked()]))
                box.addWidget(cb)
                store[n] = cb
        for n, cb in store.items():
            cb.setChecked(n not in hidden)

    def set_busy(self, busy: bool):
        self.b_arrange.setEnabled(not busy)
        self.b_arrange.setToolTip("Wait until the AI has finished; arranging would slow it down." if busy
                                  else "Arrange the map again (pinned nodes stay where they are)")

    def set_arranging(self, text: str):
        self.arranging.setText(text)


class SearchBox(QFrame):
    """Ctrl+F field at the canvas' top-left: typing dims non-matching nodes; Enter / Up / Down move between
    matches; Esc closes."""
    text_changed = Signal(str)
    step = Signal(int)           # +1 next, -1 previous, 0 = Enter (first / current)
    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("mapFloat")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 4, 4)
        lay.setSpacing(4)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Search the map")
        self.edit.setFixedWidth(200)
        self.edit.setStyleSheet("QLineEdit { border: none; background: transparent; padding: 3px; }")
        self.count = _meta("")
        close = QPushButton("✕")
        close.setObjectName("quiet")
        close.setFixedWidth(26)
        close.setToolTip("Close (Esc)")
        lay.addWidget(self.edit)
        lay.addWidget(self.count)
        lay.addWidget(close)
        self.edit.textChanged.connect(self.text_changed)
        self.edit.installEventFilter(self)
        close.clicked.connect(self.closed)
        self.hide()

    def eventFilter(self, obj, ev):
        if obj is self.edit and ev.type() == QEvent.KeyPress:
            k = ev.key()
            if k in (Qt.Key_Return, Qt.Key_Enter):
                self.step.emit(0)
                return True
            if k == Qt.Key_Down:
                self.step.emit(1)
                return True
            if k == Qt.Key_Up:
                self.step.emit(-1)
                return True
            if k == Qt.Key_Escape:
                self.closed.emit()
                return True
        return False


class FocusChip(QFrame):
    """'Focused: DNA  1 2 3  ✕' shown while the local graph is on."""
    depth_changed = Signal(int)
    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("mapFloat")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 3, 4, 3)
        lay.setSpacing(4)
        self.label = QLabel()
        lay.addWidget(self.label)
        lay.addSpacing(4)
        lay.addWidget(_meta("depth"))
        self.depth_btns = []
        for d in (1, 2, 3):
            b = QPushButton(str(d))
            b.setObjectName("depth")
            b.setCheckable(True)
            b.setToolTip(f"Show neighbours up to {d} step{'s' if d > 1 else ''} away")
            b.clicked.connect(lambda _c, d=d: self.depth_changed.emit(d))
            lay.addWidget(b)
            self.depth_btns.append(b)
        close = QPushButton("✕")
        close.setObjectName("quiet")
        close.setFixedWidth(26)
        close.setToolTip("Back to the full map (Esc)")
        close.clicked.connect(self.closed)
        lay.addWidget(close)
        self.hide()

    def set_state(self, name: str, depth: int):
        self.label.setText(f"Focused: <b>{name}</b>")
        for i, b in enumerate(self.depth_btns, 1):
            b.setChecked(i == depth)
        self.adjustSize()
