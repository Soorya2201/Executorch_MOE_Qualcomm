"""Shows every reusable UI component in the current (or forced) theme.

    python tools\\ui_gallery.py            # follows Windows light/dark
    python tools\\ui_gallery.py dark       # force dark
    python tools\\ui_gallery.py light out.png   # save a screenshot and exit
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,  # noqa: E402
                               QPushButton, QVBoxLayout, QWidget)

from app.ui import components as ui, theme  # noqa: E402
from app.ui import tokens as tk  # noqa: E402


def build() -> QWidget:
    w = QWidget()
    w.setWindowTitle("MeetingGraph components")
    w.setAutoFillBackground(True)
    root = QHBoxLayout(w)
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)

    side = QWidget()
    side.setObjectName("sidebar")
    side.setStyleSheet(f"QWidget#sidebar {{ background: {theme.c('sidebar')}; }}")
    side.setFixedWidth(tk.SIDEBAR_W)
    sl = QVBoxLayout(side)
    sl.setContentsMargins(12, 16, 12, 12)
    b = QPushButton("New meeting")
    b.setObjectName("primary")
    sl.addWidget(b)
    sl.addSpacing(12)
    cap = QLabel("Projects")
    cap.setObjectName("caption")
    sl.addWidget(cap)
    lst = ui.styled_list(QListWidget())
    for name, n in (("Atlas", 4), ("Bio2", 1), ("Biology", 1), ("BabaBlackSheep", 2)):
        it = QListWidgetItem(name)
        it.setData(ui.ROLE_SUBTITLE, f"{n} meeting{'s' if n != 1 else ''}")
        lst.addItem(it)
    lst.setCurrentRow(1)
    sl.addWidget(lst, 1)
    q = QPushButton("+ New project")
    q.setObjectName("quiet")
    sl.addWidget(q)
    root.addWidget(side)

    main = QWidget()
    ml = QVBoxLayout(main)
    ml.setContentsMargins(32, 16, 32, 24)
    ml.setSpacing(16)
    seg = ui.SegmentedControl(["To review", "Meetings", "Features", "Map"])
    seg.setBadge(0, 1)
    seg.setCurrentIndex(2, animate=False)
    row = QHBoxLayout()
    row.addStretch(1)
    row.addWidget(seg)
    row.addStretch(1)
    ml.addLayout(row)
    t = QLabel("DNA")
    t.setObjectName("title")
    ml.addWidget(t)
    meta = QHBoxLayout()
    meta.addWidget(ui.StatusDot("active"))
    m = QLabel("Accepted · v1 · updated Oct 18")
    m.setObjectName("meta")
    meta.addWidget(m)
    meta.addStretch(1)
    ml.addLayout(meta)
    card = ui.Card()
    h = QLabel("What it is")
    h.setObjectName("section")
    card.lay.addWidget(h)
    body = QLabel("Deoxyribonucleic acid is a molecule that contains the instructions for making proteins in cells.")
    body.setWordWrap(True)
    card.lay.addWidget(body)
    card.lay.addWidget(ui.hairline())
    chips = ui.FlowWrap()
    for c in ("+ Prokaryotes", "+ Eukaryotes", "+ DNA", "+ Ribosomes", "+ Cell Membrane", "+ Mitosis"):
        chips.add(ui.Chip(c))
    card.lay.addWidget(chips)
    ml.addWidget(card)
    dots = QHBoxLayout()
    for s in ("proposed", "active", "changed", "blocked"):
        dots.addWidget(ui.StatusDot(s))
        lab = QLabel(s)
        lab.setObjectName("meta")
        dots.addWidget(lab)
        dots.addSpacing(12)
    dots.addStretch(1)
    ml.addLayout(dots)
    inp = QLineEdit()
    inp.setPlaceholderText("Ask about Bio2…")
    ml.addWidget(inp)
    think = ui.Thinking()
    think.start()
    ml.addWidget(think)
    ml.addStretch(1)
    root.addWidget(main, 1)

    w.resize(1100, 640)
    w.toast = ui.Toast(w)
    QTimer.singleShot(50, lambda: w.toast.show_message("“Meeting Oct 18 00:10” needs review.", "Open"))
    return w


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else None
    app = QApplication(sys.argv[:1])
    theme.apply(app, mode)
    w = build()
    w.show()
    if len(sys.argv) > 2:
        def snap():
            w.grab().save(sys.argv[2])
            app.quit()
        QTimer.singleShot(600, snap)
    sys.exit(app.exec())
