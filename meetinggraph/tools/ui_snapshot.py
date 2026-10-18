"""Render the main window off-screen, save one PNG per page, and report clipped or overlapping text.

  python tools\\ui_snapshot.py OUT_DIR [--demo] [1100x700]

Uses a throwaway workspace, the rule-based LLM and no ASR, so it never calls a model or touches real data.
--demo fills it with sample meetings (one saved, one waiting for review) so every page shows real text.
Exit code 1 if any text is clipped or overlaps other text.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QRect, Qt  # noqa: E402
from PySide6.QtWidgets import QAbstractButton, QApplication, QLabel, QWidget  # noqa: E402

from app.config import load_config  # noqa: E402
from app.main import build_ctx  # noqa: E402
from app.ui import theme  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402


def demo(ctx) -> None:
    sm = ctx.sm
    p = sm.create_project("Atlas mobile app — offline sync and enterprise rollout")
    fx = ROOT / "tests" / "fixtures"
    s = sm.import_text((fx / "atlas_01_kickoff.txt").read_text(encoding="utf-8"), p["id"], p["name"],
                       "Kickoff: scope, owners and the very first milestones for the launch")
    sm.extract(s["id"])
    proposal = json.loads(sm._file(s["id"], "proposal.json").read_text(encoding="utf-8"))
    sm.approve(s["id"], proposal)
    s2 = sm.import_text((fx / "atlas_02_sprint_sync.txt").read_text(encoding="utf-8"), p["id"], p["name"],
                        "Sprint sync with a deliberately long title to test wrapping in every list")
    sm.extract(s2["id"])


def _text(w) -> str:
    return w.text() if hasattr(w, "text") else ""


def text_problems(win: QWidget) -> list[str]:
    """Visible labels/buttons whose text does not fit, and text widgets that overlap a sibling."""
    out = []
    items = [w for w in win.findChildren(QWidget)
             if isinstance(w, (QLabel, QAbstractButton)) and w.isVisibleTo(win) and w.width() > 0 and _text(w)]
    for w in items:
        fm, txt = w.fontMetrics(), _text(w)
        rich = isinstance(w, QLabel) and (w.textFormat() == Qt.RichText or (w.textFormat() == Qt.AutoText
                                                                              and "<" in txt))
        if isinstance(w, QLabel) and (w.wordWrap() or rich):
            need = w.heightForWidth(w.width()) if w.wordWrap() else w.sizeHint().height()
            if need > w.height() + 2:
                out.append(f"text cut off at the bottom ({need}px needed, {w.height()}px shown): {txt[:60]!r}")
            continue
        need = w.sizeHint().width() if isinstance(w, QAbstractButton) else \
            max(fm.horizontalAdvance(line) for line in txt.split("\n"))
        if need > w.width() + 1:
            out.append(f"text wider than its box ({need}px > {w.width()}px): {txt[:60]!r}")
    # only the part actually on screen: content scrolled under a bar is clipped by its scroll area, not overlapping
    rects = []
    for w in items:
        vis = w.visibleRegion().boundingRect()
        if not vis.isEmpty():
            rects.append((w, QRect(w.mapTo(win, vis.topLeft()), vis.size())))
    for i, (a, ra) in enumerate(rects):
        for b, rb in rects[i + 1:]:
            if a.isAncestorOf(b) or b.isAncestorOf(a):
                continue
            inter = ra.intersected(rb)
            if inter.width() > 2 and inter.height() > 2:
                out.append(f"overlap: {_text(a)[:40]!r} and {_text(b)[:40]!r}")
    return out


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    size = next((a.split("x") for a in sys.argv[2:] if a.replace("x", "", 1).isdigit()), ["1440", "900"])
    cfg = load_config()
    cfg["workspace"] = tempfile.mkdtemp(prefix="mg_snap_")
    cfg["llm"]["backend"] = "heuristic"
    cfg["asr"]["backend"] = "none"
    app = QApplication(sys.argv)
    theme.apply(app)
    ctx = build_ctx(cfg)
    if "--demo" in sys.argv:
        demo(ctx)
    win = MainWindow(ctx)
    win.resize(int(size[0]), int(size[1]))
    win.show()
    app.processEvents()
    for lst in (win.sessions, win.meetings, win.features):
        if lst.count() and lst.item(0).data(Qt.UserRole):
            lst.setCurrentRow(0)
    problems = 0
    for i, btn in win.nav.items():
        win.go(i)
        for _ in range(3):
            app.processEvents()
        name = btn.text().split("  ")[0].lower().replace(" ", "_")
        path = out / f"{i}_{name}_{size[0]}x{size[1]}.png"
        win.grab().save(str(path))
        found = text_problems(win)
        problems += len(found)
        print(path, "OK" if not found else f"{len(found)} problem(s)")
        for f in found:
            print("   -", f)
    win.close()
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
