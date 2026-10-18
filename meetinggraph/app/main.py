"""MeetingGraph desktop entry point:  python -m app.main"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox, QSystemTrayIcon

from .config import load_config
from .models.asr import BaseASR, NoASR, make_asr
from .models.llm import BaseLLM, HeuristicLLM, make_llm
from .pipeline.session import SessionManager
from .ui.common import dot_icon
from .ui.recorder_pill import RecorderPill
from .ui import theme
from .ui.main_window import MainWindow
from .workspace.paths import Layout
from .workspace.store import Store


@dataclass
class Ctx:
    cfg: dict
    layout: Layout
    store: Store
    asr: BaseASR
    llm: BaseLLM
    sm: SessionManager
    control: RecorderPill | None = None
    notes: list[str] = field(default_factory=list)
    llm_problem: str = ""


def build_ctx(cfg: dict) -> Ctx:
    layout = Layout(cfg["workspace"])
    store = Store(layout)
    store.init_workspace()
    notes = []
    try:
        asr = make_asr(cfg)
    except Exception as e:
        notes.append(f"ASR backend unavailable ({e}); using import/typed transcripts only.")
        asr = NoASR()
    llm_problem = ""
    try:
        llm = make_llm(cfg)
        ok, msg = llm.health()
        if not ok:
            llm_problem = msg
            llm = HeuristicLLM()
    except Exception as e:
        llm_problem = str(e)
        llm = HeuristicLLM()
    sm = SessionManager(cfg, layout, store, asr, llm)
    notes += sm.recover()
    return Ctx(cfg, layout, store, asr, llm, sm, notes=notes, llm_problem=llm_problem)


def main() -> int:
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    app = QApplication(sys.argv)
    app.setApplicationName("MeetingGraph")
    app.setQuitOnLastWindowClosed(False)
    theme.apply(app)
    ctx = build_ctx(cfg)

    tray = QSystemTrayIcon(dot_icon("#1e88e5"))
    tray.setToolTip("MeetingGraph — local only")

    def notify(title, msg):
        if QSystemTrayIcon.isSystemTrayAvailable():
            tray.showMessage(title, msg, QSystemTrayIcon.Information, 6000)

    win = MainWindow(ctx, notify)
    ctx.control = RecorderPill(ctx)
    ctx.control.finished.connect(win.on_recording_finished)
    tray.messageClicked.connect(lambda: (win.show(), win.raise_(), win.activateWindow()))

    menu = QMenu()
    for text, fn in (("Record", lambda: (win.showNormal(), win.open_recorder())),
                     ("Stop recording", lambda: ctx.control.stop()),
                     ("Open app", lambda: (win.show(), win.raise_())),
                     ("Quit", lambda: quit_app())):
        a = QAction(text, menu)
        a.triggered.connect(fn)
        menu.addAction(a)
    tray.setContextMenu(menu)
    tray.show()

    def quit_app():
        if ctx.control.recording:
            if QMessageBox.question(win, "Quit", "A meeting is being recorded. Stop and quit?") != QMessageBox.Yes:
                return
            ctx.control.stop()
        app.quit()

    def on_close(ev):
        if ctx.control.recording:
            if QMessageBox.question(win, "Quit", "A meeting is being recorded. Stop it and quit?") != QMessageBox.Yes:
                ev.ignore()
                return
            ctx.control.stop()
        ev.accept()
        app.quit()
    win.closeEvent = on_close
    win.show()
    if ctx.notes:
        QMessageBox.information(win, "Recovered after restart", "\n".join(ctx.notes))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
