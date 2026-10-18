"""MeetingGraph desktop entry point:  python -m app.main"""
from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox, QSystemTrayIcon

from .config import load_config
from .models.asr import BaseASR, NoASR, make_asr
from .models.llm import BaseLLM, HeuristicLLM, make_llm
from .pipeline.session import SessionManager
from .ui.common import dot_icon
from .ui.control_window import ControlWindow
from .ui import theme
from .ui.main_window import MainWindow
from .workspace.paths import Layout
from .workspace.store import Store

INSTANCE_KEY = "MeetingGraph-single-instance"


@dataclass
class Ctx:
    cfg: dict
    layout: Layout
    store: Store
    asr: BaseASR
    llm: BaseLLM
    sm: SessionManager
    control: ControlWindow | None = None
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


def _install_crash_log(log_dir: Path) -> None:
    """Log unexpected errors to workspace/logs and tell the user, instead of the window silently dying."""
    log_dir.mkdir(parents=True, exist_ok=True)

    def hook(etype, value, tb):
        text = "".join(traceback.format_exception(etype, value, tb))
        path = log_dir / f"error-{time.strftime('%Y%m%d')}.log"
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')}\n{text}")
        except OSError:
            pass
        sys.__excepthook__(etype, value, tb)
        if QApplication.instance():
            QMessageBox.warning(None, "Something went wrong",
                                f"{etype.__name__}: {value}\n\nThe app keeps running. Details were saved to\n{path}")
    sys.excepthook = hook


def _already_running() -> bool:
    """If another copy is open, ask it to come to the front and return True."""
    sock = QLocalSocket()
    sock.connectToServer(INSTANCE_KEY)
    if sock.waitForConnected(300):
        sock.write(b"show")
        sock.waitForBytesWritten(300)
        sock.disconnectFromServer()
        return True
    return False


def main() -> int:
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    app = QApplication(sys.argv)
    app.setApplicationName("MeetingGraph")
    app.setQuitOnLastWindowClosed(False)
    if _already_running():
        return 0
    _install_crash_log(Path(cfg["workspace"]) / "logs")
    theme.apply(app)
    ctx = build_ctx(cfg)

    tray = QSystemTrayIcon(dot_icon(theme.ACCENT))
    tray.setToolTip("MeetingGraph — local only")

    def notify(title, msg):
        if QSystemTrayIcon.isSystemTrayAvailable():
            tray.showMessage(title, msg, QSystemTrayIcon.Information, 6000)

    win = MainWindow(ctx, notify)
    ctx.control = ControlWindow(ctx)
    ctx.control.open_app.connect(win.bring_to_front)
    ctx.control.stopped.connect(lambda sid: (win.select_session(sid), win.process(sid)))
    tray.messageClicked.connect(win.bring_to_front)
    tray.activated.connect(lambda reason: win.bring_to_front()
                           if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick) else None)

    server = QLocalServer(app)
    QLocalServer.removeServer(INSTANCE_KEY)  # clear a stale name left by a crash
    server.newConnection.connect(lambda: (server.nextPendingConnection(), win.bring_to_front()))
    server.listen(INSTANCE_KEY)

    menu = QMenu()
    for text, fn in (("Open MeetingGraph", win.bring_to_front),
                     ("Meeting control", lambda: (ctx.control.refresh() if not ctx.control.recording else None,
                                                   ctx.control.show(), ctx.control.raise_())),
                     ("Stop recording", lambda: ctx.control.stop()),
                     ("Quit", lambda: quit_app())):
        a = QAction(text, menu)
        a.triggered.connect(fn)
        menu.addAction(a)
    tray.setContextMenu(menu)
    tray.show()

    def busy_reason() -> str:
        return "A meeting is being recorded. Stop it and quit?" if ctx.control.recording else ""

    def shutdown():
        if ctx.control.recording:
            ctx.control.stop()

    def quit_app():
        why = busy_reason()
        if why and QMessageBox.question(win, "Quit", why) != QMessageBox.Yes:
            return
        shutdown()
        app.quit()

    def on_close(ev):
        why = busy_reason()
        if why and QMessageBox.question(win, "Quit", why) != QMessageBox.Yes:
            ev.ignore()
            return
        shutdown()
        ev.accept()
        app.quit()
    win.closeEvent = on_close
    win.show()
    if ctx.notes:
        QTimer.singleShot(0, lambda: QMessageBox.information(win, "Recovered after restart", "\n".join(ctx.notes)))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
