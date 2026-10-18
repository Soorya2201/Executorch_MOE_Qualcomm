"""Small oval, always-on-top recording bar (like the Windows Game Bar capture widget).

● record (red)  ❚❚ pause  ■ stop   00:00  ✕
Pause and stop are disabled until recording starts. ✕ always asks whether to keep what was recorded.
Records the microphone AND the computer's sound (speaker loopback) as two tracks.
"""
from __future__ import annotations

from PySide6.QtCore import QPoint, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QGuiApplication, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMessageBox, QProgressBar, QToolButton, QVBoxLayout, QWidget

from ..capture import recorder

PILL_QSS = """
QWidget { color: #f0f3f6; font-family: 'Segoe UI'; }
QToolButton { border: none; border-radius: 22px; min-width: 44px; min-height: 44px; max-width: 44px; max-height: 44px;
              font-size: 15pt; background: #3a4048; color: #f0f3f6; }
QToolButton:hover { background: #4a515b; }
QToolButton:disabled { background: #2b3036; color: #5b636d; }
QToolButton#rec { background: #e5484d; color: white; font-size: 18pt; }
QToolButton#rec:hover { background: #f2555a; }
QToolButton#rec:disabled { background: #6b2c2f; color: #c9a0a2; }
QToolButton#close { min-width: 30px; min-height: 30px; max-width: 30px; max-height: 30px; border-radius: 15px;
                    font-size: 11pt; background: transparent; color: #c9d1d9; }
QToolButton#close:hover { background: #e5484d; color: white; }
QLabel#time { font-size: 15pt; font-weight: 600; }
QLabel#state { font-size: 9pt; color: #aab4be; }
QProgressBar { background: #2b3036; border: none; border-radius: 2px; max-height: 4px; }
QProgressBar::chunk { background: #3fb950; border-radius: 2px; }
"""


class RecorderPill(QWidget):
    finished = Signal(str, bool)      # (session id or "", keep the recording?)

    def __init__(self, ctx):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.ctx = ctx
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet(PILL_QSS)
        self.setFixedSize(420, 76)
        self.rec: recorder.DualRecorder | None = None
        self.sid: str | None = None
        self.project_id: str | None = None
        self.project_name = ""
        self.title = "Meeting"
        self._drag: QPoint | None = None
        self.live = None
        self._blink = False

        lay = QHBoxLayout(self)
        lay.setContentsMargins(20, 10, 14, 10)
        lay.setSpacing(10)
        self.b_rec = QToolButton(text="●")
        self.b_rec.setObjectName("rec")
        self.b_rec.setToolTip("Record")
        self.b_pause = QToolButton(text="❚❚")
        self.b_pause.setToolTip("Pause")
        self.b_stop = QToolButton(text="■")
        self.b_stop.setToolTip("Stop and make the transcript")
        for b in (self.b_rec, self.b_pause, self.b_stop):
            b.setCursor(Qt.PointingHandCursor)
            lay.addWidget(b)
        info = QVBoxLayout()
        info.setSpacing(2)
        self.time = QLabel("00:00")
        self.time.setObjectName("time")
        self.state = QLabel("Ready to record")
        self.state.setObjectName("state")
        self.level = QProgressBar()
        self.level.setRange(0, 100)
        self.level.setTextVisible(False)
        info.addWidget(self.time)
        info.addWidget(self.state)
        info.addWidget(self.level)
        lay.addLayout(info, 1)
        self.b_close = QToolButton(text="✕")
        self.b_close.setObjectName("close")
        self.b_close.setToolTip("Close")
        lay.addWidget(self.b_close, 0, Qt.AlignTop)

        self.b_rec.clicked.connect(self._on_record)
        self.b_pause.clicked.connect(self._on_pause)
        self.b_stop.clicked.connect(lambda: self._finish(True))
        self.b_close.clicked.connect(self._on_close)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._set_buttons()

    # ---------- dialogs ----------
    # Dialogs get NO parent: a child of the pill would inherit the pill's white-text style and
    # show white text on a white dialog (invisible). Parentless dialogs use the app's normal theme.
    @staticmethod
    def _box(icon, title: str, text: str, info: str = "") -> QMessageBox:
        box = QMessageBox(icon, title, text)
        if info:
            box.setInformativeText(info)
        box.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        return box

    def _message(self, icon, title: str, text: str):
        box = self._box(icon, title, text)
        box.addButton("OK", QMessageBox.AcceptRole)
        box.exec()

    # ---------- look ----------
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(r, r.height() / 2, r.height() / 2)      # oval / pill shape
        p.fillPath(path, QBrush(QColor(28, 31, 36, 240)))
        p.setPen(QPen(QColor(255, 255, 255, 40), 1))
        p.drawPath(path)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._drag = ev.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, ev):
        if self._drag is not None and ev.buttons() & Qt.LeftButton:
            self.move(ev.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, _ev):
        self._drag = None

    # ---------- public ----------
    @property
    def recording(self) -> bool:
        return self.rec is not None

    def begin(self, project_id: str, project_name: str, title: str):
        """Show the pill, ready to record (nothing is recorded until ● is pressed)."""
        self.project_id, self.project_name, self.title = project_id, project_name, title or "Meeting"
        self.sid, self.rec = None, None
        self.time.setText("00:00")
        self.state.setText(f"Ready · {self.project_name}")
        self.level.setValue(0)
        self._set_buttons()
        scr = QGuiApplication.primaryScreen().availableGeometry()
        self.move(scr.center().x() - self.width() // 2, scr.top() + 24)
        self.show()
        self.raise_()

    def stop(self):
        """Used by the tray menu / app quit: stop and keep the recording."""
        if self.isVisible():
            self._finish(True)

    # ---------- buttons ----------
    def _set_buttons(self):
        started = self.rec is not None
        paused = started and self.rec.is_paused
        self.b_rec.setEnabled(not started or paused)       # ● starts, or resumes after a pause
        self.b_rec.setToolTip("Resume" if paused else "Record")
        self.b_pause.setEnabled(started and not paused)
        self.b_stop.setEnabled(started)

    def _on_record(self):
        if self.rec is not None and self.rec.is_paused:
            self.rec.resume()
            self._set_buttons()
            return
        devs = recorder.list_devices()
        if devs.get("error"):
            self._message(QMessageBox.Warning, "Can't record", f"Recording isn't available on this computer:\n{devs['error']}")
            return
        mic, spk = devs.get("default_mic"), devs.get("default_speaker")
        s = self.ctx.sm.create("recording", self.project_id, self.project_name, self.title)
        mic_path, loop_path = self.ctx.sm.recording_paths(s["id"])
        try:
            rec = recorder.DualRecorder(mic, spk, mic_path, loop_path)
            warnings = rec.start()
        except Exception as e:
            self.ctx.sm.discard(s["id"])
            self._message(QMessageBox.Warning, "Can't record", str(e))
            return
        self.rec, self.sid = rec, s["id"]
        try:
            self.live = self.ctx.sm.start_live(s["id"], rec)      # transcript written every 30 s
        except Exception:
            self.live = None                                       # falls back to transcribing after Stop
        self.state.setToolTip("\n".join(warnings) or "Recording your microphone and the computer's sound")
        self.timer.start(250)
        self._set_buttons()
        self._tick()

    def _on_pause(self):
        if self.rec:
            self.rec.pause()
            self._set_buttons()
            self._tick()

    def _on_close(self):
        box = self._box(QMessageBox.Question, "Close recorder", "Save the transcript recorded so far?",
                        "Yes: stop and turn what was recorded into a transcript.\n"
                        "No: throw the recording away.")
        yes = box.addButton("Yes, save it", QMessageBox.YesRole)
        no = box.addButton("No, discard", QMessageBox.NoRole)
        box.addButton("Keep recording", QMessageBox.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is yes:
            if self.rec is None:
                self._message(QMessageBox.Information, "Nothing recorded",
                              "Nothing has been recorded yet, so there is nothing to save.")
                self._finish(False)
            else:
                self._finish(True)
        elif clicked is no:
            self._finish(False)

    def _finish(self, keep: bool):
        self.timer.stop()
        sid = self.sid
        if self.rec is not None:
            offsets, notes = self.rec.stop()
            self.rec = None
            if keep and sid:
                self.ctx.sm.finish_recording(sid, offsets, notes)
            elif sid:
                try:
                    self.ctx.sm.stop_live(sid)
                    self.ctx.sm.discard(sid)
                except Exception:
                    pass
        self.sid = None
        self.live = None
        self.hide()
        self.finished.emit((sid or "") if keep else "", bool(keep and sid))

    # ---------- live display ----------
    def _tick(self):
        if not self.rec:
            return
        e = int(self.rec.elapsed())
        self.time.setText(f"{e // 3600:d}:{(e % 3600) // 60:02d}:{e % 60:02d}" if e >= 3600 else f"{e // 60:02d}:{e % 60:02d}")
        if self.rec.is_paused:
            self.state.setText("Paused · press ● to continue")
            self.level.setValue(0)
        else:
            self._blink = not self._blink
            dot = "●" if self._blink else "○"
            health = self.rec.health()
            st = self.live.stats if self.live else None
            if health:
                self.state.setText(f"⚠ {health[0][:40]}")
            elif st and (st["lines"] or st["batches"]):
                notes = f" · notes {st['notes_done']}" if st["notes_done"] else ""
                self.state.setText(f"{dot} {st['lines']} lines{notes}")
                self.state.setToolTip(f"Latest: {st['last']}" if st["last"] else "Listening…")
            elif st:
                self.state.setText(f"{dot} Recording · text in ~30 s")
            else:
                self.state.setText(f"{dot} Recording · mic + PC audio")
            lv = self.rec.levels()
            self.level.setValue(int(max(lv.values() or [0]) * 100))
        if e > 60 * int(self.ctx.cfg["audio"].get("max_minutes", 60)):
            self._finish(True)
