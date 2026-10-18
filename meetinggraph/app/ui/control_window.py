"""Small always-on-top meeting control. Records only after an explicit Start click."""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QComboBox, QGridLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                               QMessageBox, QProgressBar, QPushButton, QVBoxLayout, QWidget)

from ..capture import recorder


class ControlWindow(QWidget):
    stopped = Signal(str)          # session id
    open_app = Signal()

    def __init__(self, ctx):
        super().__init__(None, Qt.Window | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.ctx = ctx
        self.rec: recorder.DualRecorder | None = None
        self.sid: str | None = None
        self.setWindowTitle("MeetingGraph · meeting control")
        self.setMinimumWidth(360)
        lay = QVBoxLayout(self)

        g = QGridLayout()
        self.project = QComboBox()
        self.title = QLineEdit("Meeting")
        self.mic = QComboBox()
        self.spk = QComboBox()
        g.addWidget(QLabel("Project"), 0, 0)
        g.addWidget(self.project, 0, 1)
        g.addWidget(QLabel("Title"), 1, 0)
        g.addWidget(self.title, 1, 1)
        g.addWidget(QLabel("Microphone (you)"), 2, 0)
        g.addWidget(self.mic, 2, 1)
        g.addWidget(QLabel("Output (remote)"), 3, 0)
        g.addWidget(self.spk, 3, 1)
        self.mic_lvl = QProgressBar()
        self.spk_lvl = QProgressBar()
        for b in (self.mic_lvl, self.spk_lvl):
            b.setRange(0, 100)
            b.setTextVisible(False)
            b.setFixedHeight(8)
        g.addWidget(QLabel("Mic level"), 4, 0)
        g.addWidget(self.mic_lvl, 4, 1)
        g.addWidget(QLabel("Output level"), 5, 0)
        g.addWidget(self.spk_lvl, 5, 1)
        lay.addLayout(g)

        self.state = QLabel("Not recording")
        self.state.setStyleSheet("font-weight:600;")
        self.health = QLabel("")
        self.health.setWordWrap(True)
        self.health.setStyleSheet("color:#b26a00;")
        lay.addWidget(self.state)
        lay.addWidget(self.health)

        row = QHBoxLayout()
        self.btn = QPushButton("● Start recording")
        self.btn.setObjectName("primary")
        b_open = QPushButton("Open app")
        b_hide = QPushButton("Hide")
        row.addWidget(self.btn)
        row.addWidget(b_open)
        row.addWidget(b_hide)
        lay.addLayout(row)
        tip = QLabel("Tell participants you are recording. Hide this window before screen sharing; "
                     "Stop is also in the tray menu.")
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#666;font-size:11px;")
        lay.addWidget(tip)

        self.btn.clicked.connect(self.toggle)
        b_open.clicked.connect(self.open_app.emit)
        b_hide.clicked.connect(self.hide)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.refresh()

    def refresh(self):
        cur = self.project.currentData()
        self.project.clear()
        for p in self.ctx.store.list_projects():
            self.project.addItem(p["name"], p["id"])
        self.project.addItem("➕ Create a new project…", None)
        i = self.project.findData(cur) if cur else 0
        self.project.setCurrentIndex(max(i, 0))
        devs = recorder.list_devices()
        self.mic.clear()
        self.spk.clear()
        self.mic.addItem("(none)", None)
        self.spk.addItem("(none)", None)
        for d in devs["mics"]:
            self.mic.addItem(d["name"], d["id"])
        for d in devs["speakers"]:
            self.spk.addItem(d["name"], d["id"])
        if devs.get("default_mic"):
            self.mic.setCurrentIndex(max(self.mic.findData(devs["default_mic"]), 0))
        if devs.get("default_speaker"):
            self.spk.setCurrentIndex(max(self.spk.findData(devs["default_speaker"]), 0))
        if devs.get("error"):
            self.health.setText(f"Recording is not available on this computer ({devs['error']}). "
                                "Use “Add a meeting” in the main window instead.")
            self.btn.setEnabled(False)
        elif not getattr(self.ctx.asr, "available", False):
            self.health.setText("Note: speech recognition is off, so a recording can't be turned into text yet "
                                "(README step 6b). You can still test that the level bars move.")

    def select_project(self, project_id: str):
        i = self.project.findData(project_id)
        if i >= 0:
            self.project.setCurrentIndex(i)

    @property
    def recording(self) -> bool:
        return self.rec is not None

    def toggle(self):
        if self.recording:
            self.stop()
        else:
            self.start()

    def start(self):
        pid = self.project.currentData()
        name = self.project.currentText()
        if pid is None:
            name, ok = QInputDialog.getText(self, "New project", "Project name:")
            if not ok or not name.strip():
                return
            p = self.ctx.sm.create_project(name.strip())
            pid, name = p["id"], p["name"]
            self.refresh()
            self.select_project(pid)
        s = self.ctx.sm.create("recording", pid, name, self.title.text())
        mic_path, loop_path = self.ctx.sm.recording_paths(s["id"])
        try:
            self.rec = recorder.DualRecorder(self.mic.currentData(), self.spk.currentData(), mic_path, loop_path)
            warnings = self.rec.start()
        except Exception as e:
            self.rec = None
            self.ctx.sm.discard(s["id"])
            QMessageBox.warning(self, "Cannot record", str(e))
            return
        self.sid = s["id"]
        self.health.setText("\n".join(warnings))
        self.btn.setText("■ Stop recording")
        for w in (self.project, self.title, self.mic, self.spk):
            w.setEnabled(False)
        self.timer.start(200)

    def stop(self):
        if not self.rec:
            return
        self.timer.stop()
        offsets, notes = self.rec.stop()
        self.rec = None
        sid = self.sid
        self.sid = None
        self.ctx.sm.finish_recording(sid, offsets, notes)
        self.btn.setText("● Start recording")
        self.state.setText("Done. The meeting is being processed on this laptop — see the main window.")
        self.mic_lvl.setValue(0)
        self.spk_lvl.setValue(0)
        for w in (self.project, self.title, self.mic, self.spk):
            w.setEnabled(True)
        self.stopped.emit(sid)

    def _tick(self):
        if not self.rec:
            return
        lv = self.rec.levels()
        self.mic_lvl.setValue(int(lv.get("mic", 0) * 100))
        self.spk_lvl.setValue(int(lv.get("loopback", 0) * 100))
        e = int(self.rec.elapsed())
        self.state.setText(f"● Recording  {e // 60:02d}:{e % 60:02d}")
        self.health.setText("\n".join(self.rec.health()))
        if e > 60 * int(self.ctx.cfg["audio"].get("max_minutes", 60)):
            self.stop()
            QMessageBox.information(self, "Limit reached", "Stopped at the configured maximum meeting length.")
