"""One dialog to add a meeting: pick/create project, title, and how to add it."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDialog, QFileDialog, QFrame, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QRadioButton, QVBoxLayout,
                               QWidget)

NEW = "__new__"


class AddMeetingDialog(QDialog):
    """Result (after exec() == Accepted): self.result_data =
    {"project_id", "title", "mode": "paste"|"text_file"|"audio_file", "text", "path"}"""

    def __init__(self, ctx, parent=None, project_id: str | None = None):
        super().__init__(parent)
        self.ctx = ctx
        self.result_data: dict = {}
        self.setWindowTitle("Add a transcript")
        self.setMinimumWidth(620)
        lay = QVBoxLayout(self)
        lay.setSpacing(10)

        h = QLabel("Add a transcript")
        h.setObjectName("h1")
        lay.addWidget(h)
        sub = QLabel("The app reads the meeting, then shows you the suggested changes to review. "
                     "Nothing is saved to the project until you click “Save to project”.")
        sub.setObjectName("muted")
        sub.setWordWrap(True)
        lay.addWidget(sub)

        # 1. project
        lay.addWidget(self._step("1", "Which project is this meeting for?"))
        self.project = QComboBox()
        for p in ctx.store.list_projects():
            self.project.addItem(p["name"], p["id"])
        self.project.addItem("➕ Create a new project…", NEW)
        if project_id:
            i = self.project.findData(project_id)
            if i >= 0:
                self.project.setCurrentIndex(i)
        self.new_name = QLineEdit()
        self.new_name.setPlaceholderText("Name of the new project, e.g. Atlas")
        lay.addWidget(self.project)
        lay.addWidget(self.new_name)

        # 2. title
        lay.addWidget(self._step("2", "What is the meeting called?"))
        self.title = QLineEdit()
        self.title.setPlaceholderText("e.g. Kickoff, Sprint sync, Design review")
        lay.addWidget(self.title)

        # 3. how
        lay.addWidget(self._step("3", "How do you want to add it?"))
        self.group = QButtonGroup(self)
        self.r_paste = QRadioButton("Paste or type the conversation")
        self.r_file = QRadioButton("Open a transcript file (.txt)")
        asr_ok = getattr(ctx.asr, "available", False)
        self.r_audio = QRadioButton("Open an audio recording (.wav, .mp3)" +
                                    ("" if asr_ok else "  — needs speech recognition (README step 6b)"))
        self.r_audio.setEnabled(asr_ok)
        for i, r in enumerate((self.r_paste, self.r_file, self.r_audio)):
            self.group.addButton(r, i)
            lay.addWidget(r)
        self.r_file.setChecked(True)

        self.paste = QPlainTextEdit()
        self.paste.setPlaceholderText("Paste the meeting text here. One line per speaker turn works best.\n"
                                      "Example:\nPriya: We need offline sync for the beta.\n"
                                      "Arjun: That depends on login being done first.")
        self.paste.setMinimumHeight(150)
        lay.addWidget(self.paste)
        frow = QHBoxLayout()
        self.file_lbl = QLabel("No file chosen")
        self.file_lbl.setObjectName("muted")
        self.browse = QPushButton("Choose file…")
        frow.addWidget(self.browse)
        frow.addWidget(self.file_lbl, 1)
        self.file_row = QWidget()
        self.file_row.setLayout(frow)
        lay.addWidget(self.file_row)
        self.path = ""

        tip = QLabel("Tip: to try it quickly, open  tests\\fixtures\\atlas_01_kickoff.txt  from the app folder.")
        tip.setObjectName("muted")
        tip.setWordWrap(True)
        lay.addWidget(tip)

        btns = QHBoxLayout()
        btns.addStretch(1)
        cancel = QPushButton("Cancel")
        self.ok = QPushButton("Add transcript")
        self.ok.setObjectName("primary")
        btns.addWidget(cancel)
        btns.addWidget(self.ok)
        lay.addLayout(btns)

        cancel.clicked.connect(self.reject)
        self.ok.clicked.connect(self._accept)
        self.browse.clicked.connect(self._browse)
        self.group.idToggled.connect(lambda *_: self._sync())
        self.project.currentIndexChanged.connect(lambda *_: self._sync())
        self._sync()

    def _step(self, n: str, text: str) -> QLabel:
        lbl = QLabel(f"<b>{n}.</b> {text}")
        lbl.setObjectName("h2")
        return lbl

    def _sync(self):
        self.new_name.setVisible(self.project.currentData() == NEW)
        mode = self.group.checkedId()
        self.paste.setVisible(mode == 0)
        self.file_row.setVisible(mode in (1, 2))
        self.ok.setText("Add transcript")
        self.adjustSize()

    def _browse(self):
        if self.group.checkedId() == 2:
            p, _ = QFileDialog.getOpenFileName(self, "Choose audio", "", "Audio (*.wav *.mp3 *.m4a *.flac *.ogg)")
        else:
            p, _ = QFileDialog.getOpenFileName(self, "Choose transcript", "", "Text (*.txt *.md)")
        if p:
            self.path = p
            self.file_lbl.setText(Path(p).name)
            if not self.title.text().strip():
                self.title.setText(Path(p).stem.replace("_", " ").strip().title())

    def _accept(self):
        pid = self.project.currentData()
        if pid == NEW:
            name = self.new_name.text().strip()
            if not name:
                QMessageBox.information(self, "Project name", "Please type a name for the new project.")
                return
            pid = self.ctx.sm.create_project(name)["id"]
        mode = ["paste", "text_file", "audio_file"][self.group.checkedId()]
        title = self.title.text().strip() or f"Meeting {dt.date.today().isoformat()}"
        text = ""
        if mode == "paste":
            text = self.paste.toPlainText().strip()
            if not text:
                QMessageBox.information(self, "Nothing to add", "Paste or type the meeting text first.")
                return
        if mode in ("text_file", "audio_file") and not self.path:
            QMessageBox.information(self, "Choose a file", "Click “Choose file…” first.")
            return
        if mode == "text_file":
            text = Path(self.path).read_text(encoding="utf-8", errors="replace")
        self.result_data = {"project_id": pid, "title": title, "mode": mode, "text": text, "path": self.path}
        self.accept()


class RecordSetupDialog(QDialog):
    """Before recording: which project, and what the meeting is called."""

    def __init__(self, ctx, parent=None, project_id: str | None = None):
        super().__init__(parent)
        self.ctx = ctx
        self.result_data: dict = {}
        self.setWindowTitle("Record a meeting")
        self.setMinimumWidth(460)
        lay = QVBoxLayout(self)
        h = QLabel("Record a meeting")
        h.setObjectName("h1")
        lay.addWidget(h)
        sub = QLabel("The app records your microphone and the computer's sound (the other people in the call). "
                     "This window will minimize and a small recorder appears at the top of the screen. "
                     "Tell everyone in the meeting that you are recording.")
        sub.setWordWrap(True)
        sub.setObjectName("muted")
        lay.addWidget(sub)
        lay.addWidget(QLabel("<b>Project</b>"))
        self.project = QComboBox()
        for p in ctx.store.list_projects():
            self.project.addItem(p["name"], p["id"])
        self.project.addItem("➕ Create a new project…", NEW)
        if project_id and self.project.findData(project_id) >= 0:
            self.project.setCurrentIndex(self.project.findData(project_id))
        self.new_name = QLineEdit()
        self.new_name.setPlaceholderText("Name of the new project")
        lay.addWidget(self.project)
        lay.addWidget(self.new_name)
        lay.addWidget(QLabel("<b>Meeting name</b>"))
        self.title = QLineEdit(f"Meeting {dt.datetime.now().strftime('%b %d %H:%M')}")
        lay.addWidget(self.title)
        row = QHBoxLayout()
        row.addStretch(1)
        cancel = QPushButton("Cancel")
        ok = QPushButton("Open recorder")
        ok.setObjectName("record")
        row.addWidget(cancel)
        row.addWidget(ok)
        lay.addLayout(row)
        cancel.clicked.connect(self.reject)
        ok.clicked.connect(self._accept)
        self.project.currentIndexChanged.connect(
            lambda *_: self.new_name.setVisible(self.project.currentData() == NEW))
        self.new_name.setVisible(self.project.currentData() == NEW)

    def _accept(self):
        pid = self.project.currentData()
        if pid == NEW:
            name = self.new_name.text().strip()
            if not name:
                QMessageBox.information(self, "Project name", "Please type a name for the new project.")
                return
            pid = self.ctx.sm.create_project(name)["id"]
        self.result_data = {"project_id": pid, "title": self.title.text().strip() or "Meeting"}
        self.accept()
