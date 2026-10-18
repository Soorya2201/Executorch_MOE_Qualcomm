"""Main window: projects | To review / Meetings / Features / Map | Ask. A banner always says what to do next."""
from __future__ import annotations

from PySide6.QtCore import Qt, QThreadPool, QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QMainWindow, QMessageBox, QPushButton, QSplitter, QStatusBar, QTabWidget,
                               QTextBrowser, QVBoxLayout, QWidget)

from ..pipeline.query import answer
from ..workspace.graph import build_graph
from .add_meeting import AddMeetingDialog, RecordSetupDialog
from .common import STAGE_COLOR, STAGE_TEXT, STATUS_COLOR, Job, dot_icon, esc
from .graph_view import GraphView
from .pages import feature_html, meeting_html
from .components import Toast
from .review import ReviewPanel
from . import theme

BUSY = ("recording", "recorded", "transcribing", "summarizing", "saving")
TAB_REVIEW, TAB_MEETINGS, TAB_FEATURES, TAB_MAP = range(4)


class MainWindow(QMainWindow):
    def __init__(self, ctx, notify=None):
        super().__init__()
        self.ctx = ctx
        self.notify = notify or (lambda title, msg: None)
        self.pool = QThreadPool.globalInstance()
        self.model_pool = QThreadPool()          # one model job at a time (one NPU)
        self.model_pool.setMaxThreadCount(1)
        self.project_id: str | None = None
        self.busy: set[str] = set()
        self._banner_action = None
        self.setWindowTitle("MeetingGraph — private project memory")
        self.resize(1440, 900)

        central = QWidget()
        cl = QVBoxLayout(central)
        cl.setContentsMargins(10, 10, 10, 4)
        self.setCentralWidget(central)

        # banner: what to do next
        self.banner = QFrame()
        self.banner.setObjectName("banner")
        bl = QHBoxLayout(self.banner)
        self.banner_text = QLabel()
        self.banner_text.setWordWrap(True)
        self.banner_text.setTextFormat(Qt.RichText)
        self.banner_btn = QPushButton("Show me")
        self.banner_btn.setObjectName("primary")
        bl.addWidget(self.banner_text, 1)
        bl.addWidget(self.banner_btn)
        cl.addWidget(self.banner)

        root = QSplitter()
        cl.addWidget(root, 1)

        # left: projects + main actions
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 6, 0)
        self.b_add = QPushButton("＋  Add a transcript")
        self.b_add.setObjectName("primary")
        self.b_add.setMinimumHeight(40)
        ll.addWidget(self.b_add)
        self.b_record = QPushButton("●  Record")
        self.b_record.setObjectName("record")
        self.b_record.setMinimumHeight(40)
        self.b_record.setToolTip("Record a meeting: your microphone + the computer's sound")
        ll.addWidget(self.b_record)
        lbl = QLabel("Projects")
        lbl.setObjectName("h2")
        ll.addSpacing(8)
        ll.addWidget(lbl)
        self.projects = QListWidget()
        ll.addWidget(self.projects, 1)
        self.b_new = QPushButton("＋  New project")
        ll.addWidget(self.b_new)
        root.addWidget(left)

        # center: tabs
        self.tabs = QTabWidget()
        rev = QWidget()
        rl = QHBoxLayout(rev)
        rl.setContentsMargins(0, 0, 0, 0)
        rsplit = QSplitter()
        rl.addWidget(rsplit)
        sl = QWidget()
        sll = QVBoxLayout(sl)
        sll.setContentsMargins(6, 6, 0, 6)
        t = QLabel("Meetings waiting for you")
        t.setObjectName("h2")
        sll.addWidget(t)
        self.sessions = QListWidget()
        self.sessions.setWordWrap(True)
        self.sessions.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sll.addWidget(self.sessions, 1)
        self.review = ReviewPanel(ctx)
        rsplit.addWidget(sl)
        rsplit.addWidget(self.review)
        rsplit.setSizes([260, 900])
        self.tabs.addTab(rev, "To review")

        tl = QWidget()
        tll = QHBoxLayout(tl)
        tsplit = QSplitter()
        tll.addWidget(tsplit)
        self.meetings = QListWidget()
        self.meeting_view = QTextBrowser()
        self.meeting_view.setOpenLinks(False)
        tsplit.addWidget(self.meetings)
        tsplit.addWidget(self.meeting_view)
        tsplit.setSizes([260, 800])
        self.tabs.addTab(tl, "Meetings")

        fw = QWidget()
        fl = QHBoxLayout(fw)
        fsplit = QSplitter()
        fl.addWidget(fsplit)
        fleft = QWidget()
        fll = QVBoxLayout(fleft)
        fll.setContentsMargins(0, 0, 0, 0)
        self.feature_filter = QLineEdit()
        self.feature_filter.setPlaceholderText("Search features…")
        self.features = QListWidget()
        fll.addWidget(self.feature_filter)
        fll.addWidget(self.features, 1)
        self.feature_view = QTextBrowser()
        self.feature_view.setOpenLinks(False)
        fsplit.addWidget(fleft)
        fsplit.addWidget(self.feature_view)
        fsplit.setSizes([260, 800])
        self.tabs.addTab(fw, "Features")

        self.graph = GraphView()
        self.tabs.addTab(self.graph, "Map")
        root.addWidget(self.tabs)

        # right: ask
        right = QWidget()
        rr = QVBoxLayout(right)
        rr.setContentsMargins(6, 0, 0, 0)
        self.chat_title = QLabel("Ask a question")
        self.chat_title.setObjectName("h2")
        self.pending = QLabel("")
        self.pending.setWordWrap(True)
        self.pending.setObjectName("warn")
        self.chat = QTextBrowser()
        self.chat.setOpenLinks(False)
        self.ask = QLineEdit()
        self.ask.setPlaceholderText("e.g. What changed in offline sync?")
        b_ask = QPushButton("Ask")
        b_ask.setObjectName("primary")
        row = QHBoxLayout()
        row.addWidget(self.ask, 1)
        row.addWidget(b_ask)
        rr.addWidget(self.chat_title)
        rr.addWidget(self.pending)
        rr.addWidget(self.chat, 1)
        rr.addLayout(row)
        note = QLabel("Answers come only from this project's saved meetings and features, with sources.")
        note.setWordWrap(True)
        note.setObjectName("muted")
        rr.addWidget(note)
        root.addWidget(right)
        root.setSizes([230, 890, 320])

        sb = QStatusBar()
        self.setStatusBar(sb)
        self.models_lbl = QLabel(f"AI model: {ctx.llm.label}   ·   Speech: {ctx.asr.label}   ·   "
                                 "Nothing leaves this laptop")
        sb.addPermanentWidget(self.models_lbl)

        m = self.menuBar().addMenu("&Workspace")
        for text, fn in (("Open workspace folder", lambda: QDesktopServices.openUrl(
                              QUrl.fromLocalFile(str(ctx.layout.root)))),
                         ("Show master log of this project", self.show_log),
                         ("Rebuild search index", self.rebuild_index)):
            a = QAction(text, self)
            a.triggered.connect(fn)
            m.addAction(a)

        # wiring
        self.projects.currentItemChanged.connect(self._on_project)
        self.sessions.currentItemChanged.connect(self._on_session)
        self.meetings.currentItemChanged.connect(self._on_meeting)
        self.features.currentItemChanged.connect(self._on_feature)
        self.feature_filter.textChanged.connect(self.refresh_features)
        for v in (self.meeting_view, self.feature_view, self.chat):
            v.anchorClicked.connect(self._open_link)
        self.graph.open_node.connect(self._open_from_map)
        self.nav_back: list = []                     # where Alt+Left returns to
        self.toast = Toast(self)
        keys = [QKeySequence(Qt.ALT | Qt.Key_Left)]          # one shortcut: two with the same key would cancel out
        keys += [k for k in QKeySequence.keyBindings(QKeySequence.Back) if k not in keys]
        self._back_sc = QShortcut(self)
        self._back_sc.setKeys(keys)
        self._back_sc.setContext(Qt.WindowShortcut)
        self._back_sc.activated.connect(self.go_back)
        self.graph.go_meetings.connect(lambda: self.tabs.setCurrentIndex(TAB_MEETINGS))
        self.graph.set_context(ctx.store, ctx.layout.root)
        self.review.request.connect(self.handle_request)
        self.b_add.clicked.connect(self.add_meeting)
        self.b_new.clicked.connect(self.new_project)
        self.b_record.clicked.connect(self.open_recorder)
        self.banner_btn.clicked.connect(lambda: self._banner_action and self._banner_action())
        b_ask.clicked.connect(self.ask_question)
        self.ask.returnPressed.connect(self.ask_question)

        theme.signals.changed.connect(self._on_theme)

        self.poll = QTimer(self)
        self.poll.timeout.connect(self.refresh_sessions)
        self.poll.start(1500)
        self.refresh_all()

    def _on_theme(self, _mode: str):
        """Light/dark switched: re-render the views that contain generated HTML or painted items."""
        self._last_sig = None
        self.refresh_all()
        if self.review.sid:
            self._load_review(self.review.sid)

    # ---------- banner ----------
    def update_banner(self, sessions=None):
        sessions = sessions if sessions is not None else self.ctx.sm.list()
        projects = self.ctx.store.list_projects()
        warn = False
        llm_problem = getattr(self.ctx, "llm_problem", "")
        busy = [s for s in sessions if s.get("stage") in BUSY]
        ready = [s for s in sessions if s.get("stage") == "ready"]
        bad = [s for s in sessions if s.get("stage") in ("failed", "cleanup_failed", "transcribed")]
        if ready:
            s = ready[0]
            text = (f"<b>“{esc(s['title'])}” is ready for you.</b> Check the suggested changes, answer any yellow "
                    "questions, then click <b>Save to project</b>.")
            btn, act = "Review it", (lambda sid=s["id"]: self.select_session(sid))
        elif bad:
            s = bad[0]
            warn = True
            text = f"<b>“{esc(s['title'])}” needs your attention.</b> Open it to see what to do."
            btn, act = "Open it", (lambda sid=s["id"]: self.select_session(sid))
        elif busy:
            s = busy[0]
            text = (f"<b>Working on “{esc(s['title'])}”…</b> {STAGE_TEXT.get(s['stage'], '')}. "
                    "It will show up here when it's ready to review.")
            btn, act = "See progress", (lambda sid=s["id"]: self.select_session(sid))
        elif not projects:
            text = ("<b>Start here:</b> click <b>● Record</b> to record a meeting, or <b>＋ Add a transcript</b> to "
                    "paste or open one. To try it, add <i>tests\\fixtures\\atlas_01_kickoff.txt</i>.")
            btn, act = "Add a transcript", self.add_meeting
        else:
            name = next((p["name"] for p in projects if p["id"] == self.project_id), projects[0]["name"])
            text = (f"<b>All caught up.</b> Record or add the next meeting for “{esc(name)}”, look through Meetings, "
                    "Features and the Map, or ask a question on the right.")
            btn, act = "Add a transcript", self.add_meeting
        if llm_problem:
            text += f"<br><span style='color:{theme.c('warning')}'>AI model not available: {esc(llm_problem)} " \
                    "Using simple rules until it is fixed.</span>"
        self.banner.setObjectName("bannerWarn" if warn else "banner")
        self.banner.style().unpolish(self.banner)
        self.banner.style().polish(self.banner)
        self.banner_text.setText(text)
        self.banner_btn.setText(btn)
        self._banner_action = act

    # ---------- refresh ----------
    def refresh_all(self):
        self.refresh_projects()
        self._last_sig = None
        self.refresh_sessions()

    def refresh_projects(self, select: str | None = None):
        cur = select or self.project_id
        self.projects.blockSignals(True)
        self.projects.clear()
        for p in self.ctx.store.list_projects():
            n = len(p.get("meetings") or [])
            it = QListWidgetItem(f"{p['name']}\n{n} meeting{'s' if n != 1 else ''} saved")
            it.setData(Qt.UserRole, p["id"])
            self.projects.addItem(it)
            if p["id"] == cur:
                self.projects.setCurrentItem(it)
        self.projects.blockSignals(False)
        if self.projects.currentItem() is None and self.projects.count():
            self.projects.setCurrentRow(0)
        else:
            self._on_project(self.projects.currentItem(), None)

    def models_busy(self) -> bool:
        """True while the AI is answering, transcribing or making notes (the Map waits with its layout)."""
        return self.model_pool.activeThreadCount() > 0 or bool(self.ctx.sm.live)

    def refresh_sessions(self):
        self.graph.set_busy(self.models_busy())
        sessions = self.ctx.sm.list()
        cur = self.review.sid
        sig = [(s["id"], s.get("stage"), s.get("updated_at")) for s in sessions]
        if getattr(self, "_last_sig", None) == sig:
            return
        changed_current = cur and any(s["id"] == cur and (s.get("stage"), s.get("updated_at")) !=
                                      getattr(self, "_cur_state", None) for s in sessions)
        self._last_sig = sig
        self.sessions.blockSignals(True)
        self.sessions.clear()
        ready = 0
        for s in sessions:
            stage = s.get("stage", "")
            ready += stage == "ready"
            it = QListWidgetItem(dot_icon(STAGE_COLOR.get(stage, STATUS_COLOR["proposed"]), 14),
                                 f"{s['title']}\n{s.get('project_name', '')} · {STAGE_TEXT.get(stage, stage)}")
            it.setData(Qt.UserRole, s["id"])
            self.sessions.addItem(it)
            if s["id"] == cur:
                self.sessions.setCurrentItem(it)
        if not sessions:
            it = QListWidgetItem("Nothing to review.\nUse “● Record” or\n“＋ Add a transcript”.")
            it.setFlags(Qt.NoItemFlags)
            self.sessions.addItem(it)
        self.sessions.blockSignals(False)
        self.tabs.setTabText(TAB_REVIEW, f"To review ({ready})" if ready else "To review")
        if cur and not any(s["id"] == cur for s in sessions):
            self.review.clear()
        elif changed_current:
            self._load_review(cur)
        n = sum(1 for s in sessions if s.get("project_id") == self.project_id)
        self.pending.setText(f"{n} meeting(s) for this project are not saved yet, so answers don't include them."
                             if n else "")
        self.update_banner(sessions)

    def _on_project(self, cur, _prev):
        if cur is None:
            self.project_id = None
            self.meetings.clear()
            self.features.clear()
            self.meeting_view.clear()
            self.feature_view.clear()
            self.graph.load_project(None)
            self.chat_title.setText("Ask a question")
            return
        self.project_id = cur.data(Qt.UserRole)
        name = cur.text().split("\n")[0]
        self.chat_title.setText(f"Ask about {name}")
        self.chat.clear()
        self.refresh_meetings()
        self.refresh_features()
        self.refresh_graph()
        self._last_sig = None
        self.refresh_sessions()

    def refresh_meetings(self):
        self.meetings.clear()
        self.meeting_view.clear()
        if not self.project_id:
            return
        ms = list(reversed(self.ctx.store.meetings(self.project_id)))
        for m in ms:
            it = QListWidgetItem(f"{m.get('title', '')}\n{m.get('date', '')}")
            it.setData(Qt.UserRole, m["id"])
            self.meetings.addItem(it)
        if not ms:
            self.meeting_view.setHtml(f"<p style='color:{theme.c('secondary')}'>No saved meetings yet. Add one and save it from "
                                      "“To review”.</p>")

    def refresh_features(self, *_):
        self.features.clear()
        if not self.project_id:
            return
        q = self.feature_filter.text().lower().strip()
        feats = sorted(self.ctx.store.features(self.project_id),
                       key=lambda f: (f.get("status") == "archived", f["title"].lower()))
        for f in feats:
            if q and q not in (f["title"] + " " + str(f.get("description", ""))).lower():
                continue
            it = QListWidgetItem(dot_icon(STATUS_COLOR.get(f.get("status"), STATUS_COLOR["proposed"]), 14),
                                 f"{f['title']}\n{f.get('status', '')}")
            it.setData(Qt.UserRole, f["id"])
            self.features.addItem(it)
        if not feats:
            self.feature_view.setHtml(f"<p style='color:{theme.c('secondary')}'>No features yet. They appear after you save a "
                                      "meeting.</p>")

    def refresh_graph(self):
        if self.project_id:
            self.graph.load_project(self.project_id)     # read + (rarely) laid out in a background thread

    # ---------- viewers ----------
    def _kind_of(self, iid: str) -> str:
        return "meeting" if self.project_id and self.ctx.store.meeting(self.project_id, iid) else "feature"

    def _on_meeting(self, cur, _prev):
        if not cur or not self.project_id or not cur.data(Qt.UserRole):
            return
        m = self.ctx.store.meeting(self.project_id, cur.data(Qt.UserRole))
        if m:
            self.meeting_view.setHtml(meeting_html(m, self.ctx.store.features(self.project_id),
                                                   self.ctx.store.meetings(self.project_id)))

    def _on_feature(self, cur, _prev):
        if not cur or not self.project_id or not cur.data(Qt.UserRole):
            return
        f = self.ctx.store.feature(self.project_id, cur.data(Qt.UserRole))
        if f:
            self.feature_view.setHtml(feature_html(f, self.ctx.store.features(self.project_id),
                                                   self.ctx.store.meetings(self.project_id)))

    def _open_link(self, url: QUrl):
        s = url.toString()
        if s.startswith("open:"):
            _, kind, iid = s.split(":", 2)
            self.open_item(kind, iid)

    def _open_from_map(self, kind: str, iid: str):
        """Double-click / Open on the Map: remember the Map's view so Alt+Left comes back to it."""
        self.nav_back.append(("map", self.graph.view_state()))
        del self.nav_back[:-20]
        self.open_item(kind, iid)
        name = self.graph.canvas.label_of(iid)
        self.toast.show_message(f"Opened “{name}”.", "Back to Map", self.go_back)

    def go_back(self):
        if not self.nav_back:
            return
        where, state = self.nav_back.pop()
        if where == "map":
            self.toast.dismiss()
            self.tabs.setCurrentIndex(TAB_MAP)
            QTimer.singleShot(0, lambda: self.graph.restore_state(state))

    def open_item(self, kind: str, iid: str):
        lst, tab = (self.meetings, TAB_MEETINGS) if kind == "meeting" else (self.features, TAB_FEATURES)
        if kind == "feature":
            self.feature_filter.clear()
        for i in range(lst.count()):
            if lst.item(i).data(Qt.UserRole) == iid:
                lst.setCurrentRow(i)
                self.tabs.setCurrentIndex(tab)
                return

    def show_log(self):
        if not self.project_id:
            return
        tb = QTextBrowser()
        tb.setPlainText(self.ctx.store.master_log(self.project_id))
        tb.resize(800, 600)
        tb.setWindowTitle("MASTER_LOG.md")
        tb.show()
        self._log_win = tb

    def rebuild_index(self):
        from ..workspace.index import ProjectIndex
        if self.project_id:
            ProjectIndex(self.ctx.store, self.project_id).rebuild()
            self.statusBar().showMessage("Search index rebuilt from saved files.", 4000)

    # ---------- adding meetings ----------
    def new_project(self):
        name, ok = QInputDialog.getText(self, "New project", "Project name:")
        if ok and name.strip():
            p = self.ctx.sm.create_project(name.strip())
            self.refresh_projects(select=p["id"])
            self.statusBar().showMessage(f"Project “{p['name']}” created. Now add its first meeting.", 6000)

    def add_meeting(self):
        d = AddMeetingDialog(self.ctx, self, self.project_id)
        if d.exec():
            self.handle_new_meeting(d.result_data)

    def handle_new_meeting(self, r: dict):
        self.refresh_projects(select=r["project_id"])
        pname = (self.ctx.store.project(r["project_id"]) or {}).get("name", "")
        if r["mode"] in ("paste", "text_file"):
            s = self.ctx.sm.import_text(r["text"], r["project_id"], pname, r["title"])
            self.select_session(s["id"])
            self.run_model_job("extract", s["id"])
        else:
            s = self.ctx.sm.import_audio(r["path"], r["project_id"], pname, r["title"])
            self.select_session(s["id"])
            self.process(s["id"])

    def open_recorder(self, project_id: str | None = None, title: str | None = None):
        """● Record: pick project + title, minimize this window, show the floating recorder."""
        pill = self.ctx.control
        if pill is None:
            return
        if pill.isVisible():
            pill.raise_()
            return
        if not getattr(self.ctx.asr, "available", False):
            box = QMessageBox(self)
            box.setWindowTitle("Speech recognition is off")
            box.setText("The recording can be made, but it can't be turned into a transcript until speech "
                        "recognition is installed.")
            box.setInformativeText("In the terminal run:  pip install -r requirements-asr.txt\n"
                                   "then restart the app. Record anyway (the audio is kept for later)?")
            go = box.addButton("Record anyway", QMessageBox.AcceptRole)
            box.addButton("Cancel", QMessageBox.RejectRole)
            box.exec()
            if box.clickedButton() is not go:
                return
        d = RecordSetupDialog(self.ctx, self, project_id or self.project_id)
        if title:
            d.title.setText(title)
        if not d.exec():
            return
        r = d.result_data
        self.refresh_projects(select=r["project_id"])
        pname = (self.ctx.store.project(r["project_id"]) or {}).get("name", "")
        self.showMinimized()
        pill.begin(r["project_id"], pname, r["title"])

    def on_recording_finished(self, sid: str, keep: bool):
        """Called when the recorder pill closes (■ stop or ✕)."""
        self.showMaximized()
        self.raise_()
        self.activateWindow()
        if keep and sid:
            self.select_session(sid)
            self.process(sid)
            if sid in self.ctx.sm.live:
                self.statusBar().showMessage("Recording saved. Finishing the last seconds of the transcript and "
                                             "the notes…", 8000)
            else:
                self.statusBar().showMessage("Recording saved. Turning it into a transcript on this laptop…", 8000)
        else:
            self.statusBar().showMessage("Recording discarded.", 5000)

    # ---------- sessions ----------
    def _on_session(self, cur, _prev):
        if cur and cur.data(Qt.UserRole):
            self._load_review(cur.data(Qt.UserRole))

    def _load_review(self, sid: str):
        self.review.load(sid)
        try:
            s = self.ctx.sm.load(sid)
            self._cur_state = (s.get("stage"), s.get("updated_at"))
        except OSError:
            self._cur_state = None

    def select_session(self, sid: str):
        self.tabs.setCurrentIndex(TAB_REVIEW)
        self._last_sig = None
        self.refresh_sessions()
        for i in range(self.sessions.count()):
            if self.sessions.item(i).data(Qt.UserRole) == sid:
                self.sessions.setCurrentRow(i)
                self._load_review(sid)
                break

    def process(self, sid: str):
        """After Stop or audio import: transcribe, then summarize, in the background."""
        def pipeline(sid=sid):
            sm = self.ctx.sm
            if sid in sm.live:                   # transcript was written while recording: finish it off
                return sm.finish_live(sid)
            s = sm.load(sid)
            if s.get("live_transcript_ok") and s["stage"] == "failed" and sm.transcript(sid).strip():
                return sm.extract(sid)           # only the notes failed; the transcript is complete
            if s.get("audio") and s["stage"] in ("recorded", "failed"):
                s = self.ctx.sm.transcribe(sid)
                if s["stage"] != "transcribed":
                    return s
            return self.ctx.sm.extract(sid)
        self._submit(sid, pipeline, model=True)

    def run_model_job(self, what: str, sid: str):
        fn = {"transcribe": self.ctx.sm.transcribe, "extract": self.ctx.sm.extract}[what]
        self._submit(sid, lambda: fn(sid), model=True)

    def _submit(self, sid: str, fn, model: bool, after=None):
        if sid in self.busy:
            return
        self.busy.add(sid)

        def done(res):
            self.busy.discard(sid)
            self._last_sig = None
            self.refresh_sessions()
            if self.review.sid == sid:
                self._load_review(sid)
            if isinstance(res, dict) and res.get("saved_transcript"):
                self.statusBar().showMessage(f"Transcript saved to {res['saved_transcript']}", 15000)
            if isinstance(res, dict) and res.get("stage") == "ready":
                self.notify("Meeting ready for review", f"{res.get('title')}: check the changes and save.")
            elif isinstance(res, dict) and res.get("stage") == "failed":
                self.notify("Something went wrong", res.get("error", "")[:200])
            if after:
                after(res)

        def err(msg):
            self.busy.discard(sid)
            QMessageBox.warning(self, "Error", msg)
            self._last_sig = None
            self.refresh_sessions()

        (self.model_pool if model else self.pool).start(Job(fn, on_done=done, on_error=err))
        self.graph.set_busy(self.models_busy())

    def handle_request(self, action: str, sid: str, payload):
        sm = self.ctx.sm
        if not sid:
            return
        if action == "save_transcript":
            sm.save_transcript(sid, payload)
        elif action in ("transcribe", "extract"):
            self.run_model_job(action, sid)
        elif action == "process":
            self.process(sid)
        elif action == "rematch":
            sm.rematch(sid)
            self._load_review(sid)
        elif action == "reject":
            sm.reject(sid)
            self._load_review(sid)
            self.statusBar().showMessage("Changes discarded. The project was not changed.", 6000)
        elif action == "discard":
            try:
                sm.discard(sid)
            except Exception as e:
                QMessageBox.warning(self, "Could not delete", str(e))
            self.review.clear()
        elif action == "cleanup":
            r = sm.cleanup(sid)
            if r["ok"]:
                self.review.clear()
                self.statusBar().showMessage("Temporary transcript and audio deleted.", 4000)
            else:
                QMessageBox.warning(self, "Still could not delete", r["error"])
        elif action == "approve":
            self._submit(sid, lambda: sm.approve(sid, payload), model=False, after=self._after_save)
        self._last_sig = None
        self.refresh_sessions()

    def _after_save(self, res):
        if not isinstance(res, dict):
            return
        if not res.get("ok"):
            QMessageBox.warning(self, "Not saved", "The project was not changed:\n\n• " +
                                "\n• ".join(res.get("errors") or ["unknown error"]))
            return
        self.review.clear()
        self.refresh_projects(select=res["project_id"])
        if not res.get("cleanup_ok"):
            QMessageBox.warning(self, "Saved, but…", "The meeting was saved, but the temporary transcript could "
                                "not be deleted. Open it under “To review” and click “Retry deleting temporary "
                                "files”.\n\n" + res.get("cleanup_error", ""))
        self.notify("Saved", "Project updated. The temporary transcript was deleted.")
        box = QMessageBox(self)
        box.setWindowTitle("Saved")
        box.setText("Saved to the project. The temporary transcript was deleted.")
        box.setInformativeText("What would you like to see next?")
        b_map = box.addButton("See the Map", QMessageBox.AcceptRole)
        b_feat = box.addButton("See Features", QMessageBox.AcceptRole)
        box.addButton("Done", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is b_map:
            self.tabs.setCurrentIndex(TAB_MAP)
        elif box.clickedButton() is b_feat:
            self.tabs.setCurrentIndex(TAB_FEATURES)

    # ---------- chat ----------
    def ask_question(self):
        q = self.ask.text().strip()
        if not q:
            return
        if not self.project_id:
            QMessageBox.information(self, "Pick a project", "Select a project on the left first.")
            return
        self.ask.clear()
        pid = self.project_id
        self.chat.append(f"<p><b>You:</b> {esc(q)}</p>")
        self.statusBar().showMessage("Thinking… (the AI is reading this project's saved files)")

        def done(res):
            self.statusBar().clearMessage()
            if pid != self.project_id:
                return
            cites = "".join(
                f"<br>[{c['n']}] <a href='open:{c['kind']}:{c['doc_id']}'>{esc(c['title'])}</a> "
                f"<span style='color:{theme.c('secondary')}'>({c['kind']})</span>" for c in res["citations"])
            body = esc(res["answer"]).replace("\n", "<br>")
            src = f"<br><span style='color:{theme.c('secondary')}'>Sources:</span>{cites}" if cites else ""
            self.chat.append(f"<p><b>Answer:</b> {body}{src}</p><hr>")

        self.model_pool.start(Job(lambda: answer(self.ctx.store, self.ctx.llm, pid, q), on_done=done,
                                  on_error=lambda m: self.chat.append(f"<p style='color:{theme.c('danger')}'>{esc(m)}</p>")))
        self.graph.set_busy(True)
