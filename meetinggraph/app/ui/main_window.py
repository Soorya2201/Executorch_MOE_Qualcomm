"""Main window: app bar (Review / Meetings / Features / Map, Ask toggle), then projects | page | Ask.

Fluent 2 look: line icons, white panels with soft shadows floating over a faint doodle wallpaper (art.py).
Sized to the screen: opens maximised and fits a 1280x752 work area (2560x1600 at 200 % scaling).
"""
from __future__ import annotations

from PySide6.QtCore import QRect, QSize, Qt, QThreadPool, QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QFontMetrics, QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (QButtonGroup, QFrame, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
                               QListView, QListWidgetItem, QMainWindow, QMenu, QMessageBox, QPushButton, QSplitter,
                               QStatusBar, QStyledItemDelegate, QTabWidget, QTextBrowser, QToolButton, QVBoxLayout,
                               QWidget)

from ..pipeline.query import answer
from .add_meeting import AddMeetingDialog, RecordSetupDialog
from .common import STAGE_COLOR, STAGE_TEXT, STATUS_COLOR, Job, dot_icon, esc, wrapping_list
from . import art, theme
from .components import Toast
from .graph_view import GraphView
from .pages import feature_html, meeting_html
from .review import ReviewPanel

BUSY = ("recording", "recorded", "transcribing", "summarizing", "saving")
TAB_REVIEW, TAB_MEETINGS, TAB_FEATURES, TAB_MAP = range(4)
NAV = ((TAB_REVIEW, "Review", "review"), (TAB_MEETINGS, "Meetings", "meetings"),
       (TAB_FEATURES, "Features", "features"), (TAB_MAP, "Map", "map"))


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
        self.setWindowTitle("MeetingGraph")
        self.setWindowIcon(art.logo_icon())
        # Never larger than the screen: this laptop's work area is 1280x752 (2560x1600 at 200 %).
        avail = QGuiApplication.primaryScreen().availableGeometry()
        self.setMinimumSize(min(960, avail.width()), min(600, avail.height()))
        self.resize(min(1480, avail.width()), min(920, avail.height()))

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.setCentralWidget(central)

        # app bar: brand · navigation · model status · more
        bar = QFrame()
        bar.setObjectName("appbar")
        bar.setFixedHeight(52)
        al = QHBoxLayout(bar)
        al.setContentsMargins(16, 0, 12, 0)
        al.setSpacing(4)
        logo = QLabel()
        logo.setPixmap(art.logo_pixmap(26))
        al.addWidget(logo)
        al.addSpacing(6)
        brand = QLabel("MeetingGraph")
        brand.setObjectName("brand")
        al.addWidget(brand)
        al.addSpacing(24)
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav: dict[int, QPushButton] = {}
        for idx, text, ic in NAV:
            b = QPushButton(text)
            b.setObjectName("nav")
            b.setIcon(art.icon(ic))
            b.setIconSize(QSize(18, 18))
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, i=idx: self.go(i))
            self.nav_group.addButton(b)
            self.nav[idx] = b
            al.addWidget(b)
        al.addStretch(1)
        self.b_ask_toggle = QToolButton()
        self.b_ask_toggle.setObjectName("appTool")
        self.b_ask_toggle.setText("Ask")
        self.b_ask_toggle.setIcon(art.icon("ask"))
        self.b_ask_toggle.setIconSize(QSize(18, 18))
        self.b_ask_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.b_ask_toggle.setCheckable(True)
        self.b_ask_toggle.setChecked(True)
        self.b_ask_toggle.setCursor(Qt.PointingHandCursor)
        self.b_ask_toggle.setToolTip("Show or hide the Ask panel")
        self.b_ask_toggle.toggled.connect(lambda _on: self._sync_nav(self.tabs.currentIndex()))
        al.addWidget(self.b_ask_toggle)
        more = QToolButton()
        more.setIcon(art.icon("more"))
        more.setIconSize(QSize(18, 18))
        more.setObjectName("more")
        more.setPopupMode(QToolButton.InstantPopup)
        more.setToolTip("More")
        m = QMenu(more)
        for text, fn in (("Open workspace folder", lambda: QDesktopServices.openUrl(
                              QUrl.fromLocalFile(str(ctx.layout.root)))),
                         ("Show master log of this project", self.show_log),
                         ("Rebuild search index", self.rebuild_index)):
            a = QAction(text, self)
            a.triggered.connect(fn)
            m.addAction(a)
        more.setMenu(m)
        al.addWidget(more)
        outer.addWidget(bar)

        work = self.backdrop = art.Backdrop()
        cl = QVBoxLayout(work)
        cl.setContentsMargins(14, 14, 14, 10)
        cl.setSpacing(12)
        outer.addWidget(work, 1)

        root = QSplitter()
        root.setChildrenCollapsible(False)
        cl.addWidget(root, 1)

        # left: projects + main actions
        left = QFrame()
        left.setObjectName("sidebar")
        left.setMinimumWidth(200)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(14, 14, 14, 14)
        ll.setSpacing(8)
        self.b_add = QPushButton("Add a transcript")
        self.b_add.setObjectName("primary")
        self.b_add.setIcon(art.icon("add", "#ffffff", "#ffffff"))
        self.b_add.setMinimumHeight(36)
        ll.addWidget(self.b_add)
        self.b_record = QPushButton("Record")
        self.b_record.setObjectName("record")
        self.b_record.setIcon(art.icon("record", "#ffffff", "#ffffff"))
        self.b_record.setMinimumHeight(36)
        self.b_record.setToolTip("Record a meeting: your microphone + the computer's sound")
        ll.addWidget(self.b_record)
        ll.addSpacing(10)
        lbl = QLabel("PROJECTS")
        lbl.setObjectName("eyebrow")
        ll.addWidget(lbl)
        self.projects = wrapping_list()
        self.projects.setObjectName("bare")
        ll.addWidget(self.projects, 1)
        self.b_new = QPushButton("New project")
        self.b_new.setObjectName("ghost")
        self.b_new.setIcon(art.icon("add", theme.TEXT_2, theme.TEXT_2))
        ll.addWidget(self.b_new)
        root.addWidget(left)

        # center: pages (the tab bar is hidden; the app bar navigates)
        self.tabs = QTabWidget()
        self.tabs.tabBar().hide()
        rev = QWidget()
        rl = QHBoxLayout(rev)
        rl.setContentsMargins(0, 0, 0, 0)
        rsplit = QSplitter()
        rsplit.setChildrenCollapsible(False)
        rl.addWidget(rsplit)
        sl = QFrame()
        sl.setObjectName("panel")
        sl.setMinimumWidth(200)
        sll = QVBoxLayout(sl)
        sll.setContentsMargins(12, 14, 12, 12)
        t = QLabel("WAITING FOR YOU")
        t.setObjectName("eyebrow")
        sll.addWidget(t)
        self.sessions = wrapping_list()
        self.sessions.setObjectName("bare")
        sll.addWidget(self.sessions, 1)
        self.review = ReviewPanel(ctx)
        rsplit.addWidget(sl)
        rsplit.addWidget(self.review)
        rsplit.setSizes([250, 900])
        self.tabs.addTab(rev, "Review")

        def list_and_view(lst, view, extra=None):
            page = QWidget()
            pl = QHBoxLayout(page)
            pl.setContentsMargins(0, 0, 0, 0)
            split = QSplitter()
            split.setChildrenCollapsible(False)
            pl.addWidget(split)
            lcard = QFrame()
            lcard.setObjectName("panel")
            lcard.setMinimumWidth(200)
            self.backdrop.add_panels(lcard, view)
            lcl = QVBoxLayout(lcard)
            lcl.setContentsMargins(12, 12, 12, 12)
            if extra is not None:
                lcl.addWidget(extra)
            lst.setObjectName("bare")
            lcl.addWidget(lst, 1)
            split.addWidget(lcard)
            split.addWidget(view)
            split.setSizes([250, 800])
            return page

        self.meetings = wrapping_list()
        self.meeting_view = QTextBrowser()
        self.meeting_view.setOpenLinks(False)
        self.tabs.addTab(list_and_view(self.meetings, self.meeting_view), "Meetings")

        self.feature_filter = QLineEdit()
        self.feature_filter.setPlaceholderText("Search features…")
        self.feature_filter.setClearButtonEnabled(True)
        self.features = wrapping_list()
        self.feature_view = QTextBrowser()
        self.feature_view.setOpenLinks(False)
        self.tabs.addTab(list_and_view(self.features, self.feature_view, self.feature_filter), "Features")

        self.graph = GraphView()                      # = map_view.MapView (RP-1106's Map)
        map_panel = QFrame()
        map_panel.setObjectName("panel")
        mpl = QVBoxLayout(map_panel)
        mpl.setContentsMargins(1, 1, 1, 1)
        mpl.addWidget(self.graph)
        self.tabs.addTab(map_panel, "Map")
        root.addWidget(self.tabs)

        # right: ask
        right = self.ask_panel = QFrame()
        right.setObjectName("panel")
        right.setMinimumWidth(260)
        rr = QVBoxLayout(right)
        rr.setContentsMargins(16, 16, 16, 14)
        rr.setSpacing(8)
        self.chat_title = QLabel("Ask a question")
        self.chat_title.setObjectName("h2")
        self.chat_title.setWordWrap(True)
        self.pending = QLabel("")
        self.pending.setWordWrap(True)
        self.pending.setObjectName("warnText")
        self.chat = QTextBrowser()
        self.chat.setObjectName("bare")
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
        note.setObjectName("small")
        rr.addWidget(note)
        root.addWidget(right)
        root.setSizes([230, 900, 300])
        self.backdrop.add_panels(left, sl, right, map_panel)

        sb = QStatusBar()
        sb.setSizeGripEnabled(False)
        self.setStatusBar(sb)
        if getattr(ctx, "llm_problem", ""):
            warn = QLabel(f"AI model not available, using simple rules: {ctx.llm_problem}")
            warn.setObjectName("warnText")
            warn.setToolTip(ctx.llm_problem)
            sb.addWidget(warn, 1)
        self.tabs.currentChanged.connect(self._sync_nav)
        self.go(TAB_REVIEW)

        # wiring
        self.projects.currentItemChanged.connect(self._on_project)
        self.sessions.currentItemChanged.connect(self._on_session)
        self.meetings.currentItemChanged.connect(self._on_meeting)
        self.features.currentItemChanged.connect(self._on_feature)
        self.feature_filter.textChanged.connect(self.refresh_features)
        for v in (self.meeting_view, self.feature_view, self.chat):
            v.anchorClicked.connect(self._open_link)
        self.graph.open_node.connect(self._open_from_map)
        self.graph.go_meetings.connect(lambda: self.go(TAB_MEETINGS))
        self.graph.set_context(ctx.store, ctx.layout.root)
        self.nav_back: list = []                     # where Alt+Left returns to (after opening a node from the Map)
        self.toast = Toast(self)
        keys = [QKeySequence(Qt.ALT | Qt.Key_Left)]
        keys += [k for k in QKeySequence.keyBindings(QKeySequence.Back) if k not in keys]
        self._back_sc = QShortcut(self)
        self._back_sc.setKeys(keys)
        self._back_sc.setContext(Qt.WindowShortcut)
        self._back_sc.activated.connect(self.go_back)
        self.review.request.connect(self.handle_request)
        self.b_add.clicked.connect(self.add_meeting)
        self.b_new.clicked.connect(self.new_project)
        self.b_record.clicked.connect(self.open_recorder)
        b_ask.clicked.connect(self.ask_question)
        self.ask.returnPressed.connect(self.ask_question)

        self.poll = QTimer(self)
        self.poll.timeout.connect(self.refresh_sessions)
        self.poll.start(1500)
        self.refresh_all()

    # ---------- navigation ----------
    def go(self, idx: int):
        self.tabs.setCurrentIndex(idx)
        self._sync_nav(idx)

    def _sync_nav(self, idx: int):
        if idx in self.nav:
            self.nav[idx].setChecked(True)
        self.ask_panel.setVisible(idx != TAB_REVIEW and self.b_ask_toggle.isChecked())
        self.b_ask_toggle.setEnabled(idx != TAB_REVIEW)

    def bring_to_front(self):
        """Restore from minimised/hidden and focus (tray, second launch, notifications)."""
        if self.isMinimized():
            self.showNormal()
        else:
            self.show()
        self.raise_()
        self.activateWindow()

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

    def refresh_sessions(self):
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
            it = QListWidgetItem(dot_icon(STAGE_COLOR.get(stage, "#8c959f"), 14),
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
        self.nav[TAB_REVIEW].setText(f"Review  ·  {ready}" if ready else "Review")
        if cur and not any(s["id"] == cur for s in sessions):
            self.review.clear()
        elif changed_current:
            self._load_review(cur)
        n = sum(1 for s in sessions if s.get("project_id") == self.project_id)
        self.pending.setText(f"{n} meeting(s) for this project are not saved yet, so answers don't include them."
                             if n else "")

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
            self.meeting_view.setHtml("<p style='color:#667085'>No saved meetings yet. Add one and save it from "
                                      "“Review”.</p>")

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
            it = QListWidgetItem(dot_icon(STATUS_COLOR.get(f.get("status"), "#999"), 14),
                                 f"{f['title']}\n{f.get('status', '')}")
            it.setData(Qt.UserRole, f["id"])
            self.features.addItem(it)
        if not feats:
            self.feature_view.setHtml("<p style='color:#667085'>No features yet. They appear after you save a "
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
            saved = self.ctx.sm.transcript_path(self.project_id, m["id"]).exists()
            self.meeting_view.setHtml(meeting_html(m, self.ctx.store.features(self.project_id),
                                                   self.ctx.store.meetings(self.project_id), saved))

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
            if kind == "transcript":
                self.show_transcript(iid)
            else:
                self.open_item(kind, iid)

    def open_item(self, kind: str, iid: str):
        lst, tab = (self.meetings, TAB_MEETINGS) if kind == "meeting" else (self.features, TAB_FEATURES)
        if kind == "feature":
            self.feature_filter.clear()
        for i in range(lst.count()):
            if lst.item(i).data(Qt.UserRole) == iid:
                lst.setCurrentRow(i)
                self.go(tab)
                return

    def models_busy(self) -> bool:
        """True while the AI is answering, transcribing or making notes (the Map waits with its layout)."""
        return self.model_pool.activeThreadCount() > 0 or bool(self.ctx.sm.live)

    def _open_from_map(self, kind: str, iid: str):
        """Double-click / Open on the Map: remember the Map's view so Alt+Left (or the toast) comes back to it."""
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
            self.go(TAB_MAP)
            QTimer.singleShot(0, lambda: self.graph.restore_state(state))

    def show_transcript(self, meeting_id: str):
        if not self.project_id:
            return
        p = self.ctx.sm.transcript_path(self.project_id, meeting_id)
        m = self.ctx.store.meeting(self.project_id, meeting_id) or {}
        tb = QTextBrowser()
        tb.setPlainText(p.read_text(encoding="utf-8") if p.exists() else "This transcript was not kept.")
        tb.resize(820, 640)
        tb.setWindowTitle(f"Transcript — {m.get('title', meeting_id)}")
        tb.show()
        self._transcript_win = tb

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
        self.ctx.llm.warm_up()                  # notes are about to be made: load the NPU model now
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
        self.go(TAB_REVIEW)
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
            QTimer.singleShot(0, lambda: self.graph.set_busy(self.models_busy()))
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
            QMessageBox.warning(self, "Saved, but…", "The meeting was saved, but its temporary files could not "
                                "be cleaned up. Open it under “Review” and click “Retry deleting temporary "
                                "files”.\n\n" + res.get("cleanup_error", ""))
        kept = bool(self.ctx.cfg["audio"].get("keep_transcript_after_commit", True))
        what = "The full transcript is kept with the meeting." if kept else "The temporary transcript was deleted."
        self.notify("Saved", f"Project updated. {what}")
        box = QMessageBox(self)
        box.setWindowTitle("Saved")
        box.setText(f"Saved to the project. {what}")
        box.setInformativeText("What would you like to see next?")
        b_map = box.addButton("See the Map", QMessageBox.AcceptRole)
        b_feat = box.addButton("See Features", QMessageBox.AcceptRole)
        box.addButton("Done", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is b_map:
            self.go(TAB_MAP)
        elif box.clickedButton() is b_feat:
            self.go(TAB_FEATURES)

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
                f"<span style='color:#667085'>({c['kind']})</span>" for c in res["citations"])
            body = esc(res["answer"]).replace("\n", "<br>")
            src = f"<br><span style='color:#667085'>Sources:</span>{cites}" if cites else ""
            self.chat.append(f"<p><b>Answer:</b> {body}{src}</p><hr>")

        self.model_pool.start(Job(lambda: answer(self.ctx.store, self.ctx.llm, pid, q), on_done=done,
                                  on_error=lambda m: self.chat.append(f"<p style='color:#d92d20'>{esc(m)}</p>")))
