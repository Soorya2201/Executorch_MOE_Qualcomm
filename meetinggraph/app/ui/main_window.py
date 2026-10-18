"""Main window: app bar (To review / Meetings / Features / Map), then projects | page | Ask.

A banner always says what to do next.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QThreadPool, QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (QButtonGroup, QFrame, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMainWindow, QMenu, QMessageBox, QPushButton, QSplitter,
                               QStatusBar, QTabWidget, QTextBrowser, QToolButton, QVBoxLayout,
                               QWidget)

from ..pipeline.query import answer
from ..workspace.graph import build_graph
from .add_meeting import AddMeetingDialog
from .common import STAGE_COLOR, STAGE_TEXT, STATUS_COLOR, Job, dot_icon, esc
from . import theme
from .graph_view import GraphView
from .pages import feature_html, meeting_html
from .review import ReviewPanel

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
        self.setWindowTitle("MeetingGraph")
        self.resize(1480, 920)
        self.setMinimumSize(1100, 700)

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.setCentralWidget(central)

        # app bar: brand · navigation · model status · more
        bar = QFrame()
        bar.setObjectName("appbar")
        bar.setFixedHeight(60)
        al = QHBoxLayout(bar)
        al.setContentsMargins(20, 0, 16, 0)
        al.setSpacing(6)
        logo = QLabel()
        logo.setPixmap(dot_icon(theme.ACCENT, 28).pixmap(22, 22))
        al.addWidget(logo)
        brand = QLabel("MeetingGraph")
        brand.setObjectName("brand")
        al.addWidget(brand)
        al.addSpacing(28)
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav: dict[int, QPushButton] = {}
        for idx, text in ((TAB_REVIEW, "To review"), (TAB_MEETINGS, "Meetings"), (TAB_FEATURES, "Features"),
                          (TAB_MAP, "Map")):
            b = QPushButton(text)
            b.setObjectName("nav")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, i=idx: self.go(i))
            self.nav_group.addButton(b)
            self.nav[idx] = b
            al.addWidget(b)
        al.addStretch(1)
        self.chip_llm = self._chip(f"AI  ·  {ctx.llm.label}", bool(getattr(ctx, "llm_problem", "")))
        self.chip_asr = self._chip(f"Speech  ·  {ctx.asr.label}", False)
        for chip in (self.chip_llm, self.chip_asr, self._chip("On this laptop only", False, good=True)):
            al.addWidget(chip, 0, Qt.AlignVCenter)
        al.addSpacing(6)
        more = QToolButton()
        more.setText("•••")
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

        work = QWidget()
        cl = QVBoxLayout(work)
        cl.setContentsMargins(16, 14, 16, 8)
        cl.setSpacing(12)
        outer.addWidget(work, 1)

        # banner: what to do next
        self.banner = QFrame()
        self.banner.setObjectName("banner")
        bl = QHBoxLayout(self.banner)
        bl.setContentsMargins(18, 12, 12, 12)
        self.banner_text = QLabel()
        self.banner_text.setWordWrap(True)
        self.banner_text.setTextFormat(Qt.RichText)
        self.banner_btn = QPushButton("Show me")
        self.banner_btn.setObjectName("primary")
        bl.addWidget(self.banner_text, 1)
        bl.addWidget(self.banner_btn)
        cl.addWidget(self.banner)

        root = QSplitter()
        root.setChildrenCollapsible(False)
        cl.addWidget(root, 1)

        # left: projects + main actions
        left = QFrame()
        left.setObjectName("sidebar")
        left.setMinimumWidth(220)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(14, 14, 14, 14)
        ll.setSpacing(8)
        self.b_add = QPushButton("＋  Add a meeting")
        self.b_add.setObjectName("primary")
        self.b_add.setMinimumHeight(40)
        ll.addWidget(self.b_add)
        self.b_record = QPushButton("●  Record a live meeting")
        ll.addWidget(self.b_record)
        ll.addSpacing(10)
        lbl = QLabel("PROJECTS")
        lbl.setObjectName("eyebrow")
        ll.addWidget(lbl)
        self.projects = QListWidget()
        self.projects.setObjectName("bare")
        ll.addWidget(self.projects, 1)
        self.b_new = QPushButton("＋  New project")
        self.b_new.setObjectName("ghost")
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
        sl.setMinimumWidth(220)
        sll = QVBoxLayout(sl)
        sll.setContentsMargins(12, 14, 12, 12)
        t = QLabel("WAITING FOR YOU")
        t.setObjectName("eyebrow")
        sll.addWidget(t)
        self.sessions = QListWidget()
        self.sessions.setObjectName("bare")
        self.sessions.setWordWrap(True)
        self.sessions.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sll.addWidget(self.sessions, 1)
        self.review = ReviewPanel(ctx)
        rsplit.addWidget(sl)
        rsplit.addWidget(self.review)
        rsplit.setSizes([260, 900])
        self.tabs.addTab(rev, "To review")

        def list_and_view(lst, view, extra=None):
            page = QWidget()
            pl = QHBoxLayout(page)
            pl.setContentsMargins(0, 0, 0, 0)
            split = QSplitter()
            split.setChildrenCollapsible(False)
            pl.addWidget(split)
            lcard = QFrame()
            lcard.setObjectName("panel")
            lcard.setMinimumWidth(220)
            lcl = QVBoxLayout(lcard)
            lcl.setContentsMargins(12, 12, 12, 12)
            if extra is not None:
                lcl.addWidget(extra)
            lst.setObjectName("bare")
            lcl.addWidget(lst, 1)
            split.addWidget(lcard)
            split.addWidget(view)
            split.setSizes([260, 800])
            return page

        self.meetings = QListWidget()
        self.meeting_view = QTextBrowser()
        self.meeting_view.setOpenLinks(False)
        self.tabs.addTab(list_and_view(self.meetings, self.meeting_view), "Meetings")

        self.feature_filter = QLineEdit()
        self.feature_filter.setPlaceholderText("Search features…")
        self.feature_filter.setClearButtonEnabled(True)
        self.features = QListWidget()
        self.feature_view = QTextBrowser()
        self.feature_view.setOpenLinks(False)
        self.tabs.addTab(list_and_view(self.features, self.feature_view, self.feature_filter), "Features")

        self.graph = GraphView()
        self.tabs.addTab(self.graph, "Map")
        root.addWidget(self.tabs)

        # right: ask
        right = QFrame()
        right.setObjectName("panel")
        right.setMinimumWidth(280)
        rr = QVBoxLayout(right)
        rr.setContentsMargins(16, 16, 16, 14)
        rr.setSpacing(8)
        self.chat_title = QLabel("Ask a question")
        self.chat_title.setObjectName("h2")
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
        root.setSizes([240, 900, 330])

        sb = QStatusBar()
        sb.setSizeGripEnabled(False)
        self.setStatusBar(sb)
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
        self.graph.open_node.connect(self.open_item)
        self.review.request.connect(self.handle_request)
        self.b_add.clicked.connect(self.add_meeting)
        self.b_new.clicked.connect(self.new_project)
        self.b_record.clicked.connect(self.open_recorder)
        self.banner_btn.clicked.connect(lambda: self._banner_action and self._banner_action())
        b_ask.clicked.connect(self.ask_question)
        self.ask.returnPressed.connect(self.ask_question)

        self.poll = QTimer(self)
        self.poll.timeout.connect(self.refresh_sessions)
        self.poll.start(1500)
        self.refresh_all()

    # ---------- navigation ----------
    def _chip(self, text: str, warn: bool, good: bool = False) -> QFrame:
        fr = QFrame()
        fr.setObjectName("chipGood" if good else "chipWarn" if warn else "chip")
        fr.setFixedHeight(28)
        lay = QHBoxLayout(fr)
        lay.setContentsMargins(10, 4, 12, 4)
        lay.setSpacing(6)
        dot = QLabel()
        dot.setPixmap(dot_icon(theme.GOOD if good else theme.WARN if warn else theme.ACCENT, 12).pixmap(8, 8))
        lb = QLabel(text)
        lb.setObjectName("chipGoodText" if good else "chipText")
        lay.addWidget(dot)
        lay.addWidget(lb)
        if warn:
            fr.setToolTip(f"AI model not available: {getattr(self.ctx, 'llm_problem', '')}")
        return fr

    def go(self, idx: int):
        self.tabs.setCurrentIndex(idx)
        self.nav[idx].setChecked(True)

    def _sync_nav(self, idx: int):
        if idx in self.nav:
            self.nav[idx].setChecked(True)

    def bring_to_front(self):
        """Restore from minimised/hidden and focus (tray, second launch, notifications)."""
        if self.isMinimized():
            self.showNormal()
        else:
            self.show()
        self.raise_()
        self.activateWindow()

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
            text = ("<b>Start here:</b> click <b>＋ Add a meeting</b>. Create a project, then paste a conversation or "
                    "open a transcript file. To try it, use <i>tests\\fixtures\\atlas_01_kickoff.txt</i>.")
            btn, act = "Add a meeting", self.add_meeting
        else:
            name = next((p["name"] for p in projects if p["id"] == self.project_id), projects[0]["name"])
            text = (f"<b>All caught up.</b> Add the next meeting for “{esc(name)}”, look through Meetings, "
                    "Features and the Map, or ask a question on the right.")
            btn, act = "Add a meeting", self.add_meeting
        if llm_problem:
            text += f"<br><span style='color:{theme.WARN}'>AI model not available: {esc(llm_problem)} " \
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
            it = QListWidgetItem("Nothing to review.\nUse “＋ Add a meeting”.")
            it.setFlags(Qt.NoItemFlags)
            self.sessions.addItem(it)
        self.sessions.blockSignals(False)
        self.nav[TAB_REVIEW].setText(f"To review  ·  {ready}" if ready else "To review")
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
            self.graph.show_graph({"nodes": [], "edges": []})
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
            it = QListWidgetItem(dot_icon(STATUS_COLOR.get(f.get("status"), "#999"), 14),
                                 f"{f['title']}\n{f.get('status', '')}")
            it.setData(Qt.UserRole, f["id"])
            self.features.addItem(it)
        if not feats:
            self.feature_view.setHtml("<p style='color:#667085'>No features yet. They appear after you save a "
                                      "meeting.</p>")

    def refresh_graph(self):
        if self.project_id:
            self.graph.show_graph(build_graph(self.ctx.store, self.project_id))

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

    def open_item(self, kind: str, iid: str):
        lst, tab = (self.meetings, TAB_MEETINGS) if kind == "meeting" else (self.features, TAB_FEATURES)
        if kind == "feature":
            self.feature_filter.clear()
        for i in range(lst.count()):
            if lst.item(i).data(Qt.UserRole) == iid:
                lst.setCurrentRow(i)
                self.go(tab)
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
        if r["mode"] == "record":
            self.open_recorder(r["project_id"], r["title"])
            return
        if r["mode"] in ("paste", "text_file"):
            s = self.ctx.sm.import_text(r["text"], r["project_id"], pname, r["title"])
            self.select_session(s["id"])
            self.run_model_job("extract", s["id"])
        else:
            s = self.ctx.sm.import_audio(r["path"], r["project_id"], pname, r["title"])
            self.select_session(s["id"])
            self.process(s["id"])

    def open_recorder(self, project_id: str | None = None, title: str | None = None):
        c = self.ctx.control
        if not c:
            return
        if not c.recording:
            c.refresh()
            if project_id or self.project_id:
                c.select_project(project_id or self.project_id)
            if title:
                c.title.setText(title)
        c.show()
        c.raise_()

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
            s = self.ctx.sm.load(sid)
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
